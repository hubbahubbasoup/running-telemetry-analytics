from datetime import datetime
import os
import sqlite3
import numpy as np
import pandas as pd

DB_NAME = 'training_vault.db'

STANDARD_DISTANCES_M = {
    '400m': 400.0,
    '800m': 800.0,
    '1000m': 1000.0,
    '1 Mile': 1609.34,
    '2 Mile': 3218.68,
    '5K': 5000.0,
    '10K': 10000.0,
    'Half Marathon': 21097.5,
    'Marathon': 42195.0,
}


def format_seconds_to_time(seconds):
  """Formats raw seconds into clean MM:SS or Xh Ym Zs strings."""
  if seconds is None or pd.isna(seconds) or seconds <= 0:
    return 'N/A'
  total_sec = int(round(seconds))
  hours = total_sec // 3600
  mins = (total_sec % 3600) // 60
  secs = total_sec % 60
  if hours > 0:
    return f'{hours}h {mins:02d}m {secs:02d}s'
  return f'{mins}m {secs:02d}s'


def format_pace_per_mile(seconds, dist_m):
  """Formats pace per mile (MM:SS /mi) given duration and distance in meters."""
  if not seconds or not dist_m or seconds <= 0 or dist_m <= 0:
    return 'N/A'
  sec_per_mile = (seconds / dist_m) * 1609.34
  p_min = int(sec_per_mile // 60)
  p_sec = int(round(sec_per_mile % 60))
  if p_sec == 60:
    p_min += 1
    p_sec = 0
  return f'{p_min}:{p_sec:02d} /mi'


def init_baselines_table(db_path=DB_NAME, default_max=190.0, default_rest=50.0):
  """Creates athlete_baselines and best_efforts tables in SQLite."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS athlete_baselines (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          hr_max REAL,
          hr_rest REAL,
          peak_source_file TEXT,
          last_updated TEXT
      )
  ''')
  row = cursor.execute(
      'SELECT hr_max, hr_rest FROM athlete_baselines WHERE id = 1'
  ).fetchone()
  if not row:
    cursor.execute(
        '''
        INSERT INTO athlete_baselines (id, hr_max, hr_rest, peak_source_file, last_updated)
        VALUES (1, ?, ?, 'Default Baseline', ?)
        ''',
        (default_max, default_rest, datetime.now().strftime('%Y-%m-%d %H:%M')),
    )

  cursor.execute('''
      CREATE TABLE IF NOT EXISTS best_efforts (
          file_name TEXT,
          distance_label TEXT,
          distance_meters REAL,
          duration_seconds REAL,
          activity_date TEXT,
          PRIMARY KEY (file_name, distance_label)
      )
  ''')
  conn.commit()
  conn.close()


def extract_best_efforts_from_df(df_records):
  """Runs sliding-window search across a cleaned/raw records DataFrame

  and returns a list of dicts with fastest times for standard distances.
  """
  if df_records.empty or 'distance' not in df_records.columns:
    return []

  df = df_records.copy()
  if 'datetime' not in df.columns:
    if 'timestamp' in df.columns:
      df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
      return []

  df = df.dropna(subset=['distance', 'datetime']).sort_values('datetime')
  if df.empty:
    return []

  distances = df['distance'].values
  times = pd.to_datetime(df['datetime']).values.astype('datetime64[ns]')
  max_dist = distances[-1] - distances[0]

  efforts = []
  for label, target_dist in STANDARD_DISTANCES_M.items():
    if max_dist < target_dist:
      continue

    best_time_sec = float('inf')
    n = len(df)
    left = 0
    for right in range(n):
      while (distances[right] - distances[left]) >= target_dist:
        dur_sec = float((times[right] - times[left]) / np.timedelta64(1, 's'))
        if dur_sec > 0 and dur_sec < best_time_sec:
          best_time_sec = dur_sec
        left += 1

    # Basic sanity filter against GPS teleportation glitches (> 10.5 m/s world record sprint)
    if best_time_sec != float('inf') and (target_dist / best_time_sec) <= 10.5:
      efforts.append({
          'distance_label': label,
          'distance_meters': target_dist,
          'duration_seconds': round(best_time_sec, 1),
          'Time': format_seconds_to_time(best_time_sec),
          'Pace': format_pace_per_mile(best_time_sec, target_dist),
      })

  return efforts


def save_best_efforts_to_db(
    file_name, activity_date, efforts_list, db_path=DB_NAME
):
  """Upserts a run's best efforts into the SQLite best_efforts table."""
  if not efforts_list:
    return
  init_baselines_table(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  for e in efforts_list:
    cursor.execute(
        '''
        INSERT INTO best_efforts (file_name, distance_label, distance_meters, duration_seconds, activity_date)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(file_name, distance_label) DO UPDATE SET
            duration_seconds = MIN(best_efforts.duration_seconds, excluded.duration_seconds),
            activity_date = excluded.activity_date
        ''',
        (
            file_name,
            e['distance_label'],
            e['distance_meters'],
            e['duration_seconds'],
            str(activity_date)[:10],
        ),
    )
  conn.commit()
  conn.close()


def get_all_time_prs(db_path=DB_NAME):
  """Returns a DataFrame of all-time fastest recorded PRs across all files in SQLite."""
  init_baselines_table(db_path)
  conn = sqlite3.connect(db_path)
  query = '''
      SELECT
          b.distance_label,
          b.distance_meters,
          b.duration_seconds,
          b.file_name,
          b.activity_date
      FROM best_efforts b
      INNER JOIN (
          SELECT distance_label, MIN(duration_seconds) AS min_sec
          FROM best_efforts
          GROUP BY distance_label
      ) grouped
      ON b.distance_label = grouped.distance_label
      AND b.duration_seconds = grouped.min_sec
      GROUP BY b.distance_label
      ORDER BY b.distance_meters ASC
  '''
  df_prs = pd.read_sql(query, conn)
  conn.close()

  if not df_prs.empty:
    df_prs['Recorded PR Time'] = df_prs['duration_seconds'].apply(
        format_seconds_to_time
    )
    df_prs['Recorded PR Pace'] = df_prs.apply(
        lambda r: format_pace_per_mile(
            r['duration_seconds'], r['distance_meters']
        ),
        axis=1,
    )
  return df_prs


def get_athlete_baselines(db_path=DB_NAME):
  """Returns the active (hr_max, hr_rest, peak_source_file, last_updated) from SQLite."""
  init_baselines_table(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  tables = {
      r[0]
      for r in cursor.execute(
          "SELECT name FROM sqlite_master WHERE type='table'"
      ).fetchall()
  }
  if 'activities' in tables:
    cols = {
        r[1]
        for r in cursor.execute('PRAGMA table_info(activities)').fetchall()
    }
    if 'max_heart_rate' in cols:
      peak_row = cursor.execute('''
          SELECT max_heart_rate, file_name
          FROM activities
          WHERE max_heart_rate IS NOT NULL AND max_heart_rate <= 220
          ORDER BY max_heart_rate DESC
          LIMIT 1
      ''').fetchone()
      if peak_row and peak_row[0]:
        curr_max = cursor.execute(
            'SELECT hr_max FROM athlete_baselines WHERE id = 1'
        ).fetchone()[0]
        if float(peak_row[0]) > float(curr_max):
          cursor.execute(
              '''
              UPDATE athlete_baselines
              SET hr_max = ?, peak_source_file = ?, last_updated = ?
              WHERE id = 1
              ''',
              (
                  float(peak_row[0]),
                  peak_row[1],
                  datetime.now().strftime('%Y-%m-%d %H:%M'),
              ),
          )
          conn.commit()

  row = cursor.execute(
      'SELECT hr_max, hr_rest, peak_source_file, last_updated FROM athlete_baselines WHERE id = 1'
  ).fetchone()
  conn.close()
  return float(row[0]), float(row[1]), row[2], row[3]


def recalculate_vault_trimp(db_path=DB_NAME):
  """Recalculates Banister TRIMP across all stored activities when HR baselines change."""
  if not os.path.exists(db_path):
    return
  hr_max, hr_rest, _, _ = get_athlete_baselines(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  tables = {
      r[0]
      for r in cursor.execute(
          "SELECT name FROM sqlite_master WHERE type='table'"
      ).fetchall()
  }
  if 'activities' in tables:
    rows = cursor.execute(
        'SELECT file_name, duration_minutes, avg_heart_rate FROM activities'
    ).fetchall()
    for fname, dur_min, avg_hr in rows:
      new_trimp = compute_banister_trimp(
          dur_min or 0.0, avg_hr, hr_max, hr_rest
      )
      cursor.execute(
          'UPDATE activities SET relative_effort_trimp = ? WHERE file_name = ?',
          (new_trimp, fname),
      )
    conn.commit()
  conn.close()


def update_athlete_baselines(
    db_path=DB_NAME,
    candidate_max_hr=None,
    source_file=None,
    new_hr_rest=None,
    manual_hr_max=None,
):
  """Updates HR baselines if a new peak HR is observed or manual settings are saved."""
  hr_max, hr_rest, peak_file, _ = get_athlete_baselines(db_path)
  changed = False

  if (
      candidate_max_hr is not None
      and 100 < candidate_max_hr <= 220
      and candidate_max_hr > hr_max
  ):
    hr_max = round(float(candidate_max_hr), 1)
    peak_file = source_file or 'Uploaded Activity'
    changed = True

  if manual_hr_max is not None and float(manual_hr_max) != hr_max:
    hr_max = round(float(manual_hr_max), 1)
    peak_file = 'Manual Override'
    changed = True

  if new_hr_rest is not None and float(new_hr_rest) != hr_rest:
    hr_rest = round(float(new_hr_rest), 1)
    changed = True

  if changed:
    conn = sqlite3.connect(db_path)
    conn.execute(
        '''
        UPDATE athlete_baselines
        SET hr_max = ?, hr_rest = ?, peak_source_file = ?, last_updated = ?
        WHERE id = 1
        ''',
        (hr_max, hr_rest, peak_file, datetime.now().strftime('%Y-%m-%d %H:%M')),
    )
    conn.commit()
    conn.close()
    recalculate_vault_trimp(db_path)

  return hr_max, hr_rest, changed


def calculate_karvonen_zones(hr_max, hr_rest):
  """Computes 5-zone heart rate boundaries using the Karvonen HRR formula."""
  hrr = hr_max - hr_rest
  return {
      'Zone 1 (Recovery)': (
          round(hr_rest + (0.50 * hrr)),
          round(hr_rest + (0.60 * hrr)),
      ),
      'Zone 2 (Endurance)': (
          round(hr_rest + (0.60 * hrr)),
          round(hr_rest + (0.70 * hrr)),
      ),
      'Zone 3 (Tempo)': (
          round(hr_rest + (0.70 * hrr)),
          round(hr_rest + (0.80 * hrr)),
      ),
      'Zone 4 (Threshold)': (
          round(hr_rest + (0.80 * hrr)),
          round(hr_rest + (0.90 * hrr)),
      ),
      'Zone 5 (VO2 Max)': (
          round(hr_rest + (0.90 * hrr)),
          round(hr_max),
      ),
  }


def compute_time_in_zones(df_records, zones):
  """Calculates exact minutes and percentages spent in each Karvonen HR zone."""
  zone_keys = [
      'Zone 1 (Recovery)',
      'Zone 2 (Endurance)',
      'Zone 3 (Tempo)',
      'Zone 4 (Threshold)',
      'Zone 5 (VO2 Max)',
  ]
  empty_df = pd.DataFrame({
      'Zone': zone_keys,
      'HR Range': [f'{zones[k][0]}-{zones[k][1]} bpm' for k in zone_keys],
      'Minutes': [0.0] * 5,
      'Percentage': [0.0] * 5,
  })

  if (
      df_records.empty
      or 'heart_rate' not in df_records.columns
      or not df_records['heart_rate'].notna().any()
  ):
    return empty_df

  df = df_records.copy()
  if 'dt' not in df.columns:
    if 'datetime' in df.columns:
      df['dt'] = df['datetime'].diff().dt.total_seconds().fillna(1.0)
    elif 'timestamp' in df.columns:
      df['dt'] = (
          pd.to_datetime(df['timestamp']).diff().dt.total_seconds().fillna(1.0)
      )
    else:
      df['dt'] = 1.0

  df['dt_clean'] = df['dt'].clip(lower=0.0, upper=5.0)
  df_valid = df[df['heart_rate'].notna() & (df['heart_rate'] > 30)].copy()

  if df_valid.empty:
    return empty_df

  z1_hi = zones['Zone 1 (Recovery)'][1]
  z2_hi = zones['Zone 2 (Endurance)'][1]
  z3_hi = zones['Zone 3 (Tempo)'][1]
  z4_hi = zones['Zone 4 (Threshold)'][1]

  def assign_zone(hr):
    if hr <= z1_hi:
      return 'Zone 1 (Recovery)'
    elif hr <= z2_hi:
      return 'Zone 2 (Endurance)'
    elif hr <= z3_hi:
      return 'Zone 3 (Tempo)'
    elif hr <= z4_hi:
      return 'Zone 4 (Threshold)'
    else:
      return 'Zone 5 (VO2 Max)'

  df_valid['zone'] = df_valid['heart_rate'].apply(assign_zone)
  sec_by_zone = df_valid.groupby('zone')['dt_clean'].sum().to_dict()
  total_sec = sum(sec_by_zone.values())

  rows = []
  for k in zone_keys:
    mins = sec_by_zone.get(k, 0.0) / 60.0
    pct = (
        (sec_by_zone.get(k, 0.0) / total_sec * 100.0) if total_sec > 0 else 0.0
    )
    rows.append({
        'Zone': k,
        'HR Range': f'{zones[k][0]}-{zones[k][1]} bpm',
        'Minutes': round(mins, 1),
        'Percentage': round(pct, 1),
    })

  return pd.DataFrame(rows)


def compute_banister_trimp(duration_min, avg_hr, hr_max, hr_rest):
  """Calculates Banister's TRIMP (Training Impulse) for a workout."""
  if not avg_hr or pd.isna(avg_hr) or avg_hr <= hr_rest or hr_max <= hr_rest:
    return round(duration_min * 1.0, 1)

  delta_hr = (avg_hr - hr_rest) / (hr_max - hr_rest)
  delta_hr = np.clip(delta_hr, 0.0, 1.5)

  b = 1.92
  trimp = duration_min * delta_hr * 0.64 * np.exp(b * delta_hr)
  return round(trimp, 1)


def compute_performance_management_chart(df_history):
  """Computes CTL (Fitness), ATL (Fatigue), and TSB (Form/Freshness)."""
  if df_history.empty or 'date' not in df_history.columns:
    return pd.DataFrame()

  df = df_history.copy()
  df['date'] = pd.to_datetime(df['date'], errors='coerce')
  df = df.sort_values('date').dropna(subset=['date'])

  daily_load = (
      df.groupby(df['date'].dt.date)['relative_effort_trimp']
      .sum()
      .reset_index()
  )
  daily_load['date'] = pd.to_datetime(daily_load['date'])

  full_range = pd.date_range(
      start=daily_load['date'].min(), end=pd.Timestamp.today()
  )
  df_timeline = pd.DataFrame({'date': full_range})
  df_timeline = df_timeline.merge(
      daily_load, on='date', how='left'
  ).fillna({'relative_effort_trimp': 0})

  df_timeline['CTL'] = (
      df_timeline['relative_effort_trimp'].ewm(span=42, adjust=False).mean()
  )
  df_timeline['ATL'] = (
      df_timeline['relative_effort_trimp'].ewm(span=7, adjust=False).mean()
  )
  df_timeline['TSB'] = (
      df_timeline['CTL'].shift(1).fillna(0)
      - df_timeline['ATL'].shift(1).fillna(0)
  )

  return df_timeline


def predict_race_times(known_dist_m, known_time_sec, df_prs=None):
  """Predicts race times using Riegel's Formula: T2 = T1 * (D2 / D1)^1.06,

  while keeping Actual Recorded PRs in a separate comparison column.
  """
  target_distances = [
      ('1 Mile', 1609.34),
      ('2 Mile', 3218.68),
      ('5K', 5000.0),
      ('10K', 10000.0),
      ('Half Marathon', 21097.5),
      ('Marathon', 42195.0),
  ]

  pr_lookup = {}
  if df_prs is not None and not df_prs.empty:
    for _, row in df_prs.iterrows():
      pr_lookup[row['distance_label']] = (
          row['duration_seconds'],
          row['Recorded PR Time'],
          row['activity_date'],
      )

  predictions = []
  for label, target_dist in target_distances:
    predicted_sec = known_time_sec * (target_dist / known_dist_m) ** 1.06
    pred_time_str = format_seconds_to_time(predicted_sec)
    pred_pace_str = format_pace_per_mile(predicted_sec, target_dist)

    if label in pr_lookup:
      actual_sec, actual_str, actual_date = pr_lookup[label]
      actual_display = f'{actual_str} ({actual_date})'
      diff_sec = actual_sec - predicted_sec
      if abs(diff_sec) < 1.0:
        gap_str = 'Benchmark PR'
      elif diff_sec > 0:
        gap_str = f'-{format_seconds_to_time(diff_sec)} faster potential'
      else:
        gap_str = f'+{format_seconds_to_time(abs(diff_sec))} ahead of curve'
    else:
      actual_display = 'Not raced / logged yet'
      gap_str = 'Projected'

    predictions.append({
        'Race Distance': label,
        'Actual Recorded PR (Vault)': actual_display,
        'Riegel Predicted Time': pred_time_str,
        'Predicted Pace': pred_pace_str,
        'PR vs. Prediction': gap_str,
    })

  return pd.DataFrame(predictions)