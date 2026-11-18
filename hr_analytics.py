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


def parse_time_str_to_seconds(time_str):
  """Parses 'MM:SS', 'HH:MM:SS', or '16m 35s' strings into float seconds."""
  if not time_str or not str(time_str).strip():
    return None
  s = str(time_str).strip().lower()
  try:
    if ':' in s:
      parts = [float(p.strip()) for p in s.split(':')]
      if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
      if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    total = 0.0
    matched = False
    for token in s.replace('s', 's ').replace('m', 'm ').replace('h', 'h ').split():
      if token.endswith('h'):
        total += float(token[:-1]) * 3600.0
        matched = True
      elif token.endswith('m'):
        total += float(token[:-1]) * 60.0
        matched = True
      elif token.endswith('s'):
        total += float(token[:-1])
        matched = True
    if matched and total > 0:
      return total
    return float(s)
  except Exception:
    return None


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
  """Creates athlete_baselines, best_efforts, and manual_prs tables in SQLite."""
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

  cursor.execute('''
      CREATE TABLE IF NOT EXISTS manual_prs (
          distance_label TEXT PRIMARY KEY,
          distance_meters REAL,
          duration_seconds REAL,
          activity_date TEXT
      )
  ''')
  conn.commit()
  conn.close()


def set_manual_pr(distance_label, duration_seconds, activity_date=None, db_path=DB_NAME):
  """Saves or updates a manual PR override for a standard distance."""
  if distance_label not in STANDARD_DISTANCES_M or not duration_seconds or duration_seconds <= 0:
    return False
  init_baselines_table(db_path)
  dist_m = STANDARD_DISTANCES_M[distance_label]
  date_str = str(activity_date)[:10] if activity_date else datetime.now().strftime('%Y-%m-%d')
  conn = sqlite3.connect(db_path)
  conn.execute(
      '''
      INSERT INTO manual_prs (distance_label, distance_meters, duration_seconds, activity_date)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(distance_label) DO UPDATE SET
          duration_seconds = excluded.duration_seconds,
          activity_date = excluded.activity_date
      ''',
      (distance_label, dist_m, round(float(duration_seconds), 1), date_str),
  )
  conn.commit()
  conn.close()
  return True


def delete_manual_pr(distance_label, db_path=DB_NAME):
  """Removes a manual PR override so the vault reverts to the fastest .fit effort."""
  init_baselines_table(db_path)
  conn = sqlite3.connect(db_path)
  conn.execute('DELETE FROM manual_prs WHERE distance_label = ?', (distance_label,))
  conn.commit()
  conn.close()


def extract_best_efforts_from_df(df_records):
  """Runs sliding-window search across a cleaned/raw records DataFrame."""
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
  """Returns all-time PRs, merging manual_prs overrides on top of .fit best_efforts."""
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
  df_fit_prs = pd.read_sql(query, conn)
  df_manual = pd.read_sql('SELECT * FROM manual_prs', conn)
  conn.close()

  pr_map = {}
  for _, row in df_fit_prs.iterrows():
    pr_map[row['distance_label']] = {
        'distance_label': row['distance_label'],
        'distance_meters': float(row['distance_meters']),
        'duration_seconds': float(row['duration_seconds']),
        'file_name': row['file_name'],
        'activity_date': row['activity_date'],
    }

  for _, mrow in df_manual.iterrows():
    lbl = mrow['distance_label']
    pr_map[lbl] = {
        'distance_label': lbl,
        'distance_meters': float(mrow['distance_meters']),
        'duration_seconds': float(mrow['duration_seconds']),
        'file_name': 'Manual PR Override',
        'activity_date': mrow['activity_date'],
    }

  if not pr_map:
    return pd.DataFrame()

  df_prs = pd.DataFrame(list(pr_map.values())).sort_values('distance_meters').reset_index(drop=True)
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
  """Returns (hr_max, hr_rest, peak_source_file, last_updated) while honoring Manual Override."""
  init_baselines_table(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  row = cursor.execute(
      'SELECT hr_max, hr_rest, peak_source_file, last_updated FROM athlete_baselines WHERE id = 1'
  ).fetchone()
  curr_max, curr_rest, peak_src, last_upd = (
      float(row[0]),
      float(row[1]),
      row[2],
      row[3],
  )

  # Only auto-promote peak HR from activities if NOT locked by Manual Override
  if peak_src != 'Manual Override':
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
        if peak_row and peak_row[0] and float(peak_row[0]) > curr_max:
          curr_max = float(peak_row[0])
          peak_src = peak_row[1]
          last_upd = datetime.now().strftime('%Y-%m-%d %H:%M')
          cursor.execute(
              '''
              UPDATE athlete_baselines
              SET hr_max = ?, peak_source_file = ?, last_updated = ?
              WHERE id = 1
              ''',
              (curr_max, peak_src, last_upd),
          )
          conn.commit()

  conn.close()
  return curr_max, curr_rest, peak_src, last_upd


def recalculate_vault_trimp(db_path=DB_NAME, hr_max=None, hr_rest=None):
  """Recalculates Banister TRIMP across all stored activities when HR baselines change."""
  if not os.path.exists(db_path):
    return
  if hr_max is None or hr_rest is None:
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
    reset_auto_max=False,
):
  """Updates HR baselines and recalculates vault TRIMP while respecting Manual Override."""
  hr_max, hr_rest, peak_file, _ = get_athlete_baselines(db_path)
  changed = False

  if reset_auto_max:
    conn = sqlite3.connect(db_path)
    peak_row = conn.execute('''
        SELECT max_heart_rate, file_name
        FROM activities
        WHERE max_heart_rate IS NOT NULL AND max_heart_rate <= 220
        ORDER BY max_heart_rate DESC
        LIMIT 1
    ''').fetchone()
    conn.close()
    if peak_row and peak_row[0]:
      hr_max = round(float(peak_row[0]), 1)
      peak_file = peak_row[1]
    else:
      peak_file = 'Default Baseline'
    changed = True

  if (
      candidate_max_hr is not None
      and peak_file != 'Manual Override'
      and 100 < candidate_max_hr <= 220
      and candidate_max_hr > hr_max
  ):
    hr_max = round(float(candidate_max_hr), 1)
    peak_file = source_file or 'Uploaded Activity'
    changed = True

  if manual_hr_max is not None:
    if round(float(manual_hr_max), 1) != hr_max or peak_file != 'Manual Override':
      hr_max = round(float(manual_hr_max), 1)
      peak_file = 'Manual Override'
      changed = True

  if new_hr_rest is not None and round(float(new_hr_rest), 1) != hr_rest:
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
    recalculate_vault_trimp(db_path, hr_max=hr_max, hr_rest=hr_rest)

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


def predict_race_times(
    known_dist_m=None,
    known_time_sec=None,
    df_prs=None,
    current_ctl=45.0,
    weekly_miles_avg=35.0,
    selected_pr_labels=None,
):
  """Projects race times using an Aggregate Multi-PR + CTL Endurance Decay model

  alongside pure Riegel (b=1.06) for comparison.
  """
  target_distances = [
      ('1 Mile', 1609.34, 30.0, 25.0),
      ('2 Mile', 3218.68, 38.0, 30.0),
      ('5K', 5000.0, 45.0, 35.0),
      ('10K', 10000.0, 58.0, 45.0),
      ('Half Marathon', 21097.5, 72.0, 55.0),
      ('Marathon', 42195.0, 90.0, 70.0),
  ]

  pr_lookup = {}
  if df_prs is not None and not df_prs.empty:
    for _, row in df_prs.iterrows():
      pr_lookup[row['distance_label']] = (
          float(row['distance_meters']),
          float(row['duration_seconds']),
          row['Recorded PR Time'],
          row['activity_date'],
          row['file_name'],
      )

  # Determine active PR benchmarks to aggregate
  active_benchmarks = []
  if selected_pr_labels and pr_lookup:
    for lbl in selected_pr_labels:
      if lbl in pr_lookup:
        d_m, t_s, _, _, _ = pr_lookup[lbl]
        active_benchmarks.append((lbl, d_m, t_s))
  elif known_dist_m and known_time_sec:
    active_benchmarks.append(('Selected', float(known_dist_m), float(known_time_sec)))
  elif pr_lookup:
    for lbl, (d_m, t_s, _, _, _) in pr_lookup.items():
      active_benchmarks.append((lbl, d_m, t_s))

  # Fit personal fatigue exponent from selected PRs (bounded to physiological range)
  base_exponent = 1.075
  if len(active_benchmarks) >= 2:
    xs = np.log([b[1] for b in active_benchmarks])
    ys = np.log([b[2] for b in active_benchmarks])
    if np.std(xs) > 0:
      slope = float(np.polyfit(xs, ys, 1)[0])
      base_exponent = float(np.clip(slope, 1.055, 1.12))

  predictions = []
  for label, target_dist, req_ctl, req_mpw in target_distances:
    # Compute CTL & Weekly Mileage endurance penalty for distances > 3200m
    ctl_deficit = max(0.0, (req_ctl - float(current_ctl)) / 100.0)
    mpw_deficit = max(0.0, (req_mpw - float(weekly_miles_avg)) / 100.0)
    endurance_penalty = (0.11 * ctl_deficit) + (0.07 * mpw_deficit)

    weighted_preds = []
    weights = []
    riegel_preds = []

    for _, b_dist, b_time in active_benchmarks:
      dist_ratio = target_dist / b_dist
      if dist_ratio > 1.0:
        eff_exp = base_exponent + endurance_penalty
      else:
        eff_exp = max(1.05, base_exponent - 0.01)

      pred_i = b_time * (dist_ratio ** eff_exp)
      riegel_i = b_time * (dist_ratio ** 1.06)

      # Weight closer distances higher in log-distance space
      log_dist_gap = abs(np.log(target_dist / b_dist))
      w_i = 1.0 / (0.25 + log_dist_gap)
      weighted_preds.append(pred_i * w_i)
      riegel_preds.append(riegel_i * w_i)
      weights.append(w_i)

    predicted_sec = sum(weighted_preds) / sum(weights) if weights else 0.0
    riegel_sec = sum(riegel_preds) / sum(weights) if weights else 0.0

    pred_time_str = format_seconds_to_time(predicted_sec)
    pred_pace_str = format_pace_per_mile(predicted_sec, target_dist)
    riegel_time_str = format_seconds_to_time(riegel_sec)

    if label in pr_lookup:
      _, actual_sec, actual_str, actual_date, src_file = pr_lookup[label]
      src_tag = ' [Manual]' if src_file == 'Manual PR Override' else ''
      actual_display = f'{actual_str} ({actual_date}){src_tag}'
      diff_sec = actual_sec - predicted_sec
      if abs(diff_sec) < 2.0:
        gap_str = 'Aligned with PR'
      elif diff_sec > 0:
        gap_str = f'-{format_seconds_to_time(diff_sec)} faster potential'
      else:
        gap_str = f'+{format_seconds_to_time(abs(diff_sec))} ahead of model'
    else:
      actual_display = 'Not raced / logged yet'
      gap_str = 'CTL-Adjusted Projection'

    predictions.append({
        'Race Distance': label,
        'Actual / Calibrated PR': actual_display,
        'CTL + Multi-PR Prediction': pred_time_str,
        'Predicted Pace': pred_pace_str,
        'Pure Riegel (1.06)': riegel_time_str,
        'PR vs. Model': gap_str,
    })

  return pd.DataFrame(predictions)