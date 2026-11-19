import glob
import gzip
import io
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import requests
from fitparse import FitFile
from cloud_vault import is_cloud_configured, push_fit_gz_to_cloud
from hr_analytics import (
    calculate_karvonen_zones,
    compute_banister_trimp,
    compute_time_in_zones,
    extract_best_efforts_from_df,
    get_athlete_baselines,
    init_baselines_table,
    save_best_efforts_to_db,
    update_athlete_baselines,
)

DB_NAME = 'training_vault.db'

KNOWN_HUBS = [
    (37.485, -122.236, 'Redwood City', 'California'),
    (37.453, -122.182, 'Menlo Park', 'California'),
    (37.442, -122.143, 'Palo Alto', 'California'),
    (37.461, -122.198, 'Atherton', 'California'),
    (37.429, -122.254, 'Woodside', 'California'),
    (37.507, -122.260, 'San Carlos', 'California'),
    (37.520, -122.276, 'Belmont', 'California'),
    (37.563, -122.325, 'San Mateo', 'California'),
    (37.386, -122.084, 'Mountain View', 'California'),
    (37.368, -122.036, 'Sunnyvale', 'California'),
    (37.338, -121.886, 'San Jose', 'California'),
    (37.775, -122.419, 'San Francisco', 'California'),
    (36.974, -122.031, 'Santa Cruz', 'California'),
    (40.233, -111.658, 'Provo', 'Utah'),
    (40.297, -111.694, 'Orem', 'Utah'),
    (40.760, -111.891, 'Salt Lake City', 'Utah'),
    (40.646, -111.498, 'Park City', 'Utah'),
    (37.096, -113.568, 'St. George', 'Utah'),
]


def init_db(db_path=DB_NAME):
  """Initializes all SQLite tables including activities, fit_files, gps_points, and location_cache."""
  init_baselines_table(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities (
            file_name TEXT PRIMARY KEY,
            date TEXT,
            distance_miles REAL,
            duration_minutes REAL,
            avg_pace TEXT,
            avg_heart_rate REAL,
            max_heart_rate REAL,
            calories REAL,
            relative_effort_trimp REAL,
            workout_tag TEXT,
            efficiency_factor REAL,
            aerobic_decoupling REAL,
            z1_min REAL,
            z2_min REAL,
            z3_min REAL,
            z4_min REAL,
            z5_min REAL,
            city TEXT,
            state TEXT
        )
    ''')
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS fit_files (
            file_name TEXT PRIMARY KEY,
            raw_gz BLOB
        )
    ''')
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS gps_points (
            file_name TEXT,
            lat REAL,
            lon REAL
        )
    ''')
  cursor.execute(
      'CREATE INDEX IF NOT EXISTS idx_gps_file ON gps_points(file_name)'
  )
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS location_cache (
            grid_key TEXT PRIMARY KEY,
            city TEXT,
            state TEXT
        )
    ''')

  existing_cols = [
      row[1]
      for row in cursor.execute('PRAGMA table_info(activities)').fetchall()
  ]
  for col in [
      'max_heart_rate',
      'efficiency_factor',
      'aerobic_decoupling',
      'z1_min',
      'z2_min',
      'z3_min',
      'z4_min',
      'z5_min',
  ]:
    if col not in existing_cols:
      cursor.execute(f'ALTER TABLE activities ADD COLUMN {col} REAL')

  for text_col in ['city', 'state']:
    if text_col not in existing_cols:
      cursor.execute(f'ALTER TABLE activities ADD COLUMN {text_col} TEXT')

  conn.commit()
  conn.close()


def _grid_key(lat, lon):
  """Rounds coordinates to a ~550m grid cell (0.005 deg) for accurate municipal caching."""
  r_lat = round(float(lat) * 200.0) / 200.0
  r_lon = round(float(lon) * 200.0) / 200.0
  return f'{r_lat:.3f},{r_lon:.3f}'


def _offline_hub_lookup(lat, lon):
  """Fallback nearest-city resolver if offline."""
  best_city, best_state, best_dist = 'Outdoor Run', 'Unknown', 999.0
  for h_lat, h_lon, c_name, s_name in KNOWN_HUBS:
    d = ((lat - h_lat) ** 2 + (lon - h_lon) ** 2) ** 0.5
    if d < best_dist:
      best_dist = d
      best_city, best_state = c_name, s_name
  if best_dist <= 0.12:
    return best_city, best_state
  return 'Outdoor Run', 'Unknown'


def resolve_lat_lon_to_city_state(lat, lon, polite_delay=0.0):
  """Resolves (lat, lon) to exact municipality (city, state) via OpenStreetMap Nominatim,

  falling back to BigDataCloud adminLevel=8 and offline hub lookup.
  """
  if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
    return 'Indoor', 'Treadmill'

  if polite_delay > 0:
    time.sleep(polite_delay)

  try:
    nom_url = (
        'https://nominatim.openstreetmap.org/reverse'
        f'?format=jsonv2&lat={float(lat):.5f}&lon={float(lon):.5f}&zoom=14&addressdetails=1'
    )
    headers = {
        'User-Agent': 'RunningTelemetryAnalytics/1.0 (personal-training-vault)'
    }
    resp = requests.get(nom_url, headers=headers, timeout=6)
    if resp.status_code == 200:
      addr = resp.json().get('address', {})
      city = (
          addr.get('city')
          or addr.get('town')
          or addr.get('village')
          or addr.get('municipality')
          or addr.get('hamlet')
          or addr.get('suburb')
      )
      state = addr.get('state') or addr.get('region') or addr.get('country') or ''
      if city:
        return str(city).strip(), str(state).strip()
  except Exception:
    pass

  try:
    bdc_url = (
        'https://api.bigdatacloud.net/data/reverse-geocode-client'
        f'?latitude={float(lat):.4f}&longitude={float(lon):.4f}&localityLanguage=en'
    )
    resp = requests.get(bdc_url, timeout=4)
    if resp.status_code == 200:
      data = resp.json()
      if data.get('lookupSource') == 'coordinates':
        admin_list = data.get('localityInfo', {}).get('administrative', [])
        lvl8 = [
            a.get('name')
            for a in admin_list
            if a.get('adminLevel') == 8 and a.get('name')
        ]
        city = (
            (lvl8[0] if lvl8 else None)
            or data.get('locality')
            or data.get('city')
            or 'Outdoor Run'
        )
        state = data.get('principalSubdivision') or data.get('countryName') or ''
        return str(city).strip(), str(state).strip()
  except Exception:
    pass

  return _offline_hub_lookup(float(lat), float(lon))


def _get_activity_start_coords(cursor):
  """Returns {file_name: (start_lat, start_lon)} from the first recorded GPS point of each run."""
  rows = cursor.execute('''
      SELECT g.file_name, g.lat, g.lon
      FROM gps_points g
      INNER JOIN (
          SELECT file_name, MIN(rowid) AS first_id
          FROM gps_points
          GROUP BY file_name
      ) f ON g.file_name = f.file_name AND g.rowid = f.first_id
  ''').fetchall()
  return {
      r[0]: (float(r[1]), float(r[2]))
      for r in rows
      if r[1] is not None and r[2] is not None
  }


def backfill_activity_locations(db_path=DB_NAME):
  """Populates missing city and state values for activities not yet geocoded."""
  init_db(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  missing = cursor.execute(
      "SELECT file_name FROM activities WHERE city IS NULL OR city = ''"
  ).fetchall()
  if not missing:
    conn.close()
    return False

  start_coords = _get_activity_start_coords(cursor)
  cached_grids = {
      r[0]: (r[1], r[2])
      for r in cursor.execute(
          'SELECT grid_key, city, state FROM location_cache'
      ).fetchall()
  }

  to_resolve = {}
  file_to_grid = {}
  for (fname,) in missing:
    if fname in start_coords:
      lat, lon = start_coords[fname]
      gkey = _grid_key(lat, lon)
      file_to_grid[fname] = gkey
      if gkey not in cached_grids and gkey not in to_resolve:
        to_resolve[gkey] = (lat, lon)

  for idx, (gkey, (lat_v, lon_v)) in enumerate(to_resolve.items()):
    c_str, s_str = resolve_lat_lon_to_city_state(
        lat_v, lon_v, polite_delay=1.05 if idx > 0 else 0.0
    )
    cached_grids[gkey] = (c_str, s_str)
    cursor.execute(
        '''
        INSERT INTO location_cache (grid_key, city, state)
        VALUES (?, ?, ?)
        ON CONFLICT(grid_key) DO UPDATE SET city = excluded.city, state = excluded.state
        ''',
        (gkey, c_str, s_str),
    )

  for (fname,) in missing:
    if fname in file_to_grid:
      c_val, s_val = cached_grids.get(
          file_to_grid[fname], ('Outdoor Run', 'Unknown')
      )
    else:
      c_val, s_val = ('Indoor', 'Treadmill')
    cursor.execute(
        'UPDATE activities SET city = ?, state = ? WHERE file_name = ?',
        (c_val, s_val, fname),
    )

  conn.commit()
  conn.close()
  return True


def regeocode_all_activities(db_path=DB_NAME, progress_callback=None):
  """Clears stale location cache and accurately re-geocodes every run from its starting GPS point."""
  init_db(db_path)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  cursor.execute('DELETE FROM location_cache')
  all_files = [
      r[0] for r in cursor.execute('SELECT file_name FROM activities').fetchall()
  ]
  start_coords = _get_activity_start_coords(cursor)

  unique_grids = {}
  file_to_grid = {}
  for fname in all_files:
    if fname in start_coords:
      lat, lon = start_coords[fname]
      gkey = _grid_key(lat, lon)
      file_to_grid[fname] = gkey
      if gkey not in unique_grids:
        unique_grids[gkey] = (lat, lon)

  resolved = {}
  total_grids = len(unique_grids)
  for idx, (gkey, (lat_v, lon_v)) in enumerate(unique_grids.items()):
    if progress_callback:
      progress_callback(idx + 1, total_grids)
    c_str, s_str = resolve_lat_lon_to_city_state(
        lat_v, lon_v, polite_delay=1.05 if idx > 0 else 0.0
    )
    resolved[gkey] = (c_str, s_str)
    cursor.execute(
        '''
        INSERT INTO location_cache (grid_key, city, state)
        VALUES (?, ?, ?)
        ON CONFLICT(grid_key) DO UPDATE SET city = excluded.city, state = excluded.state
        ''',
        (gkey, c_str, s_str),
    )

  for fname in all_files:
    if fname in file_to_grid:
      c_val, s_val = resolved.get(
          file_to_grid[fname], ('Outdoor Run', 'Unknown')
      )
    else:
      c_val, s_val = ('Indoor', 'Treadmill')
    cursor.execute(
        'UPDATE activities SET city = ?, state = ? WHERE file_name = ?',
        (c_val, s_val, fname),
    )

  conn.commit()
  conn.close()
  return len(all_files), total_grids


def delete_activity_from_vault(file_name, db_path=DB_NAME):
  """Deletes a single activity cleanly across activities, fit_files, gps_points, and best_efforts."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  for table in ['activities', 'fit_files', 'gps_points', 'best_efforts']:
    cursor.execute(f'DELETE FROM {table} WHERE file_name = ?', (file_name,))
  conn.commit()
  conn.close()


def deduplicate_vault(db_path=DB_NAME):
  """Detects and removes duplicate activities with matching start timestamps and distances.

  Prefers removing legacy *_ACTIVITY.fit test files when a Garmin export copy exists.
  """
  init_db(db_path)
  conn = sqlite3.connect(db_path)
  df = pd.read_sql(
      'SELECT file_name, date, distance_miles FROM activities ORDER BY date ASC',
      conn,
  )
  conn.close()

  if df.empty:
    return []

  df['dt'] = pd.to_datetime(df['date'], errors='coerce')
  df = df.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)

  kept = []
  to_delete = []

  for _, row in df.iterrows():
    fname = row['file_name']
    dt = row['dt']
    dist = float(row['distance_miles']) if pd.notna(row['distance_miles']) else 0.0

    matched_idx = None
    for idx, k in enumerate(kept):
      if (
          abs((dt - k['dt']).total_seconds()) <= 90
          and abs(dist - k['dist']) <= 0.2
      ):
        matched_idx = idx
        break

    if matched_idx is None:
      kept.append({'file_name': fname, 'dt': dt, 'dist': dist})
    else:
      existing = kept[matched_idx]
      if existing['file_name'].endswith('_ACTIVITY.fit') and not fname.endswith(
          '_ACTIVITY.fit'
      ):
        to_delete.append(existing['file_name'])
        kept[matched_idx] = {'file_name': fname, 'dt': dt, 'dist': dist}
      else:
        to_delete.append(fname)

  for dup_fname in to_delete:
    delete_activity_from_vault(dup_fname, db_path)

  return to_delete


def compute_telemetry_metrics(df_records, dist_mi, total_time_sec, zones):
  """Computes dynamic workout tag, Efficiency Factor, and Aerobic Decoupling."""
  if df_records.empty:
    return 'Base Run', None, None

  pace_std, pace_cv = 0.0, 0.0
  if 'enhanced_speed' in df_records.columns:
    running_mask = df_records['enhanced_speed'] >= 2.0
    clean_speeds = df_records.loc[running_mask, 'enhanced_speed'].dropna()
    if not clean_speeds.empty:
      pace_decimal = (1609.34 / clean_speeds) / 60.0
      pace_std = pace_decimal.std()
      pace_mean = pace_decimal.mean()
      pace_cv = pace_std / pace_mean if pace_mean > 0 else 0.0

  avg_hr = (
      df_records['heart_rate'].mean()
      if 'heart_rate' in df_records.columns
      and df_records['heart_rate'].notna().any()
      else None
  )

  z1_upper = zones['Zone 1 (Recovery)'][1]
  z3_upper = zones['Zone 3 (Tempo)'][1]

  is_long = dist_mi >= 9.0 or (total_time_sec / 60.0) >= 75
  if is_long:
    tag = (
        'Tempo / Quality Long Run'
        if (pace_cv > 0.10 or pace_std > 0.75)
        else 'Long Run'
    )
  elif pace_std > 1.2 or pace_cv > 0.15:
    tag = 'Intervals / Speedwork'
  elif avg_hr and avg_hr <= z1_upper:
    tag = 'Recovery / Easy'
  elif avg_hr and avg_hr >= z3_upper:
    tag = 'Tempo / Threshold'
  else:
    tag = 'Base Run'

  ef_score, decoupling_pct = None, None
  if (
      'enhanced_speed' in df_records.columns
      and 'heart_rate' in df_records.columns
  ):
    df_valid = df_records[
        (df_records['enhanced_speed'] >= 2.0)
        & (df_records['heart_rate'].notna())
        & (df_records['heart_rate'] > 0)
    ]
    if not df_valid.empty:
      ef_score = round(
          df_valid['enhanced_speed'].mean() / df_valid['heart_rate'].mean(), 4
      )
      if len(df_valid) >= 60:
        mid = len(df_valid) // 2
        r1 = (
            df_valid.iloc[:mid]['heart_rate']
            / df_valid.iloc[:mid]['enhanced_speed']
        ).mean()
        r2 = (
            df_valid.iloc[mid:]['heart_rate']
            / df_valid.iloc[mid:]['enhanced_speed']
        ).mean()
        if r1 > 0:
          decoupling_pct = round(((r2 - r1) / r1) * 100, 1)

  return tag, ef_score, decoupling_pct


def ingest_fit_bytes(
    file_name, raw_bytes, db_path=DB_NAME, downsample_step=3
):
  """Parses raw .fit bytes and saves summary, location, PRs, GPS points, and compressed binary."""
  init_db(db_path)
  fitfile = FitFile(io.BytesIO(raw_bytes))

  session_data = {
      field.name: field.value
      for session in fitfile.get_messages('session')
      for field in session
  }
  records = [
      {field.name: field.value for field in record}
      for record in fitfile.get_messages('record')
  ]
  df_records = pd.DataFrame(records)

  sport = str(session_data.get('sport', 'running')).lower()
  if sport != 'running':
    return None

  total_dist_m = session_data.get('total_distance')
  if (
      total_dist_m is None
      and not df_records.empty
      and 'distance' in df_records.columns
  ):
    total_dist_m = df_records['distance'].max()

  if (
      total_dist_m
      and not df_records.empty
      and 'distance' in df_records.columns
  ):
    raw_max_dist = df_records['distance'].max()
    if pd.notna(raw_max_dist) and raw_max_dist > 0:
      cal_ratio = total_dist_m / raw_max_dist
      if abs(cal_ratio - 1.0) > 0.005:
        df_records['distance'] = df_records['distance'] * cal_ratio
        if 'enhanced_speed' in df_records.columns:
          df_records['enhanced_speed'] = df_records['enhanced_speed'] * cal_ratio

  total_time_sec = session_data.get('total_timer_time')
  if (
      total_time_sec is None
      and not df_records.empty
      and 'timestamp' in df_records.columns
  ):
    df_records['datetime'] = pd.to_datetime(df_records['timestamp'])
    total_time_sec = (
        df_records['datetime']
        .diff()
        .dt.total_seconds()
        .clip(lower=0.0, upper=5.0)
        .sum()
    )

  start_time = session_data.get('start_time')
  if (
      start_time is None
      and not df_records.empty
      and 'timestamp' in df_records.columns
  ):
    start_time = df_records['timestamp'].min()

  dist_mi = (total_dist_m / 1609.34) if total_dist_m else 0.0

  if start_time is not None:
    conn_chk = sqlite3.connect(db_path)
    dup_row = conn_chk.execute(
        '''
        SELECT file_name FROM activities
        WHERE file_name != ? AND date = ? AND ABS(distance_miles - ?) <= 0.15
        LIMIT 1
        ''',
        (file_name, str(start_time), round(dist_mi, 2)),
    ).fetchone()
    conn_chk.close()
    if dup_row:
      return None

  avg_hr = session_data.get('avg_heart_rate')
  if (
      avg_hr is None
      and not df_records.empty
      and 'heart_rate' in df_records.columns
  ):
    avg_hr = df_records['heart_rate'].mean()

  max_hr = session_data.get('max_heart_rate')
  if not df_records.empty and 'heart_rate' in df_records.columns:
    valid_hrs = df_records.loc[
        (df_records['heart_rate'] > 40) & (df_records['heart_rate'] <= 220),
        'heart_rate',
    ]
    if not valid_hrs.empty:
      rec_max = float(valid_hrs.max())
      if max_hr is None or max_hr > 220 or rec_max > max_hr:
        max_hr = rec_max

  hr_max, hr_rest, _ = update_athlete_baselines(
      db_path, candidate_max_hr=max_hr, source_file=file_name
  )
  zones = calculate_karvonen_zones(hr_max, hr_rest)

  efforts = extract_best_efforts_from_df(df_records)
  save_best_efforts_to_db(file_name, str(start_time), efforts, db_path)

  duration_min = (total_time_sec / 60.0) if total_time_sec else 0.0

  if dist_mi > 0 and duration_min > 0:
    sec_per_mile = total_time_sec / dist_mi
    avg_pace_display = (
        f'{int(sec_per_mile // 60)}:{int(sec_per_mile % 60):02d} /mi'
    )
  else:
    avg_pace_display = '0:00 /mi'

  trimp_score = compute_banister_trimp(duration_min, avg_hr, hr_max, hr_rest)
  tag, ef_score, decoupling_pct = compute_telemetry_metrics(
      df_records, dist_mi, total_time_sec or 0.0, zones
  )

  df_tiz = compute_time_in_zones(df_records, zones)
  z_mins = df_tiz['Minutes'].tolist()

  semicircle_to_deg = 180.0 / (2**31)
  df_gps_clean = pd.DataFrame()
  city_name, state_name = 'Indoor', 'Treadmill'

  if (
      not df_records.empty
      and 'position_lat' in df_records.columns
      and 'position_long' in df_records.columns
  ):
    df_gps = df_records[['position_lat', 'position_long']].dropna().copy()
    if not df_gps.empty:
      df_gps['lat'] = df_gps['position_lat'] * semicircle_to_deg
      df_gps['lon'] = df_gps['position_long'] * semicircle_to_deg
      df_gps_clean = df_gps[
          df_gps['lat'].between(-89.9, 89.9)
          & df_gps['lon'].between(-179.9, 179.9)
      ].iloc[::downsample_step]

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  if not df_gps_clean.empty:
    start_lat = float(df_gps_clean['lat'].iloc[0])
    start_lon = float(df_gps_clean['lon'].iloc[0])
    gkey = _grid_key(start_lat, start_lon)
    cached = cursor.execute(
        'SELECT city, state FROM location_cache WHERE grid_key = ?', (gkey,)
    ).fetchone()
    if cached:
      city_name, state_name = cached[0], cached[1]
    else:
      city_name, state_name = resolve_lat_lon_to_city_state(start_lat, start_lon)
      cursor.execute(
          '''
          INSERT INTO location_cache (grid_key, city, state)
          VALUES (?, ?, ?)
          ON CONFLICT(grid_key) DO UPDATE SET city = excluded.city, state = excluded.state
          ''',
          (gkey, city_name, state_name),
      )

  summary = {
      'file_name': file_name,
      'date': str(start_time),
      'distance_miles': round(dist_mi, 2),
      'duration_minutes': round(duration_min, 1),
      'avg_pace': avg_pace_display,
      'avg_heart_rate': round(avg_hr, 1) if avg_hr is not None else None,
      'max_heart_rate': round(max_hr, 1) if max_hr is not None else None,
      'calories': session_data.get('total_calories', 0),
      'relative_effort_trimp': trimp_score,
      'workout_tag': tag,
      'efficiency_factor': ef_score,
      'aerobic_decoupling': decoupling_pct,
      'z1_min': z_mins[0],
      'z2_min': z_mins[1],
      'z3_min': z_mins[2],
      'z4_min': z_mins[3],
      'z5_min': z_mins[4],
      'city': city_name,
      'state': state_name,
  }

  cursor.execute(
      '''
      INSERT INTO activities (
          file_name, date, distance_miles, duration_minutes, avg_pace,
          avg_heart_rate, max_heart_rate, calories, relative_effort_trimp,
          workout_tag, efficiency_factor, aerobic_decoupling,
          z1_min, z2_min, z3_min, z4_min, z5_min, city, state
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(file_name) DO UPDATE SET
          distance_miles = excluded.distance_miles,
          avg_pace = excluded.avg_pace,
          max_heart_rate = excluded.max_heart_rate,
          relative_effort_trimp = excluded.relative_effort_trimp,
          efficiency_factor = excluded.efficiency_factor,
          aerobic_decoupling = excluded.aerobic_decoupling,
          z1_min = excluded.z1_min,
          z2_min = excluded.z2_min,
          z3_min = excluded.z3_min,
          z4_min = excluded.z4_min,
          z5_min = excluded.z5_min,
          city = excluded.city,
          state = excluded.state
      ''',
      (
          summary['file_name'],
          summary['date'],
          summary['distance_miles'],
          summary['duration_minutes'],
          summary['avg_pace'],
          summary['avg_heart_rate'],
          summary['max_heart_rate'],
          summary['calories'],
          summary['relative_effort_trimp'],
          summary['workout_tag'],
          summary['efficiency_factor'],
          summary['aerobic_decoupling'],
          summary['z1_min'],
          summary['z2_min'],
          summary['z3_min'],
          summary['z4_min'],
          summary['z5_min'],
          summary['city'],
          summary['state'],
      ),
  )

  compressed_blob = gzip.compress(raw_bytes)
  if is_cloud_configured():
    push_fit_gz_to_cloud(file_name, compressed_blob)
    stored_blob = None
  else:
    stored_blob = sqlite3.Binary(compressed_blob)

  cursor.execute(
      '''
      INSERT INTO fit_files (file_name, raw_gz)
      VALUES (?, ?)
      ON CONFLICT(file_name) DO UPDATE SET raw_gz = excluded.raw_gz
      ''',
      (file_name, stored_blob),
  )

  has_gps = cursor.execute(
      'SELECT 1 FROM gps_points WHERE file_name = ? LIMIT 1', (file_name,)
  ).fetchone()
  if not has_gps and not df_gps_clean.empty:
    df_to_save = df_gps_clean[['lat', 'lon']].copy()
    df_to_save.insert(0, 'file_name', file_name)
    df_to_save.to_sql('gps_points', conn, if_exists='append', index=False)

  conn.commit()
  conn.close()
  return summary


def build_activity_database(folder_path='fit test files'):
  init_db(DB_NAME)
  fit_files = glob.glob(os.path.join(folder_path, '*.fit'))
  if not fit_files:
    fit_files = glob.glob('*.fit') + glob.glob('**/*.fit', recursive=True)

  print(f'Found {len(fit_files)} .fit files to sync with SQLite.')

  for file_path in fit_files:
    try:
      fname = os.path.basename(file_path)
      with open(file_path, 'rb') as f:
        raw_bytes = f.read()
      s = ingest_fit_bytes(fname, raw_bytes, DB_NAME)
      if s is None:
        print(f'Skipped non-running or duplicate file: {fname}')
        continue
      print(
          f"Synced: {s['file_name']} | {s['city']} | Max HR: {s['max_heart_rate']} | {s['workout_tag']}"
      )
    except Exception as e:
      print(f'Error processing {file_path}: {e}')


if __name__ == '__main__':
  build_activity_database()