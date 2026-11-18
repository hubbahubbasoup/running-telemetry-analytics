import glob
import gzip
import io
import os
import sqlite3
import numpy as np
import pandas as pd
from fitparse import FitFile
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


def init_db(db_path=DB_NAME):
  """Initializes all SQLite tables including compressed fit_files and gps_points."""
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
            z5_min REAL
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

  conn.commit()
  conn.close()


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
  """Parses raw .fit bytes and saves summary, PRs, GPS points, and compressed binary to SQLite."""
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

  # 2. If a treadmill run was manually calibrated, scale the record stream to match
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

  if (
      total_dist_m is None
      and not df_records.empty
      and 'distance' in df_records.columns
  ):
    total_dist_m = df_records['distance'].max()

  total_time_sec = session_data.get('total_timer_time')
  if (
      total_time_sec is None
      and not df_records.empty
      and 'timestamp' in df_records.columns
  ):
    df_records['datetime'] = pd.to_datetime(df_records['timestamp'])
    total_time_sec = df_records['datetime'].diff().dt.total_seconds().sum()

  start_time = session_data.get('start_time')
  if (
      start_time is None
      and not df_records.empty
      and 'timestamp' in df_records.columns
  ):
    start_time = df_records['timestamp'].min()

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

  dist_mi = (total_dist_m / 1609.34) if total_dist_m else 0.0
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
  }

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # 1. Upsert activity summary
  cursor.execute(
      '''
      INSERT INTO activities (
          file_name, date, distance_miles, duration_minutes, avg_pace,
          avg_heart_rate, max_heart_rate, calories, relative_effort_trimp,
          workout_tag, efficiency_factor, aerobic_decoupling,
          z1_min, z2_min, z3_min, z4_min, z5_min
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(file_name) DO UPDATE SET
          max_heart_rate = excluded.max_heart_rate,
          relative_effort_trimp = excluded.relative_effort_trimp,
          efficiency_factor = excluded.efficiency_factor,
          aerobic_decoupling = excluded.aerobic_decoupling,
          z1_min = excluded.z1_min,
          z2_min = excluded.z2_min,
          z3_min = excluded.z3_min,
          z4_min = excluded.z4_min,
          z5_min = excluded.z5_min
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
      ),
  )

  # 2. Store compressed .fit binary for on-demand Single Activity inspection
  compressed_blob = gzip.compress(raw_bytes)
  cursor.execute(
      '''
      INSERT INTO fit_files (file_name, raw_gz)
      VALUES (?, ?)
      ON CONFLICT(file_name) DO UPDATE SET raw_gz = excluded.raw_gz
      ''',
      (file_name, sqlite3.Binary(compressed_blob)),
  )

  # 3. Cache downsampled GPS coordinates if not already present
  has_gps = cursor.execute(
      'SELECT 1 FROM gps_points WHERE file_name = ? LIMIT 1', (file_name,)
  ).fetchone()
  if (
      not has_gps
      and not df_records.empty
      and 'position_lat' in df_records.columns
      and 'position_long' in df_records.columns
  ):
    semicircle_to_deg = 180.0 / (2**31)
    df_gps = df_records[['position_lat', 'position_long']].dropna().copy()
    if not df_gps.empty:
      df_gps['lat'] = df_gps['position_lat'] * semicircle_to_deg
      df_gps['lon'] = df_gps['position_long'] * semicircle_to_deg
      df_gps = df_gps[
          df_gps['lat'].between(-89.9, 89.9)
          & df_gps['lon'].between(-179.9, 179.9)
      ].iloc[::downsample_step]
      if not df_gps.empty:
        df_to_save = df_gps[['lat', 'lon']].copy()
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
        print(f'Skipped non-running file: {fname}')
        continue
      print(
          f"Synced: {s['file_name']} | Max HR: {s['max_heart_rate']} | {s['workout_tag']}"
      )
    except Exception as e:
      print(f'Error processing {file_path}: {e}')


if __name__ == '__main__':
  build_activity_database()