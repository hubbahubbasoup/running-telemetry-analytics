import gzip
import io
import os
import sqlite3
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from fitparse import FitFile
import streamlit as st
from aggregate_runs import build_activity_database, init_db, ingest_fit_bytes
from hr_analytics import (
    calculate_karvonen_zones,
    compute_banister_trimp,
    compute_performance_management_chart,
    compute_time_in_zones,
    extract_best_efforts_from_df,
    get_all_time_prs,
    get_athlete_baselines,
    predict_race_times,
    save_best_efforts_to_db,
    update_athlete_baselines,
)
from route_heatmap import render_heatmap_tab
from cloud_vault import (
    check_password_gate,
    ensure_local_vault,
    is_cloud_configured,
    pull_vault_from_cloud,
    push_vault_to_cloud,
)


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='Running Telemetry & Physiological Analytics',
    layout='wide',
)

# --- DARK-RED THEME, CALM DARK-GREY SIDEBAR & CORMORANT FONT STYLING ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Cormorant:wght@400;600;700&display=swap');

    :root {
        --bg-primary: #0d0505;
        --bg-sidebar: #1e1e22;
        --bg-card: #1c0b0b;
        --accent-crimson: #991b1b;
        --accent-bright: #dc2626;
        --text-ivory: #f5ebeb;
        --text-muted: #c9adad;
    }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-ivory);
        font-family: 'Cormorant Garamond', 'Cormorant', Georgia, serif !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid #2e2e34;
    }

    /* Apply Cormorant Garamond to text elements without overriding Material Icon spans */
    html, body, .stMarkdown, p, label,
    h1, h2, h3, h4, h5, h6, button, input, select, textarea,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"], [data-baseweb="tab"] {
        font-family: 'Cormorant Garamond', 'Cormorant', Georgia, serif !important;
    }

    /* Preserve Streamlit Material Icons so ligatures never render as raw text */
    [data-testid="stIconMaterial"], .material-symbols-rounded, span[translate="no"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    h1, h2, h3 {
        color: var(--text-ivory) !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    p, label, .stCaption {
        font-size: 1.08rem !important;
    }

    div[data-testid="stMetric"] {
        background-color: var(--bg-card);
        border: 1px solid #451010;
        border-left: 3px solid var(--accent-bright);
        padding: 12px 16px;
        border-radius: 6px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 1.0rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-ivory) !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background-color: #28282d !important;
        color: var(--text-ivory) !important;
        border: 1px solid #4a4a52 !important;
        border-radius: 4px;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--accent-crimson) !important;
        border-color: var(--accent-bright) !important;
        color: #ffffff !important;
    }

    hr {
        border-color: #3b0d0d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Running Telemetry & Physiological Analytics')
st.markdown(
    'Self-updating analytics engine tracking telemetry, dynamic Karvonen heart'
    ' rate zones, and Banister TRIMP load.'
)

DB_NAME = 'training_vault.db'
check_password_gate()
ensure_local_vault(DB_NAME)
init_db(DB_NAME)


ZONE_PALETTE = {
    'Zone 1 (Recovery)': '#9ca3af',   # Grey (Zone 0/1)
    'Zone 2 (Endurance)': '#38bdf8',  # Light Blue
    'Zone 3 (Tempo)': '#4ade80',      # Light Green
    'Zone 4 (Threshold)': '#fb923c',  # Light-ish Orange
    'Zone 5 (VO2 Max)': '#ef4444',    # Red
}

# --- HELPER FUNCTIONS ---


def get_hr_zone_color(hr_val, zones):
  """Maps a heart rate value to its dynamic Karvonen zone color."""
  if pd.isna(hr_val):
    return ZONE_PALETTE['Zone 1 (Recovery)']
  if hr_val <= zones['Zone 1 (Recovery)'][1]:
    return ZONE_PALETTE['Zone 1 (Recovery)']
  if hr_val <= zones['Zone 2 (Endurance)'][1]:
    return ZONE_PALETTE['Zone 2 (Endurance)']
  if hr_val <= zones['Zone 3 (Tempo)'][1]:
    return ZONE_PALETTE['Zone 3 (Tempo)']
  if hr_val <= zones['Zone 4 (Threshold)'][1]:
    return ZONE_PALETTE['Zone 4 (Threshold)']
  return ZONE_PALETTE['Zone 5 (VO2 Max)']


def classify_workout(df_records, total_dist_mi, total_time_sec, zones):
  if df_records.empty:
    return 'General Run'

  pace_std = 0.0
  pace_cv = 0.0
  if 'enhanced_speed' in df_records.columns:
    running_mask = df_records['enhanced_speed'] >= 2.0
    clean_speeds = df_records.loc[running_mask, 'enhanced_speed'].dropna()
    if not clean_speeds.empty:
      sec_per_mile = 1609.34 / clean_speeds
      pace_decimal = sec_per_mile / 60.0
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

  is_long_distance = total_dist_mi >= 9.0 or (total_time_sec / 60) >= 75

  if is_long_distance:
    if pace_cv > 0.10 or pace_std > 0.75:
      return 'Tempo / Quality Long Run'
    else:
      return 'Long Run'

  if pace_std > 1.2 or pace_cv > 0.15:
    return 'Intervals / Speedwork'

  if avg_hr and avg_hr <= z1_upper:
    return 'Recovery / Easy'
  elif avg_hr and avg_hr >= z3_upper:
    return 'Tempo / Threshold'

  return 'Base Run'


def calculate_aerobic_decoupling(df_records):
  if (
      df_records.empty
      or 'enhanced_speed' not in df_records.columns
      or 'heart_rate' not in df_records.columns
  ):
    return None

  df_valid = df_records[
      (df_records['enhanced_speed'] >= 2.0)
      & (df_records['heart_rate'].notna())
      & (df_records['heart_rate'] > 0)
  ].copy()

  if len(df_valid) < 60:
    return None

  midpoint = len(df_valid) // 2
  df_first = df_valid.iloc[:midpoint]
  df_second = df_valid.iloc[midpoint:]

  ratio_1 = (df_first['heart_rate'] / df_first['enhanced_speed']).mean()
  ratio_2 = (df_second['heart_rate'] / df_second['enhanced_speed']).mean()

  if ratio_1 == 0:
    return None

  decoupling_pct = ((ratio_2 - ratio_1) / ratio_1) * 100
  return round(decoupling_pct, 1)


def calculate_efficiency_factor(df_records):
  if (
      df_records.empty
      or 'enhanced_speed' not in df_records.columns
      or 'heart_rate' not in df_records.columns
  ):
    return None

  df_valid = df_records[
      (df_records['enhanced_speed'] >= 2.0)
      & (df_records['heart_rate'].notna())
      & (df_records['heart_rate'] > 0)
  ]

  if df_valid.empty:
    return None

  avg_speed = df_valid['enhanced_speed'].mean()
  avg_hr = df_valid['heart_rate'].mean()

  return round(avg_speed / avg_hr, 4)


def parse_fit_bytes_raw(raw_bytes):
  fitfile = FitFile(io.BytesIO(raw_bytes))
  records, laps, sessions = [], [], []
  for record in fitfile.get_messages('record'):
    records.append({field.name: field.value for field in record})
  for lap in fitfile.get_messages('lap'):
    laps.append({field.name: field.value for field in lap})
  for session in fitfile.get_messages('session'):
    sessions.append({field.name: field.value for field in session})

  return {
      'records': pd.DataFrame(records),
      'laps': pd.DataFrame(laps),
      'session': pd.DataFrame(sessions),
  }


def clean_records(df_records):
  if df_records.empty:
    return df_records
  if 'timestamp' in df_records.columns:
    df_records['datetime'] = pd.to_datetime(df_records['timestamp'])

  df_records['dt'] = (
      df_records['datetime'].diff().dt.total_seconds().fillna(1.0)
  )

  if 'enhanced_speed' in df_records.columns:
    speed = df_records['enhanced_speed'].replace(0, np.nan)
    sec_per_mile = 1609.34 / speed
    df_records['pace_decimal'] = sec_per_mile / 60

  return df_records


def extract_native_laps(df_laps):
  if df_laps.empty:
    return pd.DataFrame()
  valid_laps = df_laps.dropna(axis=1, how='all').copy()

  splits = []
  for idx, row in valid_laps.iterrows():
    dist_m = row.get('total_distance', 0)
    time_sec = row.get('total_timer_time', row.get('total_elapsed_time', 0))
    avg_hr = row.get('avg_heart_rate', None)

    if dist_m > 0 and time_sec > 0:
      dist_mi = dist_m / 1609.34
      sec_per_mile = time_sec / dist_mi
      pace_min = int(sec_per_mile // 60)
      pace_sec = int(sec_per_mile % 60)
      pace_display = f'{pace_min}:{pace_sec:02d}'
    else:
      pace_display = '0:00'
      dist_mi = 0.0

    splits.append({
        'Split': int(idx + 1),
        'Distance (mi)': round(dist_mi, 2),
        'Time': f'{int(time_sec // 60)}m {int(time_sec % 60)}s',
        'Pace (/mi)': pace_display,
        'Avg HR (bpm)': round(avg_hr, 1) if avg_hr is not None else None,
    })

  return pd.DataFrame(splits)


def calculate_best_efforts(df_records):
  target_distances_m = {
      '400m': 400.0,
      '800m': 800.0,
      '1000m': 1000.0,
      '1 Mile': 1609.34,
      '2 Mile': 3218.68,
      '5K': 5000.0,
      '10K': 10000.0,
      'Half Marathon': 21097.5,
  }

  if (
      df_records.empty
      or 'distance' not in df_records.columns
      or 'datetime' not in df_records.columns
  ):
    return pd.DataFrame()

  df = df_records.dropna(subset=['distance', 'datetime']).sort_values(
      'datetime'
  )
  distances = df['distance'].values
  times = df['datetime'].values
  results = []

  for label, target_dist in target_distances_m.items():
    best_time_sec = float('inf')
    n = len(df)
    left = 0
    for right in range(n):
      while (distances[right] - distances[left]) >= target_dist:
        dur_sec = (
            pd.to_datetime(times[right]) - pd.to_datetime(times[left])
        ).total_seconds()
        if dur_sec < best_time_sec:
          best_time_sec = dur_sec
        left += 1

    if best_time_sec != float('inf') and best_time_sec > 0:
      sec_per_mile = (best_time_sec / target_dist) * 1609.34
      pace_min = int(sec_per_mile // 60)
      pace_sec = int(sec_per_mile % 60)
      results.append({
          'Distance Effort': label,
          'Time': f'{int(best_time_sec // 60)}m {int(best_time_sec % 60)}s',
          'Pace': f'{pace_min}:{pace_sec:02d} /mi',
      })

  return pd.DataFrame(results)


# --- NAVIGATION & SIDEBAR ---
st.sidebar.header('Navigation')
page_mode = st.sidebar.radio(
    'View Mode',
    [
        'Single Activity Inspector',
        'Training History & Fitness',
        'Route Heatmap',
    ],
)

st.sidebar.markdown('---')
st.sidebar.header('Add Runs to Vault')
uploaded_files = st.sidebar.file_uploader(
    'Upload .fit files to backlog', type=['fit'], accept_multiple_files=True
)

if 'ingested_files' not in st.session_state:
  st.session_state['ingested_files'] = set()

if uploaded_files:
  newly_added = 0
  for uf in uploaded_files:
    file_sig = f'{uf.name}_{uf.size}'
    if file_sig not in st.session_state['ingested_files']:
      with st.sidebar.spinner(f'Saving {uf.name} to vault...'):
        ingest_fit_bytes(uf.name, uf.getvalue(), DB_NAME)
        st.session_state['ingested_files'].add(file_sig)
        newly_added += 1
  if newly_added > 0:
    push_vault_to_cloud(DB_NAME)
    st.sidebar.success(f'Added {newly_added} run(s) to persistent vault.')

active_hr_max, active_hr_rest, peak_src, last_upd = get_athlete_baselines(
    DB_NAME
)

with st.sidebar.expander('Dynamic Athlete Baselines', expanded=False):
  st.caption(f'**Peak Source:** `{peak_src}`')
  new_max = st.number_input(
      'Max Heart Rate (bpm)',
      min_value=120.0,
      max_value=230.0,
      value=float(active_hr_max),
      step=1.0,
  )
  new_rest = st.number_input(
      'Resting Heart Rate (bpm)',
      min_value=30.0,
      max_value=100.0,
      value=float(active_hr_rest),
      step=1.0,
  )
  if st.button('Update Baselines & Recalculate TRIMP'):
    active_hr_max, active_hr_rest, _ = update_athlete_baselines(
        DB_NAME, manual_hr_max=new_max, new_hr_rest=new_rest
    )
    push_vault_to_cloud(DB_NAME)
    st.success('Vault baselines & TRIMP updated.')
    st.rerun()

  if st.button('Sync Local .fit Folder to Vault'):
    with st.spinner('Ingesting local .fit files & updating vault...'):
      build_activity_database('fit test files')
      push_vault_to_cloud(DB_NAME)
    st.success('Vault synced.')
    st.rerun()


active_zones = calculate_karvonen_zones(active_hr_max, active_hr_rest)

# --- VIEW 1: TRAINING HISTORY & FITNESS ---
if page_mode == 'Training History & Fitness':
  st.subheader('Master Training History & Physiological Load')

  conn = sqlite3.connect(DB_NAME)
  df_history = pd.read_sql('SELECT * FROM activities', conn)
  conn.close()

  if not df_history.empty and 'date' in df_history.columns:
    df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_history = df_history.sort_values('date').dropna(subset=['date'])

    df_pmc = compute_performance_management_chart(df_history)

    current_ctl = df_pmc['CTL'].iloc[-1] if not df_pmc.empty else 0.0
    current_atl = df_pmc['ATL'].iloc[-1] if not df_pmc.empty else 0.0
    current_tsb = df_pmc['TSB'].iloc[-1] if not df_pmc.empty else 0.0

    total_lifetime_miles = df_history['distance_miles'].sum()
    total_activities = len(df_history)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Total Mileage', f'{total_lifetime_miles:.2f} mi')
    m2.metric('Activities', total_activities)
    m3.metric('Fitness (CTL)', f'{current_ctl:.1f}')
    m4.metric('Fatigue (ATL)', f'{current_atl:.1f}')
    m5.metric(
        'Form (TSB)',
        f'{current_tsb:.1f}',
        delta=(
            'Fresh'
            if current_tsb > 5
            else ('Optimal' if current_tsb >= -10 else 'High Fatigue')
        ),
    )

    st.markdown('---')

    st.markdown('### Active Karvonen Heart Rate Zones')
    st.markdown(
        f'*Dynamically calibrated from $HR_{{max}}$ = **{active_hr_max:.0f}'
        f' bpm** (Source: `{peak_src}`) and $HR_{{rest}}$ ='
        f' **{active_hr_rest:.0f} bpm**.*'
    )

    zone_cols = st.columns(5)
    for i, (zone_name, (low, high)) in enumerate(active_zones.items()):
      zone_cols[i].metric(zone_name, f'{low} - {high} bpm')

    st.markdown('---')
    st.markdown('### Weekly Time-in-Zone & 80/20 Polarization')
    z_cols = ['z1_min', 'z2_min', 'z3_min', 'z4_min', 'z5_min']
    if all(c in df_history.columns for c in z_cols):
      df_zones = df_history[['date'] + z_cols].dropna().copy()
      if not df_zones.empty and df_zones[z_cols].sum().sum() > 0:
        total_z_mins = df_zones[z_cols].sum().sum()
        easy_mins = df_zones['z1_min'].sum() + df_zones['z2_min'].sum()
        mod_mins = df_zones['z3_min'].sum()
        hard_mins = df_zones['z4_min'].sum() + df_zones['z5_min'].sum()

        p1, p2, p3 = st.columns(3)
        p1.metric(
            'Low Intensity (Z1-Z2)',
            f'{(easy_mins / total_z_mins * 100):.1f}%',
            delta='Target: ~80%',
        )
        p2.metric(
            'Moderate / Tempo (Z3)',
            f'{(mod_mins / total_z_mins * 100):.1f}%',
        )
        p3.metric(
            'High Intensity (Z4-Z5)',
            f'{(hard_mins / total_z_mins * 100):.1f}%',
            delta='Target: ~20%',
        )

        df_zones['Week'] = (
            df_zones['date'].dt.to_period('W').apply(lambda r: r.start_time)
        )
        weekly_zones = (
            df_zones.groupby('Week')[z_cols]
            .sum()
            .rename(
                columns={
                    'z1_min': 'Z1 Recovery',
                    'z2_min': 'Z2 Endurance',
                    'z3_min': 'Z3 Tempo',
                    'z4_min': 'Z4 Threshold',
                    'z5_min': 'Z5 VO2 Max',
                }
            )
        )
        st.bar_chart(
            weekly_zones,
            color=['#9ca3af', '#38bdf8', '#4ade80', '#fb923c', '#ef4444'],
        )
      else:
        st.info(
            'Click Sync Local .fit Folder to Vault in the sidebar to populate'
            ' Time-in-Zone minutes.'
        )

    st.markdown('---')

    st.markdown('### Performance Management Chart (Fitness, Fatigue & Form)')
    st.markdown(
        '* **Fitness (CTL):** Long-term training load (42-day exponential'
        ' average).\n* **Fatigue (ATL):** Short-term systemic strain (7-day'
        ' exponential average).\n* **Form (TSB):** Readiness balance.'
        ' Positive = Rested, Negative = Productive Overload.'
    )

    if not df_pmc.empty:
      chart_data = df_pmc.set_index('date')[['CTL', 'ATL', 'TSB']]
      st.line_chart(
          chart_data,
          color=['#dc2626', '#38bdf8', '#4ade80'],
      )

    st.markdown('---')

    st.markdown('### Efficiency Factor (EF) Progression by Workout Type')
    if (
        'efficiency_factor' in df_history.columns
        and 'workout_tag' in df_history.columns
    ):
      available_tags = ['All'] + sorted(
          df_history['workout_tag'].dropna().unique().tolist()
      )
      filter_tag = st.selectbox(
          'Filter EF Trend by Workout Type', available_tags
      )

      df_ef = df_history.dropna(subset=['efficiency_factor']).copy()
      if filter_tag != 'All':
        df_ef = df_ef[df_ef['workout_tag'] == filter_tag]

      if not df_ef.empty:
        st.line_chart(
            df_ef.set_index('date')[['efficiency_factor']], color='#dc2626'
        )
      else:
        st.info(
            'No Efficiency Factor data available for this workout category yet.'
        )

    st.markdown('---')

    df_all_prs = get_all_time_prs(DB_NAME)

    st.markdown('### All-Time Recorded Personal Bests (From .fit Vault)')
    st.markdown(
        '*Actual fastest sustained efforts extracted directly from your'
        ' GPS/time streams (never overwritten by predictions).*'
    )

    if not df_all_prs.empty:
      pr_display = df_all_prs[[
          'distance_label',
          'Recorded PR Time',
          'Recorded PR Pace',
          'activity_date',
          'file_name',
      ]].rename(
          columns={
              'distance_label': 'Distance',
              'activity_date': 'Date Achieved',
              'file_name': 'Source File',
          }
      )
      st.dataframe(pr_display, use_container_width=True, hide_index=True)

      st.markdown('---')
      st.markdown(
          '### Riegel Race Predictor (Actual PRs vs. Predicted Potential)'
      )
      st.markdown(
          'Select one of your **Actual Recorded PRs** below to project'
          ' theoretical race potential across other distances ($T_2 = T_1'
          ' \\times (D_2 / D_1)^{1.06}$).'
      )

      pr_options = {
          f"{row['distance_label']} — {row['Recorded PR Time']} ({row['activity_date']})": (
              row['distance_meters'],
              row['duration_seconds'],
          )
          for _, row in df_all_prs.iterrows()
      }
      default_pr_idx = 0
      for idx, k in enumerate(pr_options.keys()):
        if k.startswith('5K') or k.startswith('1 Mile'):
          default_pr_idx = idx
          break

      selected_pr_label = st.selectbox(
          'Select Recorded PR Benchmark for Projection',
          list(pr_options.keys()),
          index=default_pr_idx,
      )
      benchmark_dist_m, benchmark_time_sec = pr_options[selected_pr_label]

      df_predictions = predict_race_times(
          benchmark_dist_m, benchmark_time_sec, df_all_prs
      )
      st.dataframe(df_predictions, use_container_width=True, hide_index=True)
    else:
      st.info(
          'Click Sync Local .fit Folder to Vault in the sidebar to extract'
          ' your actual recorded PRs and unlock dynamic race predictions.'
      )

    st.markdown('---')
    st.markdown('### Complete Activity Database (SQLite Vault)')
    st.dataframe(df_history, use_container_width=True)
  else:
    st.warning(
        'No activities found in training_vault.db. Upload .fit files in the'
        ' sidebar or click Sync Local .fit Folder to Vault.'
    )

# --- VIEW 2: ROUTE HEATMAP ---
elif page_mode == 'Route Heatmap':
  render_heatmap_tab()

# --- VIEW 3: DATABASE-DRIVEN SINGLE ACTIVITY INSPECTOR ---
else:
  conn = sqlite3.connect(DB_NAME)
  df_saved_runs = pd.read_sql(
      '''
      SELECT
          a.file_name,
          a.date,
          a.distance_miles,
          a.workout_tag,
          CASE WHEN f.file_name IS NOT NULL THEN 1 ELSE 0 END AS has_raw
      FROM activities a
      LEFT JOIN fit_files f ON a.file_name = f.file_name
      ORDER BY a.date DESC
      ''',
      conn,
  )
  conn.close()

  if not df_saved_runs.empty:
    run_options = {}
    for _, row in df_saved_runs.iterrows():
      date_short = str(row['date'])[:10] if row['date'] else 'Unknown Date'
      dist_val = (
          row['distance_miles'] if pd.notna(row['distance_miles']) else 0.0
      )
      tag_val = row['workout_tag'] if row['workout_tag'] else 'Base Run'
      label = f'{date_short} — {dist_val:.2f} mi ({tag_val})'
      if label in run_options:
        label = f"{label} [{row['file_name']}]"
      run_options[label] = (row['file_name'], row['has_raw'], tag_val)

    st.sidebar.markdown('---')
    st.sidebar.header('Select Saved Run')
    selected_label = st.sidebar.selectbox(
        'Saved Activities in Vault', list(run_options.keys())
    )
    selected_fname, has_raw_blob, saved_tag = run_options[selected_label]

    workout_categories = [
        'Recovery / Easy',
        'Base Run',
        'Long Run',
        'Tempo / Quality Long Run',
        'Intervals / Speedwork',
        'Tempo / Threshold',
        'Race',
    ]
    default_index = (
        workout_categories.index(saved_tag)
        if saved_tag in workout_categories
        else 1
    )

    st.sidebar.markdown('---')
    st.sidebar.header('Workout Classification')
    selected_workout_type = st.sidebar.selectbox(
        'Workout Tag (Saved in Vault)',
        workout_categories,
        index=default_index,
    )

    if st.sidebar.button('Save Tag Update to Vault'):
      conn = sqlite3.connect(DB_NAME)
      conn.execute(
          'UPDATE activities SET workout_tag = ? WHERE file_name = ?',
          (selected_workout_type, selected_fname),
      )
      conn.commit()
      conn.close()
      push_vault_to_cloud(DB_NAME)
      st.sidebar.success(f'Updated tag to "{selected_workout_type}".')
      st.rerun()

    if not has_raw_blob:
      st.warning(
          f'Run summary for `{selected_fname}` is in the database, but its raw'
          ' .fit telemetry stream has not been archived yet. Click **Sync Local'
          ' .fit Folder to Vault** inside the Dynamic Athlete Baselines'
          ' sidebar menu (or re-upload the file) once to store its full stream.'
      )
    else:
      conn = sqlite3.connect(DB_NAME)
      gz_row = conn.execute(
          'SELECT raw_gz FROM fit_files WHERE file_name = ?',
          (selected_fname,),
      ).fetchone()
      conn.close()

      raw_bytes = gzip.decompress(gz_row[0])
      data = parse_fit_bytes_raw(raw_bytes)

      df_records = clean_records(data['records'])
      df_native_splits = extract_native_laps(data['laps'])
      df_best_efforts = calculate_best_efforts(df_records)

      total_dist_mi = (
          df_records['distance'].max() / 1609.34
          if not df_records.empty and 'distance' in df_records.columns
          else 0.0
      )
      total_time_sec = df_records['dt'].sum() if not df_records.empty else 0.0

      avg_hr_run = (
          df_records['heart_rate'].mean()
          if 'heart_rate' in df_records.columns
          and df_records['heart_rate'].notna().any()
          else None
      )

      max_hr_run = None
      if (
          'heart_rate' in df_records.columns
          and df_records['heart_rate'].notna().any()
      ):
        valid_hrs = df_records.loc[
            (df_records['heart_rate'] > 40) & (df_records['heart_rate'] <= 220),
            'heart_rate',
        ]
        if not valid_hrs.empty:
          max_hr_run = float(valid_hrs.max())

      duration_min = total_time_sec / 60.0
      workout_trimp = compute_banister_trimp(
          duration_min, avg_hr_run, active_hr_max, active_hr_rest
      )
      ef_score = calculate_efficiency_factor(df_records)
      decoupling_score = calculate_aerobic_decoupling(df_records)
      df_tiz = compute_time_in_zones(df_records, active_zones)

      st.markdown(
          f'**Selected Activity:** `{selected_fname}` &nbsp;|&nbsp;'
          f' **Classification:** `{selected_workout_type}`'
      )

      col1, col2, col3, col4, col5, col6 = st.columns(6)
      col1.metric('Total Distance', f'{total_dist_mi:.2f} mi')
      col2.metric(
          'Active Moving Time',
          f'{int(total_time_sec // 3600)}h {int((total_time_sec % 3600) // 60)}m'
          if total_time_sec > 3600
          else f'{int(total_time_sec // 60)}m {int(total_time_sec % 60)}s',
      )
      col3.metric(
          'Avg / Max HR',
          f'{avg_hr_run:.0f} / {max_hr_run:.0f} bpm'
          if (avg_hr_run and max_hr_run)
          else 'N/A',
      )
      col4.metric('Relative Effort (TRIMP)', f'{workout_trimp}')
      col5.metric('Efficiency Factor', f'{ef_score}' if ef_score else 'N/A')
      col6.metric(
          'Aerobic Decoupling',
          f'{decoupling_score}%' if decoupling_score is not None else 'N/A',
          delta=(
              'Stable'
              if decoupling_score is not None and decoupling_score < 5
              else 'Drift Detected'
          ),
          delta_color='inverse',
      )

      st.markdown('---')

      tab1, tab2, tab3, tab4 = st.tabs([
          'Pace & HR Profile',
          'Native Splits',
          'Best Effort PRs',
          'Time in Zone',
      ])

      with tab1:
        st.subheader('High-Resolution Rolling Pace & Heart Rate')
        plt.rcParams['font.family'] = 'serif'
        fig, ax1 = plt.subplots(figsize=(10, 4.5), facecolor='#120606')
        ax1.set_facecolor('#120606')

        running_mask = df_records['enhanced_speed'] >= 2.0
        clean_speed = df_records['enhanced_speed'].where(running_mask, np.nan)
        sec_per_mile = 1609.34 / clean_speed
        rolling_pace = (
            sec_per_mile.rolling(window=30, min_periods=5).mean() / 60
        )
        dist_miles = df_records['distance'] / 1609.34

        # White Rolling Pace Line
        ax1.plot(
            dist_miles,
            rolling_pace,
            color='#ffffff',
            linewidth=1.6,
            alpha=0.92,
            label='30s Rolling Pace',
        )
        ax1.set_xlabel('Distance (Miles)', color='#f5ebeb')
        ax1.set_ylabel('Pace (min/mi)', color='#ffffff')
        ax1.tick_params(axis='x', colors='#f5ebeb')
        ax1.tick_params(axis='y', colors='#ffffff')

        for spine in ax1.spines.values():
          spine.set_color('#451010')

        def pace_formatter(x, pos):
          if np.isnan(x) or x <= 0:
            return ''
          mins = int(x)
          secs = int(round((x - mins) * 60))
          if secs == 60:
            mins += 1
            secs = 0
          return f'{mins}:{secs:02d}'

        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(pace_formatter))
        ax1.invert_yaxis()
        ax1.grid(True, linestyle='--', color='#3b0d0d', alpha=0.6)

        if (
            'heart_rate' in df_records.columns
            and df_records['heart_rate'].notna().any()
        ):
          ax2 = ax1.twinx()
          rolling_hr = (
              df_records['heart_rate'].rolling(window=30, min_periods=5).mean()
          )

          x_vals = dist_miles.to_numpy()
          y_vals = rolling_hr.to_numpy()
          points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
          segments = np.concatenate([points[:-1], points[1:]], axis=1)

          valid_seg = ~(
              np.isnan(x_vals[:-1])
              | np.isnan(x_vals[1:])
              | np.isnan(y_vals[:-1])
              | np.isnan(y_vals[1:])
          )
          segments = segments[valid_seg]
          seg_hrs = 0.5 * (y_vals[:-1][valid_seg] + y_vals[1:][valid_seg])
          seg_colors = [
              get_hr_zone_color(hr_v, active_zones) for hr_v in seg_hrs
          ]

          lc = LineCollection(
              segments, colors=seg_colors, linewidths=2.0, alpha=0.95
          )
          ax2.add_collection(lc)

          valid_hr_vals = y_vals[~np.isnan(y_vals)]
          if len(valid_hr_vals) > 0:
            ax2.set_ylim(valid_hr_vals.min() - 5, valid_hr_vals.max() + 5)

          ax2.set_ylabel('Heart Rate (bpm)', color='#f5ebeb')
          ax2.tick_params(axis='y', colors='#f5ebeb')
          for spine in ax2.spines.values():
            spine.set_color('#451010')

          # Zone color legend above chart
          legend_handles = [
              Line2D([0], [0], color='#ffffff', lw=1.6, label='Pace'),
              Line2D([0], [0], color='#9ca3af', lw=2.2, label='Z0/Z1'),
              Line2D([0], [0], color='#38bdf8', lw=2.2, label='Z2'),
              Line2D([0], [0], color='#4ade80', lw=2.2, label='Z3'),
              Line2D([0], [0], color='#fb923c', lw=2.2, label='Z4'),
              Line2D([0], [0], color='#ef4444', lw=2.2, label='Z5'),
          ]
          ax1.legend(
              handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.13),
              ncol=6,
              frameon=False,
              labelcolor='#f5ebeb',
              fontsize=9.5,
          )

        fig.tight_layout()
        st.pyplot(fig)

      with tab2:
        st.subheader('Garmin Native Lap Splits')
        st.dataframe(
            df_native_splits, use_container_width=True, hide_index=True
        )

      with tab3:
        st.subheader('Calculated Best Effort PRs')
        if not df_best_efforts.empty:
          st.dataframe(
              df_best_efforts, use_container_width=True, hide_index=True
          )
        else:
          st.warning('Not enough distance data to calculate best efforts.')

      with tab4:
        st.subheader('Karvonen Heart Rate Time-in-Zone Breakdown')
        if df_tiz['Minutes'].sum() > 0:
          c_chart, c_table = st.columns([3, 2])
          with c_chart:
            fig_z, ax_z = plt.subplots(figsize=(6.5, 3.5), facecolor='#120606')
            ax_z.set_facecolor('#120606')
            bar_colors = [
                ZONE_PALETTE.get(z, '#ef4444') for z in df_tiz['Zone']
            ]
            ax_z.bar(
                df_tiz['Zone'].str.replace(' \\(.*\\)', '', regex=True),
                df_tiz['Minutes'],
                color=bar_colors,
                width=0.55,
            )
            ax_z.set_ylabel('Minutes', color='#f5ebeb')
            ax_z.tick_params(axis='x', colors='#f5ebeb')
            ax_z.tick_params(axis='y', colors='#f5ebeb')
            ax_z.grid(axis='y', linestyle='--', color='#3b0d0d', alpha=0.6)
            for spine in ax_z.spines.values():
              spine.set_color('#451010')
            fig_z.tight_layout()
            st.pyplot(fig_z)
          with c_table:
            st.dataframe(df_tiz, use_container_width=True, hide_index=True)
        else:
          st.warning(
              'No valid heart rate telemetry available for zone analysis.'
          )
  else:
    st.info(
        'No saved activities in the vault yet. Upload .fit files in the'
        ' sidebar or open Dynamic Athlete Baselines and click Sync Local .fit'
        ' Folder to Vault.'
    )