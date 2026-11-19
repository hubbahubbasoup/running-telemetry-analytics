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
from aggregate_runs import (
    backfill_activity_locations,
    build_activity_database,
    deduplicate_vault,
    delete_activity_from_vault,
    init_db,
    ingest_fit_bytes,
    regeocode_all_activities,
)
from cloud_vault import (
    check_password_gate,
    ensure_local_vault,
    is_cloud_configured,
    pull_fit_gz_from_cloud,
    pull_vault_from_cloud,
    push_vault_to_cloud,
)
from hr_analytics import (
    STANDARD_DISTANCES_M,
    calculate_karvonen_zones,
    compute_banister_trimp,
    compute_performance_management_chart,
    compute_time_in_zones,
    delete_manual_pr,
    extract_best_efforts_from_df,
    format_seconds_to_time,
    get_all_time_prs,
    get_athlete_baselines,
    parse_time_str_to_seconds,
    predict_race_times,
    save_best_efforts_to_db,
    set_manual_pr,
    update_athlete_baselines,
)
from route_heatmap import render_heatmap_tab, render_single_activity_map

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='Running Telemetry & Physiological Analytics',
    layout='wide',
)

# --- DARK-RED THEME, CALM DARK-GREY SIDEBAR & NON-TRUNCATED METRIC HEADINGS ---
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
        padding: 12px 14px;
        border-radius: 6px;
    }

    /* Prevent metric headings from truncating with ellipsis (...) */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
        font-size: 0.98rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
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

if backfill_activity_locations(DB_NAME):
  push_vault_to_cloud(DB_NAME)

ZONE_PALETTE = {
    'Zone 1 (Recovery)': '#9ca3af',   # Grey (Zone 0/1)
    'Zone 2 (Endurance)': '#38bdf8',  # Light Blue
    'Zone 3 (Tempo)': '#4ade80',      # Light Green
    'Zone 4 (Threshold)': '#fb923c',  # Light-ish Orange
    'Zone 5 (VO2 Max)': '#ef4444',    # Red
}

# --- HELPER FUNCTIONS ---


def pace_str_to_seconds(pace_str):
  """Converts 'MM:SS /mi' into float seconds per mile for accurate sorting/filtering."""
  if not pace_str or not isinstance(pace_str, str):
    return 9999.0
  clean = pace_str.replace('/mi', '').strip()
  if ':' not in clean:
    return 9999.0
  try:
    m, s = clean.split(':')
    sec = int(m) * 60 + int(s)
    return float(sec) if sec > 0 else 9999.0
  except Exception:
    return 9999.0


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
  skipped_files = 0
  for uf in uploaded_files:
    file_sig = f'{uf.name}_{uf.size}'
    if file_sig not in st.session_state['ingested_files']:
      with st.sidebar.spinner(f'Processing {uf.name}...'):
        res = ingest_fit_bytes(uf.name, uf.getvalue(), DB_NAME)
        st.session_state['ingested_files'].add(file_sig)
        if res is not None:
          newly_added += 1
        else:
          skipped_files += 1
  if newly_added > 0:
    push_vault_to_cloud(DB_NAME)
    st.sidebar.success(f'Added {newly_added} run(s) to persistent vault.')
  if skipped_files > 0:
    st.sidebar.info(f'Skipped {skipped_files} non-running or duplicate file(s).')

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

  if peak_src == 'Manual Override':
    if st.button('Reset Max HR to Auto-Detected Peak'):
      active_hr_max, active_hr_rest, _ = update_athlete_baselines(
          DB_NAME, reset_auto_max=True
      )
      push_vault_to_cloud(DB_NAME)
      st.success('Re-enabled auto-detected peak HR.')
      st.rerun()

with st.sidebar.expander('Vault Maintenance', expanded=False):
  if st.button('Remove Duplicate Runs'):
    removed = deduplicate_vault(DB_NAME)
    if removed:
      push_vault_to_cloud(DB_NAME)
      st.success(f'Removed {len(removed)} duplicate run(s): {", ".join(removed)}')
      st.rerun()
    else:
      st.info('No duplicate runs found in vault.')

  if st.button('Re-Geocode All Run Locations'):
    prog_bar = st.progress(0.0, text='Resolving starting cities via OpenStreetMap...')

    def _on_prog(done, total):
      if total > 0:
        prog_bar.progress(
            min(1.0, done / total),
            text=f'Resolving unique location {done} / {total}...',
        )

    n_runs, n_grids = regeocode_all_activities(DB_NAME, progress_callback=_on_prog)
    push_vault_to_cloud(DB_NAME)
    prog_bar.empty()
    st.success(f'Re-geocoded {n_runs} runs across {n_grids} unique starting areas.')
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

    # --- PERIOD SUMMARY SELECTOR (WEEK / MONTH / YEAR) ---
    df_history['week_start'] = (
        df_history['date'].dt.to_period('W-SUN').apply(lambda r: r.start_time)
    )
    df_history['month_label'] = df_history['date'].dt.strftime('%Y-%m')
    df_history['year_label'] = df_history['date'].dt.strftime('%Y')

    c_mode, c_sel = st.columns([1.2, 2.8])
    with c_mode:
      period_granularity = st.radio(
          'Summary Timeframe',
          ['Week (Mon-Sun)', 'Month', 'Year'],
          horizontal=True,
      )

    if period_granularity == 'Week (Mon-Sun)':
      all_periods = sorted(df_history['week_start'].unique(), reverse=True)
      period_labels = {
          f"{pd.Timestamp(p).strftime('%Y-%m-%d')} to {(pd.Timestamp(p) + pd.Timedelta(days=6)).strftime('%Y-%m-%d')}": p
          for p in all_periods
      }
      with c_sel:
        chosen_lbl = st.selectbox('Select Week', list(period_labels.keys()))
      chosen_val = period_labels[chosen_lbl]
      df_period = df_history[df_history['week_start'] == chosen_val]
      prev_val = pd.Timestamp(chosen_val) - pd.Timedelta(days=7)
      df_prev = df_history[df_history['week_start'] == prev_val]
      unit_name = 'wk'
    elif period_granularity == 'Month':
      all_periods = sorted(df_history['month_label'].unique(), reverse=True)
      with c_sel:
        chosen_val = st.selectbox('Select Month', all_periods)
      df_period = df_history[df_history['month_label'] == chosen_val]
      idx_p = all_periods.index(chosen_val)
      df_prev = (
          df_history[df_history['month_label'] == all_periods[idx_p + 1]]
          if idx_p + 1 < len(all_periods)
          else pd.DataFrame()
      )
      unit_name = 'mo'
    else:
      all_periods = sorted(df_history['year_label'].unique(), reverse=True)
      with c_sel:
        chosen_val = st.selectbox('Select Year', all_periods)
      df_period = df_history[df_history['year_label'] == chosen_val]
      idx_p = all_periods.index(chosen_val)
      df_prev = (
          df_history[df_history['year_label'] == all_periods[idx_p + 1]]
          if idx_p + 1 < len(all_periods)
          else pd.DataFrame()
      )
      unit_name = 'yr'

    period_miles = df_period['distance_miles'].sum()
    prev_miles = df_prev['distance_miles'].sum() if not df_prev.empty else None
    miles_delta = (
        f'{period_miles - prev_miles:+.1f} mi vs prior {unit_name}'
        if prev_miles is not None
        else f'{len(df_period)} runs'
    )

    period_runs = len(df_period)
    prev_runs = len(df_prev) if not df_prev.empty else None
    runs_delta = (
        f'{period_runs - prev_runs:+d} vs prior {unit_name}'
        if prev_runs is not None
        else f'{(period_miles / max(1, period_runs)):.1f} mi/run'
    )

    ctl_status = (
        'Peak Base'
        if current_ctl >= 65
        else (
            'Solid Base'
            if current_ctl >= 40
            else ('Building Base' if current_ctl >= 20 else 'Developing')
        )
    )
    atl_status = (
        'Low Strain'
        if current_atl < 25
        else ('Productive Load' if current_atl <= 55 else 'High Strain')
    )
    tsb_status = (
        'Fresh'
        if current_tsb > 5
        else ('Optimal' if current_tsb >= -10 else 'High Fatigue')
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        f'{period_granularity.split()[0]} Mileage',
        f'{period_miles:.2f} mi',
        delta=miles_delta,
    )
    m2.metric(
        f'{period_granularity.split()[0]} Activities',
        period_runs,
        delta=runs_delta,
    )
    m3.metric('Fitness (CTL)', f'{current_ctl:.1f}', delta=ctl_status)
    m4.metric(
        'Fatigue (ATL)',
        f'{current_atl:.1f}',
        delta=atl_status,
        delta_color='normal' if current_atl <= 55 else 'inverse',
    )
    m5.metric(
        'Form (TSB)',
        f'{current_tsb:.1f}',
        delta=tsb_status,
        delta_color='normal' if current_tsb >= -10 else 'inverse',
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

    # --- WEEKLY MILEAGE VOLUME CHART (MONDAY - SUNDAY) ---
    st.markdown('### Weekly Mileage Volume (Monday – Sunday)')
    weekly_mi_series = df_history.groupby('week_start')['distance_miles'].sum()
    full_weeks = pd.date_range(
        start=weekly_mi_series.index.min(),
        end=weekly_mi_series.index.max(),
        freq='W-MON',
    )
    df_weekly_mi = pd.DataFrame(index=full_weeks)
    df_weekly_mi.index.name = 'Week (Mon)'
    df_weekly_mi['Weekly Miles'] = weekly_mi_series.reindex(
        full_weeks, fill_value=0.0
    )
    df_weekly_mi['4-Week Rolling Avg'] = (
        df_weekly_mi['Weekly Miles'].rolling(window=4, min_periods=1).mean()
    )

    recent_4wk_avg = float(df_weekly_mi['4-Week Rolling Avg'].iloc[-1])
    peak_wk_mi = float(df_weekly_mi['Weekly Miles'].max())

    wm_col1, wm_col2, wm_col3, wm_col4 = st.columns([1.2, 1.2, 1.3, 1.3])
    wm_col1.metric('Recent 4-Week Avg', f'{recent_4wk_avg:.1f} mi/wk')
    wm_col2.metric('Peak Weekly Volume', f'{peak_wk_mi:.1f} mi')
    with wm_col3:
      show_wk_bars = st.toggle('Show Weekly Miles', value=True)
    with wm_col4:
      show_wk_roll = st.toggle('Show 4-Week Rolling Avg', value=True)

    wk_cols_to_plot = []
    wk_colors = []
    if show_wk_bars:
      wk_cols_to_plot.append('Weekly Miles')
      wk_colors.append('#dc2626')
    if show_wk_roll:
      wk_cols_to_plot.append('4-Week Rolling Avg')
      wk_colors.append('#38bdf8')

    if wk_cols_to_plot:
      if show_wk_bars and not show_wk_roll:
        st.bar_chart(df_weekly_mi[['Weekly Miles']], color=['#dc2626'])
      else:
        st.line_chart(df_weekly_mi[wk_cols_to_plot], color=wk_colors)

    st.markdown('---')
    st.markdown('### Weekly Time-in-Zone & 80/20 Polarization')
    z_cols = ['z1_min', 'z2_min', 'z3_min', 'z4_min', 'z5_min']
    if all(c in df_history.columns for c in z_cols):
      df_zones = df_history[['date', 'week_start'] + z_cols].dropna().copy()
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

        tz1, tz2, tz3, tz4, tz5 = st.columns(5)
        t_z1 = tz1.toggle('Z1 Recovery', value=True)
        t_z2 = tz2.toggle('Z2 Endurance', value=True)
        t_z3 = tz3.toggle('Z3 Tempo', value=True)
        t_z4 = tz4.toggle('Z4 Threshold', value=True)
        t_z5 = tz5.toggle('Z5 VO2 Max', value=True)

        weekly_zones = (
            df_zones.groupby('week_start')[z_cols]
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
        zone_toggle_map = [
            ('Z1 Recovery', t_z1, '#9ca3af'),
            ('Z2 Endurance', t_z2, '#38bdf8'),
            ('Z3 Tempo', t_z3, '#4ade80'),
            ('Z4 Threshold', t_z4, '#fb923c'),
            ('Z5 VO2 Max', t_z5, '#ef4444'),
        ]
        active_z_names = [name for name, on, _ in zone_toggle_map if on]
        active_z_colors = [col for _, on, col in zone_toggle_map if on]

        if active_z_names:
          st.bar_chart(
              weekly_zones[active_z_names],
              color=active_z_colors,
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
      pmc_t1, pmc_t2, pmc_t3 = st.columns(3)
      show_ctl = pmc_t1.toggle('Show Fitness (CTL)', value=True)
      show_atl = pmc_t2.toggle('Show Fatigue (ATL)', value=True)
      show_tsb = pmc_t3.toggle('Show Form (TSB)', value=True)

      pmc_series = []
      pmc_colors = []
      if show_ctl:
        pmc_series.append('CTL')
        pmc_colors.append('#dc2626')
      if show_atl:
        pmc_series.append('ATL')
        pmc_colors.append('#38bdf8')
      if show_tsb:
        pmc_series.append('TSB')
        pmc_colors.append('#4ade80')

      if pmc_series:
        chart_data = df_pmc.set_index('date')[pmc_series]
        st.line_chart(chart_data, color=pmc_colors)

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

    st.markdown('### All-Time Recorded & Calibrated Personal Bests')
    st.markdown(
        '*Displays your fastest sustained efforts from `.fit` streams merged'
        ' with any manual GPS calibration overrides.*'
    )

    with st.expander('Manual PR Calibration & Override Editor', expanded=False):
      st.caption(
          'Override GPS-overmeasured distances with your official track/road'
          ' PRs. Changes automatically propagate to the Race Predictor.'
      )
      pe1, pe2, pe3, pe4 = st.columns([1.3, 1.3, 1.2, 1.2])
      with pe1:
        edit_dist_lbl = st.selectbox(
            'Distance', list(STANDARD_DISTANCES_M.keys()), index=5
        )
      with pe2:
        edit_time_str = st.text_input(
            'Official Time (MM:SS or HH:MM:SS)', placeholder='e.g. 16:35'
        )
      with pe3:
        edit_date_str = st.text_input(
            'Date Achieved (YYYY-MM-DD)',
            value=pd.Timestamp.today().strftime('%Y-%m-%d'),
        )
      with pe4:
        st.write('')
        if st.button('Save Manual PR'):
          parsed_sec = parse_time_str_to_seconds(edit_time_str)
          if parsed_sec and parsed_sec > 0:
            set_manual_pr(edit_dist_lbl, parsed_sec, edit_date_str, DB_NAME)
            push_vault_to_cloud(DB_NAME)
            st.success(
                f'Saved {edit_dist_lbl} PR ({format_seconds_to_time(parsed_sec)}).'
            )
            st.rerun()
          else:
            st.error('Enter a valid time like 16:35 or 4:50.')
        if st.button('Reset to .fit PR'):
          delete_manual_pr(edit_dist_lbl, DB_NAME)
          push_vault_to_cloud(DB_NAME)
          st.info(f'Reverted {edit_dist_lbl} to auto-detected .fit PR.')
          st.rerun()

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
              'file_name': 'Source File / Override',
          }
      )
      st.dataframe(pr_display, use_container_width=True, hide_index=True)

      st.markdown('---')
      st.markdown(
          '### Physiological Race Predictor (Multi-PR + Fitness CTL Model)'
      )
      st.markdown(
          'Synthesizes your selected **Personal Bests** with your current'
          f' **Fitness (`CTL = {current_ctl:.1f}`)** and **Recent 4-Week'
          f' Mileage (`{recent_4wk_avg:.1f} mi/wk`)** to model realistic'
          ' endurance decay across longer distances.'
      )

      avail_pr_labels = df_all_prs['distance_label'].tolist()
      default_benchmarks = [
          d
          for d in ['800m', '1000m', '1 Mile', '2 Mile', '5K']
          if d in avail_pr_labels
      ]
      if not default_benchmarks:
        default_benchmarks = avail_pr_labels[:3]

      selected_pr_benchmarks = st.multiselect(
          'Active PR Benchmarks Included in Aggregate Model (Uncheck casual'
          ' un-raced distances)',
          options=avail_pr_labels,
          default=default_benchmarks,
      )

      if selected_pr_benchmarks:
        df_predictions = predict_race_times(
            df_prs=df_all_prs,
            current_ctl=current_ctl,
            weekly_miles_avg=recent_4wk_avg,
            selected_pr_labels=selected_pr_benchmarks,
        )
        st.dataframe(df_predictions, use_container_width=True, hide_index=True)
      else:
        st.warning('Select at least one PR benchmark above to project race times.')
    else:
      st.info(
          'Click Sync Local .fit Folder to Vault in the sidebar to extract'
          ' your actual recorded PRs and unlock dynamic race predictions.'
      )

    st.markdown('---')
    st.markdown('### Complete Activity Database & All-Time Vault Totals')
    total_lifetime_miles = df_history['distance_miles'].sum()
    total_activities = len(df_history)
    total_hours = df_history['duration_minutes'].sum() / 60.0
    first_run_date = str(df_history['date'].min())[:10]

    at1, at2, at3, at4 = st.columns(4)
    at1.caption(f'**All-Time Mileage:** {total_lifetime_miles:.2f} mi')
    at2.caption(f'**All-Time Runs:** {total_activities}')
    at3.caption(f'**Total Moving Time:** {total_hours:.1f} hrs')
    at4.caption(f'**Vault Span:** Since {first_run_date}')

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
          a.duration_minutes,
          a.avg_pace,
          a.workout_tag,
          COALESCE(a.city, 'Indoor') AS city,
          COALESCE(a.state, 'Treadmill') AS state,
          CASE WHEN f.file_name IS NOT NULL THEN 1 ELSE 0 END AS has_raw
      FROM activities a
      LEFT JOIN fit_files f ON a.file_name = f.file_name
      ORDER BY a.date DESC
      ''',
      conn,
  )
  conn.close()

  if not df_saved_runs.empty:
    df_saved_runs['pace_sec'] = df_saved_runs['avg_pace'].apply(
        pace_str_to_seconds
    )
    df_saved_runs['date_short'] = (
        df_saved_runs['date'].astype(str).str.slice(0, 10)
    )

    st.sidebar.markdown('---')
    st.sidebar.header('Search & Sort Runs')
    search_query = st.sidebar.text_input(
        'Search Date, City, State, Distance, or Pace',
        placeholder='e.g. Redwood City, Provo, 2026-10, 5:30',
    )

    with st.sidebar.expander('Filter by Location, Distance & Pace', expanded=False):
      all_states = sorted(
          [s for s in df_saved_runs['state'].dropna().unique() if s]
      )
      selected_states = st.multiselect('Filter by State', all_states)

      avail_cities_df = (
          df_saved_runs[df_saved_runs['state'].isin(selected_states)]
          if selected_states
          else df_saved_runs
      )
      all_cities = sorted(
          [c for c in avail_cities_df['city'].dropna().unique() if c]
      )
      selected_cities = st.multiselect('Filter by City', all_cities)

      max_dist_vault = max(
          1.0, float(df_saved_runs['distance_miles'].max() or 20.0)
      )
      dist_range = st.slider(
          'Distance Range (mi)',
          min_value=0.0,
          max_value=round(max_dist_vault + 0.5, 1),
          value=(0.0, round(max_dist_vault + 0.5, 1)),
          step=0.5,
      )

      pace_filter_max_min = st.slider(
          'Max Average Pace (min/mi)',
          min_value=4.0,
          max_value=15.0,
          value=15.0,
          step=0.25,
      )

    sort_col_label = st.sidebar.selectbox(
        'Order Activities By',
        [
            'Date',
            'Distance',
            'Average Pace',
            'Location (City)',
            'Location (State)',
        ],
    )
    sort_dir = st.sidebar.radio(
        'Order Direction', ['Descending', 'Ascending'], horizontal=True
    )

    df_filtered = df_saved_runs.copy()
    if search_query.strip():
      q = search_query.strip().lower()
      mask = (
          df_filtered['date_short'].str.lower().str.contains(q, na=False)
          | df_filtered['city'].str.lower().str.contains(q, na=False)
          | df_filtered['state'].str.lower().str.contains(q, na=False)
          | df_filtered['distance_miles'].astype(str).str.contains(q, na=False)
          | df_filtered['avg_pace'].str.lower().str.contains(q, na=False)
          | df_filtered['workout_tag'].str.lower().str.contains(q, na=False)
      )
      df_filtered = df_filtered[mask]

    if selected_states:
      df_filtered = df_filtered[df_filtered['state'].isin(selected_states)]
    if selected_cities:
      df_filtered = df_filtered[df_filtered['city'].isin(selected_cities)]

    df_filtered = df_filtered[
        df_filtered['distance_miles'].between(dist_range[0], dist_range[1])
    ]
    if pace_filter_max_min < 15.0:
      df_filtered = df_filtered[
          df_filtered['pace_sec'] <= (pace_filter_max_min * 60.0)
      ]

    sort_map = {
        'Date': 'date',
        'Distance': 'distance_miles',
        'Average Pace': 'pace_sec',
        'Location (City)': 'city',
        'Location (State)': 'state',
    }
    ascending_bool = sort_dir == 'Ascending'
    df_filtered = df_filtered.sort_values(
        by=sort_map[sort_col_label], ascending=ascending_bool
    )

    if df_filtered.empty:
      st.warning('No activities match your current search/filter criteria.')
      st.stop()

    run_options = {}
    for _, row in df_filtered.iterrows():
      date_short = row['date_short'] if row['date_short'] else 'Unknown Date'
      dist_val = (
          row['distance_miles'] if pd.notna(row['distance_miles']) else 0.0
      )
      city_only = row['city'] if row['city'] else 'Indoor'
      tag_val = row['workout_tag'] if row['workout_tag'] else 'Base Run'
      label = f'{date_short} — {dist_val:.2f} mi — {city_only} ({tag_val})'
      if label in run_options:
        label = f"{label} [{row['file_name']}]"
      run_options[label] = (
          row['file_name'],
          row['has_raw'],
          tag_val,
          city_only,
          row['state'],
      )

    st.sidebar.markdown('---')
    st.sidebar.header(f'Select Saved Run ({len(run_options)})')
    selected_label = st.sidebar.selectbox(
        'Saved Activities in Vault', list(run_options.keys())
    )
    selected_fname, has_raw_blob, saved_tag, sel_city, sel_state = run_options[
        selected_label
    ]

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

    with st.sidebar.expander('Delete Selected Run', expanded=False):
      confirm_del = st.checkbox(
          f'Confirm delete `{selected_fname}`', key=f'del_{selected_fname}'
      )
      if st.button('Delete Run from Vault') and confirm_del:
        delete_activity_from_vault(selected_fname, DB_NAME)
        push_vault_to_cloud(DB_NAME)
        st.sidebar.success(f'Deleted `{selected_fname}` from vault.')
        st.rerun()

    if not has_raw_blob:
      st.warning(
          f'Run summary for `{selected_fname}` is in the database, but its raw'
          ' .fit telemetry stream has not been archived yet.'
      )
    else:
      conn = sqlite3.connect(DB_NAME)
      gz_row = conn.execute(
          'SELECT raw_gz FROM fit_files WHERE file_name = ?',
          (selected_fname,),
      ).fetchone()
      conn.close()

      gz_bytes = (
          gz_row[0]
          if (gz_row and gz_row[0])
          else pull_fit_gz_from_cloud(selected_fname)
      )
      if not gz_bytes:
        st.error(
            f'Could not retrieve raw stream for `{selected_fname}` from cloud'
            ' storage.'
        )
        st.stop()

      raw_bytes = gzip.decompress(gz_bytes)
      data = parse_fit_bytes_raw(raw_bytes)

      session_dist_m = None
      session_timer_sec = None
      if not data['session'].empty:
        if 'total_distance' in data['session'].columns:
          valid_sd = data['session']['total_distance'].dropna()
          if not valid_sd.empty:
            session_dist_m = float(valid_sd.iloc[-1])
        if 'total_timer_time' in data['session'].columns:
          valid_st = data['session']['total_timer_time'].dropna()
          if not valid_st.empty:
            session_timer_sec = float(valid_st.iloc[-1])

      if (
          session_dist_m
          and not data['records'].empty
          and 'distance' in data['records'].columns
      ):
        raw_max_m = data['records']['distance'].max()
        if pd.notna(raw_max_m) and raw_max_m > 0:
          cal_ratio = session_dist_m / raw_max_m
          if abs(cal_ratio - 1.0) > 0.005:
            data['records']['distance'] = (
                data['records']['distance'] * cal_ratio
            )
            if 'enhanced_speed' in data['records'].columns:
              data['records']['enhanced_speed'] = (
                  data['records']['enhanced_speed'] * cal_ratio
              )

      df_records = clean_records(data['records'])
      df_native_splits = extract_native_laps(data['laps'])
      df_best_efforts = calculate_best_efforts(df_records)

      total_dist_mi = (
          session_dist_m / 1609.34
          if session_dist_m
          else (
              df_records['distance'].max() / 1609.34
              if not df_records.empty and 'distance' in df_records.columns
              else 0.0
          )
      )

      if session_timer_sec and session_timer_sec > 0:
        active_time_sec = session_timer_sec
      elif not df_records.empty and 'dt' in df_records.columns:
        active_time_sec = float(df_records['dt'].clip(lower=0.0, upper=5.0).sum())
      else:
        active_time_sec = 0.0

      if total_dist_mi > 0 and active_time_sec > 0:
        sec_per_mi = active_time_sec / total_dist_mi
        p_min = int(sec_per_mi // 60)
        p_sec = int(round(sec_per_mi % 60))
        if p_sec == 60:
          p_min += 1
          p_sec = 0
        avg_pace_run = f'{p_min}:{p_sec:02d} /mi'
      else:
        avg_pace_run = 'N/A'

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

      duration_min = active_time_sec / 60.0
      workout_trimp = compute_banister_trimp(
          duration_min, avg_hr_run, active_hr_max, active_hr_rest
      )
      ef_score = calculate_efficiency_factor(df_records)
      decoupling_score = calculate_aerobic_decoupling(df_records)
      df_tiz = compute_time_in_zones(df_records, active_zones)

      st.markdown(
          f'**Selected Activity:** `{selected_fname}` &nbsp;|&nbsp;'
          f' **Location:** `{sel_city}, {sel_state}` &nbsp;|&nbsp;'
          f' **Classification:** `{selected_workout_type}`'
      )

      r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
      r1_c1.metric('Total Distance', f'{total_dist_mi:.2f} mi')
      r1_c2.metric(
          'Active Moving Time',
          f'{int(active_time_sec // 3600)}h {int((active_time_sec % 3600) // 60)}m'
          if active_time_sec > 3600
          else f'{int(active_time_sec // 60)}m {int(active_time_sec % 60)}s',
      )
      r1_c3.metric('Average Pace', avg_pace_run)
      r1_c4.metric(
          'Avg / Max HR',
          f'{avg_hr_run:.0f} / {max_hr_run:.0f} bpm'
          if (avg_hr_run and max_hr_run)
          else 'N/A',
      )

      r2_c1, r2_c2, r2_c3 = st.columns(3)
      r2_c1.metric('Relative Effort (TRIMP)', f'{workout_trimp}')
      r2_c2.metric('Efficiency Factor', f'{ef_score}' if ef_score else 'N/A')
      r2_c3.metric(
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

      render_single_activity_map(df_records, active_zones, file_key=selected_fname)

      st.markdown('---')

      tab1, tab2, tab3, tab4 = st.tabs([
          'Pace & HR Profile',
          'Native Splits',
          'Best Effort PRs',
          'Time in Zone',
      ])

      with tab1:
        st.subheader('High-Resolution Rolling Pace & Heart Rate')
        sw1, sw2 = st.columns(2)
        show_pace_line = sw1.toggle('Show 30s Rolling Pace', value=True)
        show_hr_line = sw2.toggle('Show Zone-Colored Heart Rate', value=True)

        if not show_pace_line and not show_hr_line:
          st.info('Toggle at least one telemetry stream above to display the graph.')
        else:
          plt.rcParams['font.family'] = 'serif'
          fig, ax1 = plt.subplots(figsize=(10, 4.5), facecolor='#120606')
          ax1.set_facecolor('#120606')

          dist_miles = df_records['distance'] / 1609.34
          ax1.set_xlabel('Distance (Miles)', color='#f5ebeb')
          ax1.tick_params(axis='x', colors='#f5ebeb')
          ax1.grid(True, linestyle='--', color='#3b0d0d', alpha=0.6)
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

          if show_pace_line and 'enhanced_speed' in df_records.columns:
            running_mask = df_records['enhanced_speed'] >= 2.0
            clean_speed = df_records['enhanced_speed'].where(
                running_mask, np.nan
            )
            sec_per_mile = 1609.34 / clean_speed
            rolling_pace = (
                sec_per_mile.rolling(window=30, min_periods=5).mean() / 60
            )
            ax1.plot(
                dist_miles,
                rolling_pace,
                color='#ffffff',
                linewidth=1.6,
                alpha=0.92,
                label='30s Rolling Pace',
            )
            ax1.set_ylabel('Pace (min/mi)', color='#ffffff')
            ax1.tick_params(axis='y', colors='#ffffff')
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(pace_formatter))
            ax1.invert_yaxis()

          if (
              show_hr_line
              and 'heart_rate' in df_records.columns
              and df_records['heart_rate'].notna().any()
          ):
            hr_ax = ax1.twinx() if show_pace_line else ax1
            rolling_hr = (
                df_records['heart_rate']
                .rolling(window=30, min_periods=5)
                .mean()
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
            hr_ax.add_collection(lc)

            valid_hr_vals = y_vals[~np.isnan(y_vals)]
            if len(valid_hr_vals) > 0:
              hr_ax.set_xlim(
                  float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
              )
              hr_ax.set_ylim(
                  valid_hr_vals.min() - 5, valid_hr_vals.max() + 5
              )

            hr_ax.set_ylabel('Heart Rate (bpm)', color='#f5ebeb')
            hr_ax.tick_params(axis='y', colors='#f5ebeb')
            for spine in hr_ax.spines.values():
              spine.set_color('#451010')

          legend_handles = []
          if show_pace_line:
            legend_handles.append(
                Line2D([0], [0], color='#ffffff', lw=1.6, label='Pace')
            )
          if show_hr_line:
            legend_handles.extend([
                Line2D([0], [0], color='#9ca3af', lw=2.2, label='Z0/Z1'),
                Line2D([0], [0], color='#38bdf8', lw=2.2, label='Z2'),
                Line2D([0], [0], color='#4ade80', lw=2.2, label='Z3'),
                Line2D([0], [0], color='#fb923c', lw=2.2, label='Z4'),
                Line2D([0], [0], color='#ef4444', lw=2.2, label='Z5'),
            ])
          if legend_handles:
            ax1.legend(
                handles=legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.13),
                ncol=len(legend_handles),
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
        ' sidebar or open Vault Maintenance and click Sync Local .fit'
        ' Folder to Vault.'
    )