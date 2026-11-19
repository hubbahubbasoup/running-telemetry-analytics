import glob
import os
import sqlite3
import numpy as np
import pandas as pd
import pydeck as pdk
from fitparse import FitFile
import streamlit as st

DB_NAME = 'training_vault.db'

TAG_COLORS = {
    'Recovery / Easy': [129, 199, 132, 190],
    'Base Run': [229, 57, 53, 200],
    'Long Run': [186, 104, 200, 200],
    'Tempo / Quality Long Run': [255, 143, 0, 220],
    'Intervals / Speedwork': [255, 23, 68, 235],
    'Tempo / Threshold': [255, 193, 7, 220],
    'Race': [255, 255, 255, 245],
}

HEATMAP_COLOR_RANGE = [
    [185, 35, 45, 195],    # Low density (1 run): Visible warm crimson
    [225, 65, 35, 215],    # Light-medium density: Bright scarlet
    [245, 115, 25, 230],   # Medium density: Deep orange
    [255, 170, 35, 240],   # High density: Golden amber
    [255, 225, 105, 250],  # Very high density: Bright yellow
    [255, 255, 215, 255],  # Core hotspots: White-gold glow
]

ZONE_RGB = {
    'Zone 1 (Recovery)': [156, 163, 175],   # Grey
    'Zone 2 (Endurance)': [56, 189, 248],   # Light Blue
    'Zone 3 (Tempo)': [74, 222, 128],       # Light Green
    'Zone 4 (Threshold)': [251, 146, 60],   # Orange
    'Zone 5 (VO2 Max)': [239, 68, 68],      # Red
}


def _interp_color_ramp(t, stops):
  """Linearly interpolates an RGB color across (position, [r, g, b]) stops for t in [0, 1]."""
  t = float(np.clip(t, 0.0, 1.0))
  for idx in range(len(stops) - 1):
    t0, c0 = stops[idx]
    t1, c1 = stops[idx + 1]
    if t <= t1:
      frac = (t - t0) / max(1e-6, (t1 - t0))
      return [
          int(round(c0[i] + frac * (c1[i] - c0[i])))
          for i in range(3)
      ]
  return [int(c) for c in stops[-1][1]]


def _format_pace_min_decimal(pace_dec):
  if pd.isna(pace_dec) or pace_dec <= 0 or pace_dec > 30:
    return 'N/A'
  mins = int(pace_dec)
  secs = int(round((pace_dec - mins) * 60))
  if secs == 60:
    mins += 1
    secs = 0
  return f'{mins}:{secs:02d}'


def _hr_to_zone_rgb(hr_val, zones):
  if pd.isna(hr_val):
    return ZONE_RGB['Zone 1 (Recovery)']
  if hr_val <= zones['Zone 1 (Recovery)'][1]:
    return ZONE_RGB['Zone 1 (Recovery)']
  if hr_val <= zones['Zone 2 (Endurance)'][1]:
    return ZONE_RGB['Zone 2 (Endurance)']
  if hr_val <= zones['Zone 3 (Tempo)'][1]:
    return ZONE_RGB['Zone 3 (Tempo)']
  if hr_val <= zones['Zone 4 (Threshold)'][1]:
    return ZONE_RGB['Zone 4 (Threshold)']
  return ZONE_RGB['Zone 5 (VO2 Max)']


def render_single_activity_map(df_records, active_zones, file_key='run'):
  """Renders a Garmin-style route map for a single activity with Pace & HR color overlays."""
  if (
      df_records.empty
      or 'position_lat' not in df_records.columns
      or 'position_long' not in df_records.columns
  ):
    st.info('No GPS route recorded for this activity (indoor / treadmill run).')
    return

  semicircle_to_deg = 180.0 / (2**31)
  df = df_records.copy()

  if 'enhanced_speed' in df.columns:
    clean_spd = df['enhanced_speed'].where(df['enhanced_speed'] >= 1.5, np.nan)
    raw_pace = (1609.34 / clean_spd) / 60.0
    df['smooth_pace'] = raw_pace.rolling(window=15, min_periods=1).mean()
  else:
    df['smooth_pace'] = np.nan

  if 'heart_rate' in df.columns and df['heart_rate'].notna().any():
    df['smooth_hr'] = (
        df['heart_rate'].rolling(window=15, min_periods=1).mean()
    )
    has_hr = True
  else:
    df['smooth_hr'] = np.nan
    has_hr = False

  if 'distance' in df.columns:
    df['dist_mi'] = df['distance'] / 1609.34
  else:
    df['dist_mi'] = 0.0

  df = df.dropna(subset=['position_lat', 'position_long']).copy()
  if len(df) < 2:
    st.info('Not enough GPS coordinates to render a route map for this activity.')
    return

  df['lat'] = df['position_lat'] * semicircle_to_deg
  df['lon'] = df['position_long'] * semicircle_to_deg
  df = df[
      df['lat'].between(-89.9, 89.9) & df['lon'].between(-179.9, 179.9)
  ].reset_index(drop=True)

  if len(df) < 2:
    st.info('No valid GPS coordinates found for this activity.')
    return

  st.subheader('Activity GPS Route & Telemetry Overlay')

  has_pace = df['smooth_pace'].notna().any()
  overlay_options = []
  if has_pace:
    overlay_options.append('Pace Overlay')
  if has_hr:
    overlay_options.append('Heart Rate (Karvonen Zones)')
    overlay_options.append('Heart Rate (Custom Range)')
  if not overlay_options:
    overlay_options.append('Route Only')

  overlay_mode = st.radio(
      'Route Color Overlay',
      overlay_options,
      horizontal=True,
      key=f'route_overlay_{file_key}',
  )

  pace_slider_val = None
  p_min_bound, p_max_bound = 4.0, 14.0
  def_p_lo, def_p_hi = 5.0, 10.0
  if has_pace:
    valid_p = df['smooth_pace'].dropna()
    valid_p = valid_p[(valid_p >= 3.5) & (valid_p <= 18.0)]
    if not valid_p.empty:
      p_min_bound = max(3.5, round(float(valid_p.quantile(0.01)) - 0.5, 1))
      p_max_bound = min(18.0, round(float(valid_p.quantile(0.99)) + 0.5, 1))
      if p_max_bound <= p_min_bound + 0.4:
        p_max_bound = round(p_min_bound + 1.0, 1)
      def_p_lo = round(float(valid_p.quantile(0.05)), 1)
      def_p_hi = round(float(valid_p.quantile(0.95)), 1)
      def_p_lo = max(p_min_bound, min(def_p_lo, round(p_max_bound - 0.2, 1)))
      def_p_hi = max(round(def_p_lo + 0.2, 1), min(def_p_hi, p_max_bound))

  hr_slider_val = None
  hr_min_bound, hr_max_bound = 90, 210
  def_hr_lo, def_hr_hi = 110, 195
  if has_hr:
    valid_hr = df['smooth_hr'].dropna()
    valid_hr = valid_hr[(valid_hr >= 50) & (valid_hr <= 220)]
    if not valid_hr.empty:
      hr_min_bound = max(50, int(np.floor(float(valid_hr.min()) - 5)))
      hr_max_bound = min(220, int(np.ceil(float(valid_hr.max()) + 5)))
      if hr_max_bound <= hr_min_bound + 4:
        hr_max_bound = hr_min_bound + 10
      def_hr_lo = max(hr_min_bound, int(round(float(valid_hr.quantile(0.02)))))
      def_hr_hi = min(hr_max_bound, int(round(float(valid_hr.quantile(0.98)))))
      if def_hr_hi <= def_hr_lo + 2:
        def_hr_lo, def_hr_hi = hr_min_bound, hr_max_bound

  s_col1, s_col2 = st.columns(2)
  with s_col1:
    if has_pace:
      pace_slider_val = st.slider(
          'Pace Overlay / Filter Scale (min/mi)',
          min_value=float(p_min_bound),
          max_value=float(p_max_bound),
          value=(float(def_p_lo), float(def_p_hi)),
          step=0.1,
          key=f'map_pace_slider_{file_key}',
      )
      st.caption(
          f'Selected Pace Window: **{_format_pace_min_decimal(pace_slider_val[0])} /mi**'
          f' (Fast) – **{_format_pace_min_decimal(pace_slider_val[1])} /mi** (Slow)'
      )
  with s_col2:
    if has_hr:
      hr_slider_val = st.slider(
          'Heart Rate Overlay / Filter Scale (bpm)',
          min_value=int(hr_min_bound),
          max_value=int(hr_max_bound),
          value=(int(def_hr_lo), int(def_hr_hi)),
          step=1,
          key=f'map_hr_slider_{file_key}',
      )
      st.caption(
          f'Selected HR Window: **{hr_slider_val[0]} bpm** (Low) –'
          f' **{hr_slider_val[1]} bpm** (High)'
      )

  step = max(1, len(df) // 1200)
  df_plot = df.iloc[::step].copy()
  if df_plot.index[-1] != df.index[-1]:
    df_plot = pd.concat([df_plot, df.iloc[[-1]]], ignore_index=True)
  else:
    df_plot = df_plot.reset_index(drop=True)

  pace_stops = [
      (0.00, [56, 189, 248]),   # Fastest: Light Blue
      (0.35, [74, 222, 128]),   # Fast-Moderate: Green
      (0.70, [251, 146, 60]),   # Moderate-Slow: Orange
      (1.00, [239, 68, 68]),    # Slowest: Red
  ]
  hr_stops = [
      (0.00, [56, 189, 248]),   # Low HR: Light Blue
      (0.35, [74, 222, 128]),   # Aerobic: Green
      (0.70, [251, 146, 60]),   # Threshold: Orange
      (1.00, [239, 68, 68]),    # Peak HR: Red
  ]

  pace_filter_active = (
      pace_slider_val is not None
      and (
          pace_slider_val[0] > p_min_bound + 0.05
          or pace_slider_val[1] < p_max_bound - 0.05
      )
  )
  hr_filter_active = (
      hr_slider_val is not None
      and (
          hr_slider_val[0] > hr_min_bound
          or hr_slider_val[1] < hr_max_bound
      )
  )

  lats = df_plot['lat'].to_numpy()
  lons = df_plot['lon'].to_numpy()
  paces = df_plot['smooth_pace'].to_numpy()
  hrs = df_plot['smooth_hr'].to_numpy()
  dists = df_plot['dist_mi'].to_numpy()

  segments = []
  for i in range(len(df_plot) - 1):
    mid_p = (
        0.5 * (paces[i] + paces[i + 1])
        if (not np.isnan(paces[i]) and not np.isnan(paces[i + 1]))
        else (paces[i] if not np.isnan(paces[i]) else paces[i + 1])
    )
    mid_hr = (
        0.5 * (hrs[i] + hrs[i + 1])
        if (not np.isnan(hrs[i]) and not np.isnan(hrs[i + 1]))
        else (hrs[i] if not np.isnan(hrs[i]) else hrs[i + 1])
    )
    mid_dist = 0.5 * (dists[i] + dists[i + 1]) if not np.isnan(dists[i]) else 0.0

    alpha = 235
    if overlay_mode == 'Pace Overlay' and pace_slider_val is not None:
      if np.isnan(mid_p):
        rgb = [156, 163, 175]
      else:
        t = (mid_p - pace_slider_val[0]) / max(
            0.05, pace_slider_val[1] - pace_slider_val[0]
        )
        rgb = _interp_color_ramp(t, pace_stops)
      if hr_filter_active and not np.isnan(mid_hr):
        if mid_hr < hr_slider_val[0] or mid_hr > hr_slider_val[1]:
          rgb = [90, 90, 95]
          alpha = 65
    elif overlay_mode == 'Heart Rate (Karvonen Zones)':
      rgb = _hr_to_zone_rgb(mid_hr, active_zones)
      if hr_filter_active and not np.isnan(mid_hr):
        if mid_hr < hr_slider_val[0] or mid_hr > hr_slider_val[1]:
          rgb = [90, 90, 95]
          alpha = 65
      if pace_filter_active and not np.isnan(mid_p):
        if mid_p < pace_slider_val[0] or mid_p > pace_slider_val[1]:
          rgb = [90, 90, 95]
          alpha = 65
    elif overlay_mode == 'Heart Rate (Custom Range)' and hr_slider_val is not None:
      if np.isnan(mid_hr):
        rgb = [156, 163, 175]
      else:
        t = (mid_hr - hr_slider_val[0]) / max(
            1.0, hr_slider_val[1] - hr_slider_val[0]
        )
        rgb = _interp_color_ramp(t, hr_stops)
      if pace_filter_active and not np.isnan(mid_p):
        if mid_p < pace_slider_val[0] or mid_p > pace_slider_val[1]:
          rgb = [90, 90, 95]
          alpha = 65
    else:
      rgb = [220, 38, 38]

    segments.append({
        'path': [[float(lons[i]), float(lats[i])], [float(lons[i + 1]), float(lats[i + 1])]],
        'color': [int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha)],
        'dist_str': f'{mid_dist:.2f} mi',
        'pace_str': f'{_format_pace_min_decimal(mid_p)} /mi',
        'hr_str': f'{int(round(mid_hr))} bpm' if not np.isnan(mid_hr) else 'N/A',
    })

  if overlay_mode == 'Pace Overlay':
    st.markdown(
        '<div style="font-size:0.95rem; margin-bottom:6px;">'
        '<b>Pace Scale:</b> &nbsp;'
        '<span style="color:#38bdf8; font-weight:600;">Fast</span> &rarr; '
        '<span style="color:#4ade80; font-weight:600;">Brisk</span> &rarr; '
        '<span style="color:#fb923c; font-weight:600;">Moderate</span> &rarr; '
        '<span style="color:#ef4444; font-weight:600;">Slow</span> &nbsp;|&nbsp; '
        '<span style="color:#4ade80;">&#9679; Start</span> &nbsp; '
        '<span style="color:#ef4444;">&#9679; Finish</span>'
        '</div>',
        unsafe_allow_html=True,
    )
  elif overlay_mode == 'Heart Rate (Karvonen Zones)':
    st.markdown(
        '<div style="font-size:0.95rem; margin-bottom:6px;">'
        '<b>Karvonen Zones:</b> &nbsp;'
        '<span style="color:#9ca3af; font-weight:600;">[Z1]</span> &nbsp;'
        '<span style="color:#38bdf8; font-weight:600;">[Z2]</span> &nbsp;'
        '<span style="color:#4ade80; font-weight:600;">[Z3]</span> &nbsp;'
        '<span style="color:#fb923c; font-weight:600;">[Z4]</span> &nbsp;'
        '<span style="color:#ef4444; font-weight:600;">[Z5]</span> &nbsp;|&nbsp; '
        '<span style="color:#4ade80;">&#9679; Start</span> &nbsp; '
        '<span style="color:#ef4444;">&#9679; Finish</span>'
        '</div>',
        unsafe_allow_html=True,
    )
  elif overlay_mode == 'Heart Rate (Custom Range)':
    st.markdown(
        '<div style="font-size:0.95rem; margin-bottom:6px;">'
        '<b>Heart Rate Scale:</b> &nbsp;'
        '<span style="color:#38bdf8; font-weight:600;">Low</span> &rarr; '
        '<span style="color:#4ade80; font-weight:600;">Aerobic</span> &rarr; '
        '<span style="color:#fb923c; font-weight:600;">Threshold</span> &rarr; '
        '<span style="color:#ef4444; font-weight:600;">Peak</span> &nbsp;|&nbsp; '
        '<span style="color:#4ade80;">&#9679; Start</span> &nbsp; '
        '<span style="color:#ef4444;">&#9679; Finish</span>'
        '</div>',
        unsafe_allow_html=True,
    )

  lat_min, lat_max = float(lats.min()), float(lats.max())
  lon_min, lon_max = float(lons.min()), float(lons.max())
  center_lat = 0.5 * (lat_min + lat_max)
  center_lon = 0.5 * (lon_min + lon_max)
  lat_span = max(0.001, lat_max - lat_min)
  lon_span = max(0.001, (lon_max - lon_min) * np.cos(np.radians(center_lat)))
  max_span = max(lat_span, lon_span)
  auto_zoom = float(np.clip(np.log2(360.0 / (max_span * 2.5)), 9.5, 16.0))

  df_markers = pd.DataFrame([
      {'lon': float(lons[0]), 'lat': float(lats[0]), 'color': [74, 222, 128, 255]},
      {'lon': float(lons[-1]), 'lat': float(lats[-1]), 'color': [239, 68, 68, 255]},
  ])

  path_layer = pdk.Layer(
      'PathLayer',
      data=pd.DataFrame(segments),
      get_path='path',
      get_color='color',
      width_scale=1,
      width_min_pixels=5,
      rounded=True,
      pickable=True,
      auto_highlight=True,
  )

  marker_layer = pdk.Layer(
      'ScatterplotLayer',
      data=df_markers,
      get_position='[lon, lat]',
      get_fill_color='color',
      get_line_color=[255, 255, 255, 255],
      line_width_min_pixels=2,
      stroked=True,
      radius_min_pixels=6,
      radius_max_pixels=9,
      pickable=False,
  )

  view_state = pdk.ViewState(
      latitude=center_lat,
      longitude=center_lon,
      zoom=auto_zoom,
      pitch=0,
  )

  tooltip = {
      'html': (
          '<b>Mile:</b> {dist_str}<br/>'
          '<b>Pace:</b> {pace_str}<br/>'
          '<b>Heart Rate:</b> {hr_str}'
      ),
      'style': {
          'backgroundColor': '#160707',
          'color': '#f5ebeb',
          'border': '1px solid #991b1b',
      },
  }

  st.pydeck_chart(
      pdk.Deck(
          layers=[path_layer, marker_layer],
          initial_view_state=view_state,
          map_style='dark',
          tooltip=tooltip,
      ),
      use_container_width=True,
  )


def init_gps_table():
  """Creates the gps_points table inside training_vault.db if missing."""
  conn = sqlite3.connect(DB_NAME)
  conn.execute('''
      CREATE TABLE IF NOT EXISTS gps_points (
          file_name TEXT,
          lat REAL,
          lon REAL
      )
  ''')
  conn.execute(
      'CREATE INDEX IF NOT EXISTS idx_gps_file ON gps_points(file_name)'
  )
  conn.commit()
  conn.close()


def sync_gps_to_sqlite(folder_path='fit test files', downsample_step=3):
  """Parses only uncached .fit files from folder_path and stores coordinates in SQLite."""
  init_gps_table()
  semicircle_to_deg = 180.0 / (2**31)

  fit_files = glob.glob(os.path.join(folder_path, '*.fit'))
  if not fit_files:
    fit_files = glob.glob('**/*.fit', recursive=True)

  conn = sqlite3.connect(DB_NAME)
  cached_files = {
      row[0]
      for row in conn.execute(
          'SELECT DISTINCT file_name FROM gps_points'
      ).fetchall()
  }

  new_count = 0
  for file_path in fit_files:
    fname = os.path.basename(file_path)
    if fname in cached_files:
      continue

    try:
      fitfile = FitFile(file_path)
      pts = []
      for record in fitfile.get_messages('record'):
        vals = {f.name: f.value for f in record}
        lat_raw = vals.get('position_lat')
        lon_raw = vals.get('position_long')
        if lat_raw is not None and lon_raw is not None:
          pts.append((lat_raw * semicircle_to_deg, lon_raw * semicircle_to_deg))

      if pts:
        df_pts = pd.DataFrame(pts, columns=['lat', 'lon'])
        df_pts = df_pts[
            df_pts['lat'].between(-89.9, 89.9)
            & df_pts['lon'].between(-179.9, 179.9)
        ].iloc[::downsample_step]

        if not df_pts.empty:
          df_pts.insert(0, 'file_name', fname)
          df_pts.to_sql('gps_points', conn, if_exists='append', index=False)
          new_count += 1
    except Exception:
      continue

  conn.close()
  return new_count


def load_cached_gps_with_tags():
  """Reads stored GPS coordinates joined with workout metadata from SQLite."""
  init_gps_table()
  conn = sqlite3.connect(DB_NAME)

  tables = {
      row[0]
      for row in conn.execute(
          "SELECT name FROM sqlite_master WHERE type='table'"
      ).fetchall()
  }

  if 'activities' in tables:
    query = '''
        SELECT
            g.rowid AS seq,
            g.file_name,
            g.lat,
            g.lon,
            COALESCE(a.workout_tag, 'Base Run') AS workout_tag,
            COALESCE(a.date, '') AS date,
            COALESCE(a.distance_miles, 0.0) AS distance_miles
        FROM gps_points g
        LEFT JOIN activities a ON g.file_name = a.file_name
        ORDER BY g.file_name, g.rowid
    '''
  else:
    query = '''
        SELECT
            rowid AS seq,
            file_name,
            lat,
            lon,
            'Base Run' AS workout_tag,
            '' AS date,
            0.0 AS distance_miles
        FROM gps_points
        ORDER BY file_name, rowid
    '''

  df_gps = pd.read_sql(query, conn)
  conn.close()
  return df_gps


def build_polyline_paths(df_gps):
  """Groups sequential GPS points per activity into path lists for pydeck PathLayer."""
  paths = []
  for fname, group in df_gps.groupby('file_name', sort=False):
    if len(group) < 2:
      continue
    tag = group['workout_tag'].iloc[0]
    date_str = str(group['date'].iloc[0])[:10]
    dist_mi = group['distance_miles'].iloc[0]
    coords = group[['lon', 'lat']].values.tolist()

    paths.append({
        'file_name': fname,
        'workout_tag': tag,
        'date': date_str if date_str else 'Unknown Date',
        'distance_miles': f'{dist_mi:.2f} mi',
        'path': coords,
        'color': TAG_COLORS.get(tag, [229, 57, 53, 200]),
    })
  return pd.DataFrame(paths)


def render_heatmap_tab():
  """Renders the SQLite-cached Route Heatmap & Colored Polylines tab in Streamlit."""
  st.subheader('Personal GPS Route Map')

  folder_input = st.sidebar.text_input(
      'FIT Archive Folder', value='fit test files'
  )

  map_mode = st.sidebar.radio(
      'Map Visualization Mode',
      ['Heat Density Glow', 'Individual Workout Traces'],
  )

  if st.sidebar.button('Scan Folder for New Routes'):
    with st.spinner('Scanning for new .fit GPS tracks...'):
      added = sync_gps_to_sqlite(folder_input)
      if added > 0:
        st.sidebar.success(f'Added {added} new route(s) to vault.')
      else:
        st.sidebar.info('All routes already cached in SQLite.')
  else:
    sync_gps_to_sqlite(folder_input)

  df_gps = load_cached_gps_with_tags()

  if df_gps.empty:
    st.warning(
        'No GPS coordinates cached yet. Upload .fit files in the sidebar or'
        ' click Scan Folder for New Routes.'
    )
    return

  available_tags = ['All'] + sorted(
      df_gps['workout_tag'].dropna().unique().tolist()
  )
  selected_tag = st.sidebar.selectbox(
      'Filter Map by Workout Tag', available_tags
  )

  if selected_tag != 'All':
    df_gps = df_gps[df_gps['workout_tag'] == selected_tag]

  if df_gps.empty:
    st.info(f'No mapped GPS routes found for {selected_tag}.')
    return

  num_runs = df_gps['file_name'].nunique()
  c1, c2 = st.columns(2)
  c1.metric('Mapped Activities', num_runs)
  c2.metric('Cached GPS Points', f'{len(df_gps):,}')

  view_state = pdk.ViewState(
      latitude=float(df_gps['lat'].median()),
      longitude=float(df_gps['lon'].median()),
      zoom=12,
      pitch=0,
  )

  if map_mode == 'Heat Density Glow':
    radius_px = st.sidebar.slider(
        'Heat Radius (px)', min_value=5, max_value=40, value=16
    )
    intensity = st.sidebar.slider(
        'Heat Intensity', min_value=0.5, max_value=6.0, value=2.4, step=0.2
    )

    layer = pdk.Layer(
        'HeatmapLayer',
        data=df_gps,
        get_position='[lon, lat]',
        colorRange=HEATMAP_COLOR_RANGE,
        opacity=0.95,
        radiusPixels=radius_px,
        intensity=intensity,
        threshold=0.005,
    )
    tooltip = None
  else:
    line_width = st.sidebar.slider(
        'Route Line Width (px)', min_value=2, max_value=10, value=4
    )
    df_paths = build_polyline_paths(df_gps)

    st.markdown(
        '<div style="font-size:0.95rem; margin-bottom:8px;">'
        '<span style="color:#81c784; font-weight:600;">[Recovery/Easy]</span> &nbsp;'
        '<span style="color:#e53935; font-weight:600;">[Base Run]</span> &nbsp;'
        '<span style="color:#ba68c8; font-weight:600;">[Long Run]</span> &nbsp;'
        '<span style="color:#ff8f00; font-weight:600;">[Quality Long Run]</span> &nbsp;'
        '<span style="color:#ff1744; font-weight:600;">[Intervals]</span> &nbsp;'
        '<span style="color:#ffc107; font-weight:600;">[Tempo]</span> &nbsp;'
        '<span style="color:#ffffff; font-weight:600;">[Race]</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    layer = pdk.Layer(
        'PathLayer',
        data=df_paths,
        get_path='path',
        get_color='color',
        width_scale=1,
        width_min_pixels=line_width,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        'html': (
            '<b>{workout_tag}</b><br/>Date: {date}<br/>Distance:'
            ' {distance_miles}<br/>File: {file_name}'
        ),
        'style': {
            'backgroundColor': '#160707',
            'color': '#f5e6e6',
            'border': '1px solid #8b0000',
        },
    }

  st.pydeck_chart(
      pdk.Deck(
          layers=[layer],
          initial_view_state=view_state,
          map_style='dark',
          tooltip=tooltip,
      ),
      use_container_width=True,
  )