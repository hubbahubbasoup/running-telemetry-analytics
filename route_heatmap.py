import glob
import os
import sqlite3
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

# High-visibility color ramp so single-pass roads are clearly legible while high-traffic routes glow bright
HEATMAP_COLOR_RANGE = [
    [185, 35, 45, 195],    # Low density (1 run): Visible warm crimson
    [225, 65, 35, 215],    # Light-medium density: Bright scarlet
    [245, 115, 25, 230],   # Medium density: Deep orange
    [255, 170, 35, 240],   # High density: Golden amber
    [255, 225, 105, 250],  # Very high density: Bright yellow
    [255, 255, 215, 255],  # Core hotspots: White-gold glow
]


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
        color_range=HEATMAP_COLOR_RANGE,
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
        'HeatmapLayer',
        data=df_gps,
        get_position='[lon, lat]',
        colorRange=HEATMAP_COLOR_RANGE,
        opacity=0.95,
        radiusPixels=radius_px,
        intensity=intensity,
        threshold=0.005,
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