from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
import sys
import tomllib
import requests
import streamlit as st

DB_NAME = 'training_vault.db'
BUCKET_NAME = 'vault'


def _get_secret(key):
  """Safely reads a secret from environment variables, .streamlit/secrets.toml, or st.secrets."""
  if key in os.environ:
    return os.environ[key]
  secrets_path = os.path.join('.streamlit', 'secrets.toml')
  if os.path.exists(secrets_path):
    try:
      with open(secrets_path, 'rb') as f:
        data = tomllib.load(f)
      if key in data:
        return data[key]
    except Exception:
      pass
  try:
    if key in st.secrets:
      return st.secrets[key]
  except Exception:
    pass
  return None


def is_cloud_configured():
  return bool(_get_secret('SUPABASE_URL') and _get_secret('SUPABASE_KEY'))


def pull_vault_from_cloud(db_path=DB_NAME):
  """Downloads training_vault.db from Supabase Storage."""
  url = _get_secret('SUPABASE_URL')
  key = _get_secret('SUPABASE_KEY')
  if not url or not key:
    return False

  endpoint = f"{url.rstrip('/')}/storage/v1/object/{BUCKET_NAME}/{db_path}"
  headers = {'Authorization': f'Bearer {key}', 'apikey': key}

  try:
    resp = requests.get(endpoint, headers=headers, timeout=30)
    if resp.status_code == 200 and len(resp.content) > 0:
      with open(db_path, 'wb') as f:
        f.write(resp.content)
      return True
  except Exception as e:
    print(f'Cloud pull warning: {e}')
  return False


def push_vault_to_cloud(db_path=DB_NAME):
  """Uploads the local training_vault.db file to Supabase Storage (upsert)."""
  url = _get_secret('SUPABASE_URL')
  key = _get_secret('SUPABASE_KEY')
  if not url or not key or not os.path.exists(db_path):
    return False

  endpoint = f"{url.rstrip('/')}/storage/v1/object/{BUCKET_NAME}/{db_path}"
  headers = {
      'Authorization': f'Bearer {key}',
      'apikey': key,
      'Content-Type': 'application/octet-stream',
      'x-upsert': 'true',
  }

  try:
    with open(db_path, 'rb') as f:
      resp = requests.post(endpoint, headers=headers, data=f, timeout=60)
    if resp.status_code not in (200, 201):
      print(f'Cloud push error ({resp.status_code}): {resp.text}')
    return resp.status_code in (200, 201)
  except Exception as e:
    print(f'Cloud push warning: {e}')
    return False


def push_fit_gz_to_cloud(file_name, gz_bytes):
  """Uploads a single gzipped .fit binary to vault/fit_files/<file_name>.gz."""
  url = _get_secret('SUPABASE_URL')
  key = _get_secret('SUPABASE_KEY')
  if not url or not key:
    return False

  endpoint = (
      f"{url.rstrip('/')}/storage/v1/object/{BUCKET_NAME}/fit_files/{file_name}.gz"
  )
  headers = {
      'Authorization': f'Bearer {key}',
      'apikey': key,
      'Content-Type': 'application/octet-stream',
      'x-upsert': 'true',
  }
  try:
    resp = requests.post(endpoint, headers=headers, data=gz_bytes, timeout=30)
    return resp.status_code in (200, 201)
  except Exception:
    return False


def pull_fit_gz_from_cloud(file_name):
  """Downloads a single gzipped .fit binary from vault/fit_files/<file_name>.gz."""
  url = _get_secret('SUPABASE_URL')
  key = _get_secret('SUPABASE_KEY')
  if not url or not key:
    return None

  endpoint = (
      f"{url.rstrip('/')}/storage/v1/object/{BUCKET_NAME}/fit_files/{file_name}.gz"
  )
  headers = {'Authorization': f'Bearer {key}', 'apikey': key}
  try:
    resp = requests.get(endpoint, headers=headers, timeout=30)
    if resp.status_code == 200 and len(resp.content) > 0:
      return resp.content
  except Exception:
    pass
  return None


def migrate_local_blobs_to_cloud(db_path=DB_NAME):
  """Uploads stored .fit.gz blobs from SQLite to Supabase and compacts training_vault.db."""
  if not os.path.exists(db_path):
    print(f'{db_path} not found.')
    return False

  conn = sqlite3.connect(db_path)
  rows = conn.execute(
      'SELECT file_name, raw_gz FROM fit_files WHERE raw_gz IS NOT NULL'
  ).fetchall()

  if rows:
    print(f'Uploading {len(rows)} compressed .fit streams to Supabase...')

    def _upload(row):
      fname, blob = row
      return push_fit_gz_to_cloud(fname, bytes(blob))

    with ThreadPoolExecutor(max_workers=8) as ex:
      results = list(ex.map(_upload, rows))

    ok_count = sum(1 for ok in results if ok)
    print(f'Uploaded {ok_count}/{len(rows)} streams.')
    if ok_count < len(rows):
      print('Some streams failed to upload; keeping local blobs intact.')
      conn.close()
      return False

    conn.execute('UPDATE fit_files SET raw_gz = NULL')
    conn.commit()

  print('Compacting SQLite database with VACUUM...')
  conn.execute('VACUUM')
  conn.close()

  new_mb = os.path.getsize(db_path) / (1024 * 1024)
  print(f'Shrunk {db_path} to {new_mb:.2f} MB.')
  return push_vault_to_cloud(db_path)


def ensure_local_vault(db_path=DB_NAME):
  """Pulls the cloud database once per Streamlit session on startup."""
  if not is_cloud_configured():
    return
  if 'vault_synced_from_cloud' not in st.session_state:
    if not os.path.exists(db_path) or os.path.getsize(db_path) < 16384:
      with st.spinner('Restoring training vault from cloud storage...'):
        pull_vault_from_cloud(db_path)
    st.session_state['vault_synced_from_cloud'] = True


def check_password_gate():
  """Prompts for APP_PASSWORD if configured in secrets; bypassed locally if unset."""
  app_password = _get_secret('APP_PASSWORD')
  if not app_password:
    return

  if st.session_state.get('authenticated'):
    return

  st.subheader('Vault Authentication')
  entered = st.text_input('Enter access password', type='password')
  if st.button('Unlock Dashboard'):
    if entered == app_password:
      st.session_state['authenticated'] = True
      st.rerun()
    else:
      st.error('Invalid password.')
  st.stop()


if __name__ == '__main__':
  cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ''
  if cmd == 'push':
    ok = push_vault_to_cloud(DB_NAME)
    print(
        'Successfully pushed training_vault.db to cloud.'
        if ok
        else 'Push failed. Check secrets.'
    )
  elif cmd == 'pull':
    ok = pull_vault_from_cloud(DB_NAME)
    print(
        'Successfully pulled training_vault.db from cloud.'
        if ok
        else 'Pull failed. Check secrets.'
    )
  elif cmd == 'migrate':
    ok = migrate_local_blobs_to_cloud(DB_NAME)
    print(
        'Migration complete! training_vault.db is live in Supabase.'
        if ok
        else 'Migration failed.'
    )
  else:
    print('Usage: python cloud_vault.py [push|pull|migrate]')