# football_project/utils.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from backend.features.config import DB_CONFIG # Sửa đường dẫn import

def get_pg_engine():
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(db_url)

def preprocess_market_value(value_str):
    if pd.isna(value_str) or str(value_str).lower() == 'unknown': return None
    value_str = str(value_str).replace('€', '').strip()
    multiplier = 1
    if 'm' in value_str.lower(): multiplier = 1_000_000; value_str = value_str.lower().replace('m', '')
    elif 'k' in value_str.lower(): multiplier = 1_000; value_str = value_str.lower().replace('k', '')
    try: return float(value_str) * multiplier
    except ValueError: return None

def preprocess_percentage(value_str):
    if pd.isna(value_str) or not isinstance(value_str, str): return None
    try: return float(str(value_str).replace('%', '')) / 100.0
    except ValueError: return None

def preprocess_boolean_text(value_str):
    if isinstance(value_str, bool): return value_str
    if pd.isna(value_str): return None
    s_val = str(value_str).strip().lower()
    if s_val == 'true': return True
    if s_val == 'false' or s_val == '': return False
    return None # For DB BOOLEAN, this will be NULL

# Points calculation utility
def get_points(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score): return pd.NA, pd.NA # Use pandas NA for Int64 compatibility
    if home_score > away_score: return 3, 0
    elif home_score < away_score: return 0, 3
    else: return 1, 1