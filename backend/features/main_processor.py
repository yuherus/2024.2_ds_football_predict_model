# football_project/main_processor.py
import os
from sqlalchemy import text as sqlalchemy_text  # For clearing tables

# Import functions from other modules
from config import ALL_PLAYERS_CSV_PATH, MATCH_RESULTS_CSV_PATH
from utils import get_pg_engine
from db_setup import create_db_tables
from data_loader import load_squads_data, load_raw_matches_data
from feature_engineering import process_all_seasons_and_store_features

def clear_tables_for_rerun(engine, tables_to_clear=None):
    """Optionally clear tables before a full data processing run."""
    if tables_to_clear is None:
        tables_to_clear = ['squads', 'matches_raw', 'matches_featured']
    if not tables_to_clear:
        return
    with engine.connect() as conn:
        for table_name in tables_to_clear:
            try:
                conn.execute(sqlalchemy_text(f"DELETE FROM {table_name};"))
                print(f"Table '{table_name}' cleared.")
            except Exception as e:
                print(f"Error clearing table '{table_name}': {e} (Table might not exist yet or other issue)")
        conn.commit()


def main():
    print("--- Starting Football Data Processing Pipeline ---")

    # 1. Setup Database Tables
    print("\nStep 1: Setting up database tables...")
    create_db_tables()

    # OPTIONAL: Clear tables if you want a fresh run for all data
    # Be careful with this on a production database!
    engine = get_pg_engine()
    # clear_tables_for_rerun(engine, ['squads', 'matches_raw', 'matches_featured'])

    # 2. Load Squads Data
    print("\nStep 2: Loading squads data...")
    squad_rows = load_squads_data(ALL_PLAYERS_CSV_PATH)
    print(f"Finished loading squads. {squad_rows} rows processed into DB.")

    # 3. Load Raw Matches Data
    print("\nStep 3: Loading raw matches data...")
    raw_match_rows = load_raw_matches_data(MATCH_RESULTS_CSV_PATH)
    print(f"Finished loading raw matches. {raw_match_rows} rows processed into DB.")

    # 4. Feature Engineering
    # (This reads from 'matches_raw' and writes to 'matches_featured')
    print("\nStep 4: Performing feature engineering...")
    process_all_seasons_and_store_features()
    print("Finished feature engineering.")

    print("\n--- Football Data Processing Pipeline Finished ---")


if __name__ == '__main__':
    main()