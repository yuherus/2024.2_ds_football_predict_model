# football_project/data_loader.py
import pandas as pd
import numpy as np
import os
from utils import get_pg_engine, preprocess_market_value, preprocess_percentage, preprocess_boolean_text
from config import CHUNK_SIZE, RAW_MATCH_COLS_FROM_CSV, RAW_MATCH_NUMERIC_COLS_DERIVED


def load_squads_data(csv_path):
    engine = get_pg_engine()
    if not os.path.exists(csv_path):
        print(f"Squads CSV file not found: {csv_path}")
        return 0

    print(f"Loading squads data from {csv_path}...")
    total_rows_inserted = 0

    # Define expected columns for the squads table to ensure DataFrame has them
    db_squad_cols = ['team', 'league', 'season', 'jersey_number', 'player_name', 'position',
                     'dob_age', 'age', 'nationality', 'market_value_text', 'market_value_eur',
                     'on_loan', 'loan_from', 'is_captain']

    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False, na_filter=False)):
        print(f"  Processing squad chunk {chunk_num + 1}...")
        chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_')

        # Rename CSV columns to match DB columns if needed
        # e.g. if csv has 'player id' and db has 'player_id'
        # chunk.rename(columns={'old_name': 'new_name'}, inplace=True)

        chunk.rename(columns={'market_value': 'market_value_text'}, inplace=True)  # common case
        chunk['market_value_eur'] = chunk['market_value_text'].apply(preprocess_market_value)

        # Handle boolean text based on potential column names from CSV
        if 'on_loan_text' in chunk.columns:  # if original CSV column name was 'on_loan_text'
            chunk['on_loan'] = chunk['on_loan_text'].apply(preprocess_boolean_text)
        elif 'on_loan' in chunk.columns:  # if it was just 'on_loan' and text like 'True'
            chunk['on_loan'] = chunk['on_loan'].apply(preprocess_boolean_text)
        else:
            chunk['on_loan'] = None  # Ensure column exists

        if 'is_captain_text' in chunk.columns:
            chunk['is_captain'] = chunk['is_captain_text'].apply(preprocess_boolean_text)
        elif 'is_captain' in chunk.columns:
            chunk['is_captain'] = chunk['is_captain'].apply(preprocess_boolean_text)
        else:
            chunk['is_captain'] = None

        # Ensure all db_squad_cols exist
        for col in db_squad_cols:
            if col not in chunk.columns:
                chunk[col] = None  # Or np.nan, or pd.NA based on target type

        chunk_to_db = chunk[db_squad_cols]
        chunk_to_db.drop_duplicates(subset=['team', 'season', 'player_name'], keep='first', inplace=True)

        if chunk_to_db.empty:
            print(f"    Squad chunk {chunk_num + 1} empty after deduplication.")
            continue

        try:
            chunk_to_db.to_sql('squads', engine, if_exists='append', index=False, method='multi')
            total_rows_inserted += len(chunk_to_db)
            print(f"    Inserted {len(chunk_to_db)} squad rows. Total: {total_rows_inserted}")
        except Exception as e:
            print(f"    Error inserting squad chunk {chunk_num + 1}: {e}")
            # print(chunk_to_db.head().to_dict())
            # chunk_to_db.to_csv(f"failing_squad_chunk_{chunk_num+1}.csv", index=False)

    print(f"Finished loading squads data. Total rows inserted: {total_rows_inserted}")
    return total_rows_inserted


def load_raw_matches_data(csv_path):
    engine = get_pg_engine()
    if not os.path.exists(csv_path):
        print(f"Match results CSV file not found: {csv_path}")
        return 0

    print(f"Loading raw matches data from {csv_path}...")
    total_rows_inserted = 0

    # Columns in matches_raw, excluding 'id' (auto-increment)
    # Combine RAW_MATCH_COLS_FROM_CSV and RAW_MATCH_NUMERIC_COLS_DERIVED for the full set
    expected_db_cols_for_raw_matches = RAW_MATCH_COLS_FROM_CSV + RAW_MATCH_NUMERIC_COLS_DERIVED

    # Numeric columns requiring specific casting (Int64 for nullable ints)
    numeric_cols_int = ['home_score', 'away_score', 'round', 'home_possession', 'away_possession',
                        'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
                        'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards',
                        'home_fouls', 'away_fouls', 'home_corners', 'away_corners']
    numeric_cols_real = RAW_MATCH_NUMERIC_COLS_DERIVED  # these were made real by preprocess_percentage

    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False, na_filter=False)):
        print(f"  Processing raw match chunk {chunk_num + 1}...")
        chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_')

        rename_map = {
            'date': 'match_date',  # common rename
            # Add other renames based on your CSV vs. RAW_MATCH_COLS_FROM_CSV in config
            'sway_shots': 'away_shots', 'sway_shots_on_target': 'away_shots_on_target',
            'sway_rea_cards': 'away_red_cards', 'sway_yellow_cards': 'away_yellow_cards',
            'sway_fouls': 'away_fouls',
            'away_line_up': 'away_lineup', 'away_missin': 'away_missing',
        }
        chunk.rename(columns=rename_map, inplace=True)

        if 'match_date' in chunk.columns:
            chunk['match_date'] = pd.to_datetime(chunk['match_date'], errors='coerce').dt.date
        else:
            chunk['match_date'] = None

            # Create numeric from text for percentages
        for col_suffix in ['pass_completion', 'saves']:
            for team_type in ['home', 'away']:
                # Input col name from CSV e.g., 'home_pass_completion' becomes 'home_pass_completion_text'
                csv_col_name = f'{team_type}_{col_suffix}'
                text_col_db_name = f'{team_type}_{col_suffix}_text'
                numeric_col_db_name = f'{team_type}_{col_suffix}'  # This is the target numeric column

                if csv_col_name in chunk.columns:  # if CSV has 'home_pass_completion'
                    chunk.rename(columns={csv_col_name: text_col_db_name}, inplace=True)
                    chunk[numeric_col_db_name] = chunk[text_col_db_name].apply(preprocess_percentage)
                elif text_col_db_name in chunk.columns:  # if CSV already had '_text'
                    chunk[numeric_col_db_name] = chunk[text_col_db_name].apply(preprocess_percentage)
                else:  # Column missing entirely
                    chunk[text_col_db_name] = None
                    chunk[numeric_col_db_name] = None

        # Coerce types for other numeric columns
        for col in numeric_cols_int:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('Int64')
            else:
                chunk[col] = pd.NA
        for col in numeric_cols_real:  # these are derived cols like home_pass_completion
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype(float)
            else:
                chunk[col] = np.nan

        # Ensure all columns for matches_raw table are present
        current_cols = set(chunk.columns)
        for db_col in expected_db_cols_for_raw_matches:
            if db_col not in current_cols:
                # Assign appropriate NA type based on if it should be int or float
                if db_col in numeric_cols_int:
                    chunk[db_col] = pd.NA
                elif db_col in numeric_cols_real:
                    chunk[db_col] = np.nan
                else:
                    chunk[db_col] = None  # For text or object columns

        chunk_to_db = chunk[expected_db_cols_for_raw_matches]  # Select and order
        chunk_to_db.drop_duplicates(subset=['match_date', 'home_team', 'away_team', 'season'], keep='first',
                                    inplace=True)

        if chunk_to_db.empty:
            print(f"    Raw match chunk {chunk_num + 1} empty after processing/deduplication.")
            continue
        try:
            chunk_to_db.to_sql('matches_raw', engine, if_exists='append', index=False, method='multi')
            total_rows_inserted += len(chunk_to_db)
            print(f"    Inserted {len(chunk_to_db)} raw match rows. Total: {total_rows_inserted}")
        except Exception as e:
            print(f"    Error inserting raw match chunk {chunk_num + 1}: {e}")
            # print(chunk_to_db.head().to_dict(orient='records'))
            # chunk_to_db.to_csv(f"failing_raw_match_chunk_{chunk_num+1}.csv", index=False)

    print(f"Finished loading raw matches data. Total rows inserted: {total_rows_inserted}")
    return total_rows_inserted


if __name__ == '__main__':
    from config import ALL_TEAMS_CSV_PATH, MATCH_RESULTS_CSV_PATH  # For standalone testing

    # Ensure dummy files exist or point to real files for testing this module
    print("Testing data_loader.py...")
    # load_squads_data(ALL_TEAMS_CSV_PATH)
    # load_raw_matches_data(MATCH_RESULTS_CSV_PATH)
    print("Data loader testing complete.")