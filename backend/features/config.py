# football_project/config.py

# --- PostgreSQL Connection Details ---
# IMPORTANT: Replace with your actual credentials!
DB_CONFIG = {
    "dbname": "CHANGEME",
    "user": "CHANGEME",
    "password": "CHANGEME",
    "host": "localhost",
    "port": 5432
}

# --- File Paths ---
# IMPORTANT: Replace with paths to your actual CSV files!
ALL_PLAYERS_CSV_PATH = 'backend/data/all_player_values.csv'
MATCH_RESULTS_CSV_PATH = 'backend/data/match_results.csv'

# --- Processing Constants ---
CHUNK_SIZE = 10000  # For reading large CSVs

# --- Column Definitions (for consistency) ---
# Raw match columns expected from CSV after basic renaming (see data_loader.py)
RAW_MATCH_COLS_FROM_CSV = [
    'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'league', 'season',
    'source', 'match_report_url', 'round', 'venue', 'home_lineup', 'away_lineup',
    'home_missing', 'away_missing', 'home_possession', 'away_possession', 'home_shots',
    'away_shots', 'home_shots_on_target', 'away_shots_on_target',
    # These are text columns from CSV; numeric ones are derived
    'home_pass_completion_text', 'away_pass_completion_text',
    'home_saves_text', 'away_saves_text',
    'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards',
    'home_fouls', 'away_fouls', 'home_corners', 'away_corners'
]
# Derived numeric columns (created during raw data loading)
RAW_MATCH_NUMERIC_COLS_DERIVED = [
    'home_pass_completion', 'away_pass_completion', 'home_saves', 'away_saves'
]

# Columns for the 'matches_featured' table (raw columns + engineered features)
# This list helps ensure consistency when creating and inserting data
# Order matters for to_sql if we select columns explicitly.
MATCHES_FEATURED_COLUMN_ORDER = [
    'match_date', 'home_team', 'away_team', 'home_score', 'away_score',
    'league', 'season', 'source', 'match_report_url', 'round', 'venue',
    'home_lineup', 'away_lineup', 'home_missing', 'away_missing',
    'home_possession', 'away_possession', 'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target',
    'home_pass_completion_text', 'away_pass_completion_text',
    'home_pass_completion', 'away_pass_completion',
    'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards',
    'home_saves_text', 'away_saves_text', 'home_saves', 'away_saves',
    'home_fouls', 'away_fouls', 'home_corners', 'away_corners', # End of raw match cols
    'home_points_last_5', 'away_points_last_5', 'home_standing', 'away_standing',
    'standing_diff', 'home_points', 'away_points', 'points_diff',
    'home_goals_scored_last_5', 'home_goals_conceded_last_5',
    'away_goals_scored_last_5', 'away_goals_conceded_last_5',
    'home_goal_diff', 'away_goal_diff', 'home_home_win_rate', 'away_away_win_rate',
    'home_win_streak', 'away_win_streak', 'home_loss_streak', 'away_loss_streak',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals_avg', 'h2h_away_goals_avg',
    'is_early_season', 'is_mid_season', 'is_late_season',
    'home_possession_avg', 'away_possession_avg', 'home_shots_on_target_avg',
    'away_shots_on_target_avg', 'home_corners_avg', 'away_corners_avg', 'match_result'
]
# Dtypes for featured columns for type casting before DB insert
FEATURED_FLOAT_COLS = [
    'home_points_last_5', 'away_points_last_5', 'home_standing', 'away_standing', 'standing_diff',
    'home_points', 'away_points', 'points_diff', 'home_goals_scored_last_5', 'home_goals_conceded_last_5',
    'away_goals_scored_last_5', 'away_goals_conceded_last_5', 'home_goal_diff', 'away_goal_diff',
    'home_home_win_rate', 'away_away_win_rate', 'h2h_home_goals_avg', 'h2h_away_goals_avg',
    'home_possession_avg', 'away_possession_avg', 'home_shots_on_target_avg',
    'away_shots_on_target_avg', 'home_corners_avg', 'away_corners_avg'
]
FEATURED_INT_COLS = [ # These can be Int64 in pandas to support NA
    'home_win_streak', 'away_win_streak', 'home_loss_streak', 'away_loss_streak',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
    'is_early_season', 'is_mid_season', 'is_late_season', 'match_result'
]