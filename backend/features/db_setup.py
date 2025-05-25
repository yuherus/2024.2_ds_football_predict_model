# football_project/db_setup.py
from sqlalchemy import text as sqlalchemy_text
from backend.features.utils import get_pg_engine  # Sửa đường dẫn import


def create_db_tables():
    """Creates the necessary tables in the PostgreSQL database if they don't exist."""
    engine = get_pg_engine()
    with engine.connect() as connection:
        # Squads table
        connection.execute(sqlalchemy_text('''
        CREATE TABLE IF NOT EXISTS squads (
            team TEXT, league TEXT, season INTEGER, jersey_number TEXT,
            player_name TEXT, position TEXT, dob_age TEXT, age INTEGER,
            nationality TEXT, market_value_text TEXT, market_value_eur REAL,
            on_loan BOOLEAN, loan_from TEXT, is_captain BOOLEAN,
            PRIMARY KEY (team, season, player_name)
        )
        '''))

        # Raw Matches table
        connection.execute(sqlalchemy_text('''
        CREATE TABLE IF NOT EXISTS matches_raw (
            id SERIAL PRIMARY KEY, match_date DATE, home_team TEXT, away_team TEXT, 
            home_score INTEGER, away_score INTEGER, league TEXT, season TEXT, source TEXT, 
            match_report_url TEXT, round INTEGER, venue TEXT, home_lineup TEXT, 
            away_lineup TEXT, home_missing TEXT, away_missing TEXT,
            home_possession INTEGER, away_possession INTEGER, home_shots INTEGER, away_shots INTEGER,
            home_shots_on_target INTEGER, away_shots_on_target INTEGER,
            home_pass_completion INTEGER, away_pass_completion INTEGER,
            home_red_cards INTEGER, away_red_cards INTEGER,
            home_yellow_cards INTEGER, away_yellow_cards INTEGER,
            home_saves INTEGER, away_saves INTEGER,
            home_fouls INTEGER, away_fouls INTEGER, home_corners INTEGER, away_corners INTEGER,
            CONSTRAINT unique_raw_match UNIQUE (match_date, home_team, away_team, season)
        )
        '''))
        connection.execute(sqlalchemy_text(
            '''CREATE INDEX IF NOT EXISTS idx_matches_raw_season_date ON matches_raw (season, match_date, round);'''))
        connection.execute(sqlalchemy_text(
            '''CREATE INDEX IF NOT EXISTS idx_matches_raw_teams_date ON matches_raw (home_team, away_team, match_date);'''))

        # Featured Matches table
        connection.execute(sqlalchemy_text('''
        CREATE TABLE IF NOT EXISTS matches_featured (
            match_date DATE, home_team TEXT, away_team TEXT, home_score INTEGER, away_score INTEGER,
            league TEXT, season TEXT, source TEXT, match_report_url TEXT, round INTEGER, venue TEXT,
            home_lineup TEXT, away_lineup TEXT, home_missing TEXT, away_missing TEXT,
            home_possession INTEGER, away_possession INTEGER, home_shots INTEGER, away_shots INTEGER,
            home_shots_on_target INTEGER, away_shots_on_target INTEGER,
            home_pass_completion INTEGER, away_pass_completion INTEGER,
            home_red_cards INTEGER, away_red_cards INTEGER, home_yellow_cards INTEGER, away_yellow_cards INTEGER,
            home_saves INTEGER, away_saves INTEGER,
            home_fouls INTEGER, away_fouls INTEGER, home_corners INTEGER, away_corners INTEGER,            
            home_points_last_5 REAL, away_points_last_5 REAL, home_standing REAL, away_standing REAL,
            standing_diff REAL, home_points REAL, away_points REAL, points_diff REAL,
            home_goals_scored_last_5 REAL, home_goals_conceded_last_5 REAL,
            away_goals_scored_last_5 REAL, away_goals_conceded_last_5 REAL,
            home_goal_diff REAL, away_goal_diff REAL, home_home_win_rate REAL, away_away_win_rate REAL,
            home_win_streak INTEGER, away_win_streak INTEGER, home_loss_streak INTEGER, away_loss_streak INTEGER,
            h2h_home_wins INTEGER, h2h_away_wins INTEGER, h2h_draws INTEGER,
            h2h_home_goals_avg REAL, h2h_away_goals_avg REAL,
            is_early_season INTEGER, is_mid_season INTEGER, is_late_season INTEGER,
            home_possession_avg REAL, away_possession_avg REAL, home_shots_on_target_avg REAL,
            away_shots_on_target_avg REAL, home_corners_avg REAL, away_corners_avg REAL,
            match_result INTEGER,
            PRIMARY KEY (match_date, home_team, away_team, season)
        )
        '''))
        connection.execute(sqlalchemy_text(
            '''CREATE INDEX IF NOT EXISTS idx_matches_featured_season_date ON matches_featured (season, match_date, round);'''))
        
        # Match Predictions table - Để hiển thị kết quả dự đoán lên website
        connection.execute(sqlalchemy_text('''
        CREATE TABLE IF NOT EXISTS match_predictions (
            id SERIAL PRIMARY KEY,
            match_date TIMESTAMP WITH TIME ZONE,
            round INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_win_prob REAL,
            draw_prob REAL,
            away_win_prob REAL,
            home_win_pct TEXT,
            draw_pct TEXT,
            away_win_pct TEXT,
            league TEXT,
            season TEXT,
            venue TEXT,
            prediction_model TEXT,
            predicted_result INTEGER,
            actual_result INTEGER,
            prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_prediction UNIQUE (match_date, home_team, away_team, prediction_model)
        )
        '''))
        connection.execute(sqlalchemy_text(
            '''CREATE INDEX IF NOT EXISTS idx_match_predictions_date ON match_predictions (match_date);'''))
        
        connection.commit()
        print("PostgreSQL tables 'squads', 'matches_raw', 'matches_featured', and 'match_predictions' checked/created.")


if __name__ == '__main__':
    print("Setting up database tables...")
    create_db_tables()
    print("Database setup complete.")