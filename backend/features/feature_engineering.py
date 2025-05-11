# football_project/feature_engineering.py
import pandas as pd
import numpy as np
from collections import deque
from utils import get_pg_engine, get_points  # Import from local utils
from config import MATCHES_FEATURED_COLUMN_ORDER, FEATURED_FLOAT_COLS, FEATURED_INT_COLS


def calculate_standings_for_season(matches_df_season_upto_round):
    """Calculates standings for a given set of matches (e.g., a season up to a round)."""
    standings = {}
    # Ensure home_team and away_team are not NaN before concatenation
    valid_home_teams = matches_df_season_upto_round['home_team'].dropna()
    valid_away_teams = matches_df_season_upto_round['away_team'].dropna()
    if valid_home_teams.empty and valid_away_teams.empty:
        return {}  # No teams to process
    teams = pd.concat([valid_home_teams, valid_away_teams]).unique()

    for team in teams: standings[team] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}

    for _, row in matches_df_season_upto_round.iterrows():
        ht, at = row['home_team'], row['away_team']
        hs, as_ = row['home_score'], row['away_score']

        if pd.isna(ht) or pd.isna(at) or pd.isna(hs) or pd.isna(as_): continue  # Skip incomplete rows

        hp, ap = get_points(hs, as_)
        if pd.isna(hp) or pd.isna(ap): continue

        # Update Home Team
        standings[ht]['P'] += 1;
        standings[ht]['GF'] += hs;
        standings[ht]['GA'] += as_;
        standings[ht]['Pts'] += hp
        if hp == 3:
            standings[ht]['W'] += 1
        elif hp == 1:
            standings[ht]['D'] += 1
        else:
            standings[ht]['L'] += 1

        # Update Away Team
        standings[at]['P'] += 1;
        standings[at]['GF'] += as_;
        standings[at]['GA'] += hs;
        standings[at]['Pts'] += ap
        if ap == 3:
            standings[at]['W'] += 1
        elif ap == 1:
            standings[at]['D'] += 1
        else:
            standings[at]['L'] += 1

    for team in standings: standings[team]['GD'] = standings[team]['GF'] - standings[team]['GA']

    standings_df = pd.DataFrame.from_dict(standings, orient='index')
    if standings_df.empty: return {}

    standings_df = standings_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False]).reset_index()
    standings_df['standing'] = range(1, len(standings_df) + 1)
    return standings_df.set_index('index').to_dict(orient='index')


def engineer_features_for_season(season_df, engine):
    """
    Engineers features for a single season's DataFrame.
    H2H requires querying the full matches_raw table via `engine`.
    """
    current_season_id = season_df['season'].iloc[0] if not season_df.empty else "Unknown"
    print(f"  Engineering features for season: {current_season_id}")

    # Ensure critical columns are numeric
    numeric_cols_to_check = [
        'home_score', 'away_score', 'round', 'home_possession', 'away_possession',
        'home_shots_on_target', 'away_shots_on_target', 'home_corners', 'away_corners',
        'home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards'
    ]
    for col in numeric_cols_to_check:
        if col in season_df.columns:
            season_df[col] = pd.to_numeric(season_df[col], errors='coerce')
        else:  # Add missing columns as NaN to prevent errors later
            season_df[col] = np.nan

    season_df['match_date_dt'] = pd.to_datetime(season_df['match_date'], errors='coerce')
    # Drop rows where date conversion failed as they are unusable for time-series
    season_df.dropna(subset=['match_date_dt'], inplace=True)
    season_df = season_df.sort_values(by=['match_date_dt', 'round']).reset_index(drop=True)

    all_features_list = []
    team_histories = {}  # Reset for each season processing loop

    for index, current_match in season_df.iterrows():
        if index % 100 == 0:  # Progress update less frequently for larger seasons
            print(f"    Processing match {index + 1}/{len(season_df)} in season {current_season_id}...")

        features = current_match.to_dict()  # Start with all raw match data
        current_date_dt = current_match['match_date_dt']
        # current_season_val = current_match['season'] # Redundant, it's current_season_id
        current_round = current_match['round']
        home_team, away_team = current_match['home_team'], current_match['away_team']

        if pd.isna(home_team) or pd.isna(away_team):  # Skip if teams are missing
            print(f"    Skipping match index {index} due to missing team names.")
            continue

        # --- Initialize team histories if not present for THIS season ---
        for team in [home_team, away_team]:
            if team not in team_histories:
                team_histories[team] = {
                    'points': deque(maxlen=5), 'goals_scored': deque(maxlen=5),
                    'goals_conceded': deque(maxlen=5), 'results': deque(),  # For streaks
                    'home_games_played': 0, 'home_wins': 0,
                    'away_games_played': 0, 'away_wins': 0,
                    'total_possession_sum': 0.0, 'total_shots_on_target_sum': 0.0,  # Use floats for sums
                    'total_corners_sum': 0.0, 'games_for_avg_stats': 0
                }
        home_hist = team_histories[home_team]
        away_hist = team_histories[away_team]

        # --- Standings (based on current_season_df up to this match) ---
        past_matches_this_season_df = season_df[
            (season_df['match_date_dt'] < current_date_dt) |
            ((season_df['match_date_dt'] == current_date_dt) & (
                        pd.to_numeric(season_df['round'], errors='coerce') < pd.to_numeric(current_round,
                                                                                           errors='coerce')))
            ]

        current_standings_map = {}
        if not past_matches_this_season_df.empty:
            current_standings_map = calculate_standings_for_season(past_matches_this_season_df)

        features['home_standing'] = current_standings_map.get(home_team, {}).get('standing', np.nan)
        features['away_standing'] = current_standings_map.get(away_team, {}).get('standing', np.nan)
        features['home_points'] = float(current_standings_map.get(home_team, {}).get('Pts', 0.0))
        features['away_points'] = float(current_standings_map.get(away_team, {}).get('Pts', 0.0))

        features['standing_diff'] = (features['away_standing'] - features['home_standing']) if pd.notna(
            features['home_standing']) and pd.notna(features['away_standing']) else np.nan
        features['points_diff'] = features['home_points'] - features['away_points']

        # --- Form (Last 5 games IN THIS SEASON from team_histories) ---
        features['home_points_last_5'] = float(sum(p for p in home_hist['points'] if pd.notna(p)))
        features['away_points_last_5'] = float(sum(p for p in away_hist['points'] if pd.notna(p)))

        goals_scored_home_valid = [g for g in home_hist['goals_scored'] if pd.notna(g)]
        goals_conceded_home_valid = [g for g in home_hist['goals_conceded'] if pd.notna(g)]
        goals_scored_away_valid = [g for g in away_hist['goals_scored'] if pd.notna(g)]
        goals_conceded_away_valid = [g for g in away_hist['goals_conceded'] if pd.notna(g)]

        features['home_goals_scored_last_5'] = sum(goals_scored_home_valid) / len(
            goals_scored_home_valid) if goals_scored_home_valid else 0.0
        features['home_goals_conceded_last_5'] = sum(goals_conceded_home_valid) / len(
            goals_conceded_home_valid) if goals_conceded_home_valid else 0.0
        features['away_goals_scored_last_5'] = sum(goals_scored_away_valid) / len(
            goals_scored_away_valid) if goals_scored_away_valid else 0.0
        features['away_goals_conceded_last_5'] = sum(goals_conceded_away_valid) / len(
            goals_conceded_away_valid) if goals_conceded_away_valid else 0.0

        features['home_goal_diff'] = float(current_standings_map.get(home_team, {}).get('GD', 0.0))
        features['away_goal_diff'] = float(current_standings_map.get(away_team, {}).get('GD', 0.0))

        features['home_home_win_rate'] = (home_hist['home_wins'] / home_hist['home_games_played']) if home_hist[
                                                                                                          'home_games_played'] > 0 else 0.0
        features['away_away_win_rate'] = (away_hist['away_wins'] / away_hist['away_games_played']) if away_hist[
                                                                                                          'away_games_played'] > 0 else 0.0

        def calculate_streak(results_deque, streak_type):
            streak = 0
            for res in reversed(results_deque):
                if res == streak_type:
                    streak += 1
                else:
                    break
            return streak

        features['home_win_streak'] = calculate_streak(home_hist['results'], 'W')
        features['away_win_streak'] = calculate_streak(away_hist['results'], 'W')
        features['home_loss_streak'] = calculate_streak(home_hist['results'], 'L')
        features['away_loss_streak'] = calculate_streak(away_hist['results'], 'L')

        # --- Head-to-Head (H2H from full DB history) ---
        h2h_query = f"""
            SELECT home_team, away_team, home_score, away_score FROM matches_raw
            WHERE ((home_team = %(h_team)s AND away_team = %(a_team)s) OR (home_team = %(a_team)s AND away_team = %(h_team)s))
            AND match_date < %(c_date)s
            ORDER BY match_date DESC LIMIT 5
        """
        params = {'h_team': home_team, 'a_team': away_team, 'c_date': current_date_dt.date()}
        h2h_matches_df = pd.read_sql_query(h2h_query, engine, params=params)

        h2h_home_wins_count, h2h_away_wins_count, h2h_draws_count = 0, 0, 0
        h2h_home_goals_sum, h2h_away_goals_sum = 0.0, 0.0
        if not h2h_matches_df.empty:
            valid_h2h_matches = 0
            for _, h2h_row in h2h_matches_df.iterrows():
                if pd.isna(h2h_row['home_score']) or pd.isna(h2h_row['away_score']): continue
                valid_h2h_matches += 1
                hs_h2h, as_h2h = h2h_row['home_score'], h2h_row['away_score']
                if h2h_row['home_team'] == home_team:
                    h2h_home_goals_sum += hs_h2h;
                    h2h_away_goals_sum += as_h2h
                    if hs_h2h > as_h2h:
                        h2h_home_wins_count += 1
                    elif hs_h2h < as_h2h:
                        h2h_away_wins_count += 1
                    else:
                        h2h_draws_count += 1
                else:
                    h2h_home_goals_sum += as_h2h;
                    h2h_away_goals_sum += hs_h2h
                    if as_h2h > hs_h2h:
                        h2h_home_wins_count += 1
                    elif as_h2h < hs_h2h:
                        h2h_away_wins_count += 1
                    else:
                        h2h_draws_count += 1
            features['h2h_home_wins'] = h2h_home_wins_count
            features['h2h_away_wins'] = h2h_away_wins_count
            features['h2h_draws'] = h2h_draws_count
            features['h2h_home_goals_avg'] = h2h_home_goals_sum / valid_h2h_matches if valid_h2h_matches > 0 else 0.0
            features['h2h_away_goals_avg'] = h2h_away_goals_sum / valid_h2h_matches if valid_h2h_matches > 0 else 0.0
        else:
            features.update({'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
                             'h2h_home_goals_avg': 0.0, 'h2h_away_goals_avg': 0.0})

        # --- Time Features ---
        current_round_num = pd.to_numeric(current_match.get('round'), errors='coerce')  # Ensure 'round' is numeric
        features['round_number'] = current_round_num
        features['is_early_season'] = 1 if pd.notna(current_round_num) and current_round_num < 10 else 0
        features['is_mid_season'] = 1 if pd.notna(current_round_num) and 10 <= current_round_num < 28 else 0
        features['is_late_season'] = 1 if pd.notna(current_round_num) and current_round_num >= 28 else 0

        # --- Style of Play (Season Averages from team_histories) ---
        features['home_possession_avg'] = (home_hist['total_possession_sum'] / home_hist['games_for_avg_stats']) if \
        home_hist['games_for_avg_stats'] > 0 else np.nan
        features['away_possession_avg'] = (away_hist['total_possession_sum'] / away_hist['games_for_avg_stats']) if \
        away_hist['games_for_avg_stats'] > 0 else np.nan
        features['home_shots_on_target_avg'] = (
                    home_hist['total_shots_on_target_sum'] / home_hist['games_for_avg_stats']) if home_hist[
                                                                                                      'games_for_avg_stats'] > 0 else np.nan
        features['away_shots_on_target_avg'] = (
                    away_hist['total_shots_on_target_sum'] / away_hist['games_for_avg_stats']) if away_hist[
                                                                                                      'games_for_avg_stats'] > 0 else np.nan
        features['home_corners_avg'] = (home_hist['total_corners_sum'] / home_hist['games_for_avg_stats']) if home_hist[
                                                                                                                  'games_for_avg_stats'] > 0 else np.nan
        features['away_corners_avg'] = (away_hist['total_corners_sum'] / away_hist['games_for_avg_stats']) if away_hist[
                                                                                                                  'games_for_avg_stats'] > 0 else np.nan

        all_features_list.append(features)

        # --- UPDATE team_histories (for current season) ---
        home_s, away_s = current_match['home_score'], current_match['away_score']
        if pd.notna(home_s) and pd.notna(away_s):  # Only update if scores are valid
            home_pts_earned, away_pts_earned = get_points(home_s, away_s)
            if not (pd.isna(home_pts_earned) or pd.isna(away_pts_earned)):
                home_hist['points'].append(home_pts_earned);
                home_hist['goals_scored'].append(home_s);
                home_hist['goals_conceded'].append(away_s)
                if home_pts_earned == 3:
                    home_hist['results'].append('W')
                elif home_pts_earned == 1:
                    home_hist['results'].append('D')
                else:
                    home_hist['results'].append('L')
                home_hist['home_games_played'] += 1
                if home_pts_earned == 3: home_hist['home_wins'] += 1

                if pd.notna(current_match.get('home_possession')): home_hist['total_possession_sum'] += current_match[
                    'home_possession']
                if pd.notna(current_match.get('home_shots_on_target')): home_hist['total_shots_on_target_sum'] += \
                current_match['home_shots_on_target']
                if pd.notna(current_match.get('home_corners')): home_hist['total_corners_sum'] += current_match[
                    'home_corners']
                home_hist['games_for_avg_stats'] += 1

                away_hist['points'].append(away_pts_earned);
                away_hist['goals_scored'].append(away_s);
                away_hist['goals_conceded'].append(home_s)
                if away_pts_earned == 3:
                    away_hist['results'].append('W')
                elif away_pts_earned == 1:
                    away_hist['results'].append('D')
                else:
                    away_hist['results'].append('L')
                away_hist['away_games_played'] += 1
                if away_pts_earned == 3: away_hist['away_wins'] += 1

                if pd.notna(current_match.get('away_possession')): away_hist['total_possession_sum'] += current_match[
                    'away_possession']
                if pd.notna(current_match.get('away_shots_on_target')): away_hist['total_shots_on_target_sum'] += \
                current_match['away_shots_on_target']
                if pd.notna(current_match.get('away_corners')): away_hist['total_corners_sum'] += current_match[
                    'away_corners']
                away_hist['games_for_avg_stats'] += 1

    if not all_features_list: return pd.DataFrame()  # Return empty if no features were generated

    featured_df = pd.DataFrame(all_features_list)

    # --- Target Variable (y) ---
    conditions = [featured_df['home_score'] < featured_df['away_score'],
                  featured_df['home_score'] == featured_df['away_score'],
                  featured_df['home_score'] > featured_df['away_score']]
    choices = [0, 1, 2]  # home_loss, draw, home_win
    featured_df['match_result'] = np.select(conditions, choices, default=pd.NA)  # Use pd.NA for Int64

    featured_df.drop(columns=['match_date_dt'], inplace=True, errors='ignore')

    # Cast to correct dtypes for DB insertion (using lists from config)
    for col in FEATURED_FLOAT_COLS:
        if col in featured_df.columns:
            featured_df[col] = pd.to_numeric(featured_df[col], errors='coerce').astype(float)
        else:
            featured_df[col] = np.nan  # Add if missing, as float
    for col in FEATURED_INT_COLS:
        if col in featured_df.columns:
            featured_df[col] = pd.to_numeric(featured_df[col], errors='coerce').astype('Int64')
        else:
            featured_df[col] = pd.NA  # Add if missing, as Int64

    # Ensure all columns expected by `matches_featured` table are present and in order
    final_df_cols = []
    for col in MATCHES_FEATURED_COLUMN_ORDER:
        if col in featured_df.columns:
            final_df_cols.append(col)
        # else, the column was not in the original raw data or wasn't generated, so it won't be in the final df
        # This is okay if the table definition allows NULLs for those new feature columns.
        # The casting loop above should have added them as NaN/NA if defined in FEATURED_..._COLS

    # Select and reorder columns to match the defined order
    # Handle missing columns gracefully (they should have been added as NaN/NA by previous step)
    df_for_db = pd.DataFrame()
    for col in MATCHES_FEATURED_COLUMN_ORDER:
        if col in featured_df.columns:
            df_for_db[col] = featured_df[col]
        else:
            # Assign correct NA type based on config list for new features if not present
            if col in FEATURED_INT_COLS:
                df_for_db[col] = pd.NA
            elif col in FEATURED_FLOAT_COLS:
                df_for_db[col] = np.nan
            else:
                df_for_db[col] = None  # For other types like text, if they were feature engineered

    return df_for_db


def process_all_seasons_and_store_features():
    engine = get_pg_engine()
    seasons_query = "SELECT DISTINCT season FROM matches_raw WHERE season IS NOT NULL ORDER BY season;"
    try:
        distinct_seasons = pd.read_sql_query(seasons_query, engine)['season'].tolist()
    except Exception as e:
        print(f"Error fetching distinct seasons: {e}")
        return

    print(f"Found seasons to process for feature engineering: {distinct_seasons}")

    for season_val in distinct_seasons:
        print(f"\nProcessing season for features: {season_val}")
        season_matches_query = "SELECT * FROM matches_raw WHERE season = %(s_val)s ORDER BY match_date, round;"
        season_df = pd.read_sql_query(season_matches_query, engine, params={'s_val': season_val})

        if season_df.empty:
            print(f"  No matches found for season {season_val}. Skipping.")
            continue

        featured_season_df = engineer_features_for_season(season_df.copy(), engine)  # Pass a copy

        if not featured_season_df.empty:
            try:
                # Use MATCHES_FEATURED_COLUMN_ORDER from config to ensure correct columns/order
                featured_season_df_to_db = featured_season_df[MATCHES_FEATURED_COLUMN_ORDER]

                featured_season_df_to_db.to_sql('matches_featured', engine, if_exists='append', index=False,
                                                method='multi', chunksize=1000)
                print(f"  Stored {len(featured_season_df_to_db)} featured matches for season {season_val}.")
            except Exception as e:
                print(f"  Error storing featured matches for season {season_val}: {e}")
                # featured_season_df_to_db.to_csv(f"failing_featured_season_{season_val}.csv", index=False)
        else:
            print(f"  No features generated for season {season_val}.")
    print("\nFinished feature engineering for all seasons.")


if __name__ == '__main__':
    print("Testing feature_engineering.py...")
    # This requires 'matches_raw' to be populated.
    # For standalone test, you might load a sample season from a CSV here.
    # Example:
    # test_engine = get_pg_engine()
    # sample_season_df = pd.read_sql_query("SELECT * FROM matches_raw WHERE season = '2020-2021' LIMIT 100;", test_engine) # Get some data
    # if not sample_season_df.empty:
    #     featured = engineer_features_for_season(sample_season_df, test_engine)
    #     print("Sample featured data head:")
    #     print(featured.head())
    # else:
    #     print("No sample data in matches_raw for testing.")
    #
    # Or run the full process:
    # process_all_seasons_and_store_features()
    print("Feature engineering testing placeholder complete.")