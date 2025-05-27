import pandas as pd
import numpy as np
from sqlalchemy import text
from backend.features.utils import get_pg_engine
from tqdm import tqdm
from collections import Counter

def calculate_championship_probabilities(league, season, prediction_model, n_simulations=100000):
    """
    Calculate championship probabilities using Monte Carlo simulation.
    
    Args:
        league (str): League name
        season (str): Season
        prediction_model (str): Model name ('lstm' or 'xgboost')
        n_simulations (int): Number of Monte Carlo simulations to run
    """
    engine = get_pg_engine()
    
    # First, get all teams in the league for this season
    teams_query = """
    SELECT DISTINCT team_name FROM (
        SELECT home_team as team_name FROM match_predictions
        WHERE league = :league AND season = :season AND prediction_model = :model
        UNION
        SELECT away_team as team_name FROM match_predictions
        WHERE league = :league AND season = :season AND prediction_model = :model
    ) teams
    ORDER BY team_name
    """
    
    try:
        teams_df = pd.read_sql_query(
            text(teams_query),
            engine,
            params={'league': league, 'season': season, 'model': prediction_model}
        )
        all_teams = teams_df['team_name'].tolist()
    except Exception as e:
        print(f"Error getting teams: {str(e)}")
        return None
    
    if not all_teams:
        print(f"No teams found for {league} {season} with model {prediction_model}")
        return None
    
    # Get current points for each team based on actual_result
    points_query = """
    WITH team_points AS (
        SELECT 
            home_team as team,
            SUM(CASE 
                WHEN actual_result = 0 THEN 3  -- Home win (actual_result = 0)
                WHEN actual_result = 1 THEN 1  -- Draw (actual_result = 1)
                ELSE 0  -- Home loss (actual_result = 2)
            END) as points,
            COUNT(*) as matches_played
        FROM match_predictions
        WHERE league = :league 
        AND season = :season
        AND prediction_model = :model
        AND actual_result IS NOT NULL
        GROUP BY home_team
        
        UNION ALL
        
        SELECT 
            away_team as team,
            SUM(CASE 
                WHEN actual_result = 2 THEN 3  -- Away win (actual_result = 2)
                WHEN actual_result = 1 THEN 1  -- Draw (actual_result = 1)
                ELSE 0  -- Away loss (actual_result = 0)
            END) as points,
            COUNT(*) as matches_played
        FROM match_predictions
        WHERE league = :league 
        AND season = :season
        AND prediction_model = :model
        AND actual_result IS NOT NULL
        GROUP BY away_team
    )
    SELECT 
        team,
        COALESCE(SUM(points), 0) as total_points,
        COALESCE(SUM(matches_played), 0) as total_matches_played
    FROM team_points
    GROUP BY team
    """
    
    try:
        points_df = pd.read_sql_query(
            text(points_query),
            engine,
            params={'league': league, 'season': season, 'model': prediction_model}
        )
    except Exception as e:
        print(f"Error getting points: {str(e)}")
        return None
    
    # Create a complete points dictionary for all teams
    current_points = {}
    matches_played = {}
    
    for team in all_teams:
        team_data = points_df[points_df['team'] == team]
        if not team_data.empty:
            current_points[team] = int(float(team_data['total_points'].iloc[0]))
            matches_played[team] = int(float(team_data['total_matches_played'].iloc[0]))
        else:
            current_points[team] = 0
            matches_played[team] = 0
    
    # Get remaining matches (where actual_result IS NULL)
    remaining_query = """
    SELECT 
        home_team,
        away_team,
        home_win_prob,
        draw_prob,
        away_win_prob
    FROM match_predictions
    WHERE league = :league 
    AND season = :season
    AND prediction_model = :model
    AND actual_result IS NULL
    ORDER BY match_date
    """
    
    try:
        remaining_df = pd.read_sql_query(
            text(remaining_query),
            engine,
            params={'league': league, 'season': season, 'model': prediction_model}
        )
    except Exception as e:
        print(f"Error getting remaining matches: {str(e)}")
        return None
    
    # Calculate total matches per team and remaining matches
    total_matches_query = """
    SELECT 
        team,
        COUNT(*) as total_matches
    FROM (
        SELECT home_team as team FROM match_predictions
        WHERE league = :league AND season = :season AND prediction_model = :model
        UNION ALL
        SELECT away_team as team FROM match_predictions
        WHERE league = :league AND season = :season AND prediction_model = :model
    ) all_matches
    GROUP BY team
    """
    
    try:
        total_matches_df = pd.read_sql_query(
            text(total_matches_query),
            engine,
            params={'league': league, 'season': season, 'model': prediction_model}
        )
    except Exception as e:
        print(f"Error getting total matches: {str(e)}")
        return None
    
    # Create total matches dictionary
    total_matches_per_team = {}
    for team in all_teams:
        team_data = total_matches_df[total_matches_df['team'] == team]
        if not team_data.empty:
            total_matches_per_team[team] = int(float(team_data['total_matches'].iloc[0]))
        else:
            total_matches_per_team[team] = 0
    
    # Calculate remaining matches for each team
    remaining_matches_per_team = {}
    for team in all_teams:
        remaining_matches_per_team[team] = total_matches_per_team[team] - matches_played[team]
    
    print(f"\nModel: {prediction_model}")
    print(f"Found {len(remaining_df)} remaining matches to simulate")
    print(f"Teams and their remaining matches:")
    for team in sorted(all_teams):
        print(f"  {team}: {remaining_matches_per_team[team]} matches remaining ({matches_played[team]}/{total_matches_per_team[team]} played), {current_points[team]} points")
    
    if remaining_df.empty:
        print(f"No remaining matches found for {league} {season}")
        # If no remaining matches, current standings are final
        results = []
        for team in all_teams:
            results.append({
                'team_name': team,
                'championship_probability': 1.0 if current_points[team] == max(current_points.values()) else 0.0,
                'points': current_points[team],
                'form': '',
                'league': league,
                'season': season,
                'remaining_matches': 0,
                'max_possible_points': current_points[team],
                'prediction_model': prediction_model
            })
        
        # Sort by points
        results.sort(key=lambda x: x['points'], reverse=True)
        
        # Add rank
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        return results
    
    # Initialize results
    championship_counts = {team: 0 for team in all_teams}
    total_points = {team: [] for team in all_teams}
    
    # Run Monte Carlo simulations
    for sim in tqdm(range(n_simulations), desc=f"Running Monte Carlo simulations for {league} {season} ({prediction_model})"):
        # Initialize points for this simulation
        sim_points = current_points.copy()
        
        # Simulate each remaining match
        for _, match in remaining_df.iterrows():
            try:
                # Get probabilities and normalize
                probs = [float(match['home_win_prob']), float(match['draw_prob']), float(match['away_win_prob'])]
                probs = np.array(probs)
                
                # Handle NaN or invalid probabilities
                if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                    probs = np.array([0.33, 0.33, 0.34])  # Default equal probabilities
                else:
                    probs = probs / np.sum(probs)  # Normalize
                

                # Generate outcome: 0=home_win, 1=draw, 2=away_win
                outcome = np.random.choice([0, 1, 2], p=probs)
                
                # Update points based on outcome
                if outcome == 0:  # Home win
                    sim_points[match['home_team']] += 3
                elif outcome == 1:  # Draw
                    sim_points[match['home_team']] += 1
                    sim_points[match['away_team']] += 1
                else:  # Away win (outcome == 2)
                    sim_points[match['away_team']] += 3
                    
            except Exception as e:
                print(f"Error in simulation for match {match['home_team']} vs {match['away_team']}: {str(e)}")
                continue
        
        # Record final points for this simulation
        for team in all_teams:
            total_points[team].append(sim_points[team])
        
        # Find champion(s) for this simulation
        max_points = max(sim_points.values())
        champions = [team for team, points in sim_points.items() if points == max_points]
        
        # Calculate championship probability based on points difference
        for team in all_teams:
            if sim_points[team] == max_points:
                # If team has highest points, they win
                championship_counts[team] += 1
                print(f"\n{team} has {sim_points[team]} points")
    
    # Get form for each team
    team_forms = {}
    for team in all_teams:
        form_query = """
        WITH team_matches AS (
            SELECT 
                match_date,
                CASE 
                    WHEN home_team = :team AND actual_result = 0 THEN 'W'
                    WHEN away_team = :team AND actual_result = 2 THEN 'W'
                    WHEN actual_result = 1 THEN 'D'
                    ELSE 'L'
                END as result
            FROM match_predictions
            WHERE (home_team = :team OR away_team = :team)
            AND league = :league 
            AND season = :season
            AND prediction_model = :model
            AND actual_result IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 5
        )
        SELECT STRING_AGG(result, '' ORDER BY match_date DESC) as form
        FROM team_matches;
        """
        
        try:
            form_result = pd.read_sql_query(
                text(form_query),
                engine,
                params={'team': team, 'league': league, 'season': season, 'model': prediction_model}
            )
            team_forms[team] = form_result['form'].iloc[0] if not form_result.empty and form_result['form'].iloc[0] else ''
        except:
            team_forms[team] = ''
    
    # Calculate probabilities and prepare results
    results = []
    for team in all_teams:
        championship_prob = championship_counts[team] / n_simulations
        avg_points = np.mean(total_points[team]) if total_points[team] else current_points[team]
        max_possible_points = current_points[team] + (remaining_matches_per_team[team] * 3)
        
        results.append({
            'team_name': team,
            'championship_probability': float(championship_prob),
            'points': int(current_points[team]),
            'form': team_forms[team],
            'league': league,
            'season': season,
            'remaining_matches': int(remaining_matches_per_team[team]),
            'max_possible_points': int(max_possible_points),
            'prediction_model': prediction_model
        })
    
    # Sort by championship probability, then by current points
    results.sort(key=lambda x: (x['championship_probability'], x['points']), reverse=True)
    
    # Add rank
    for i, result in enumerate(results, 1):
        result['rank'] = int(i)
    
    # Insert results into database
    try:
        with engine.begin() as conn:
            # Clear existing results for this league, season and model
            delete_query = """
            DELETE FROM championship_probabilities 
            WHERE league = :league AND season = :season AND prediction_model = :model
            """
            conn.execute(text(delete_query), {'league': league, 'season': season, 'model': prediction_model})
            
            # Insert new results
            for result in results:
                insert_query = """
                INSERT INTO championship_probabilities 
                (rank, team_name, championship_probability, points, form, league, season, 
                 remaining_matches, max_possible_points, prediction_model)
                VALUES 
                (:rank, :team_name, :championship_probability, :points, :form, :league, :season,
                 :remaining_matches, :max_possible_points, :prediction_model)
                """
                conn.execute(text(insert_query), result)
                
        print(f"Successfully inserted results for {league} {season} ({prediction_model})")
    except Exception as e:
        print(f"Error inserting results into database: {str(e)}")
        return None
    
    return results