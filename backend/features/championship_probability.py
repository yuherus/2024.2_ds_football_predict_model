import pandas as pd
import numpy as np
from sqlalchemy import text
from backend.features.utils import get_pg_engine
from tqdm import tqdm
from collections import Counter

def calculate_championship_probabilities(league, season, n_simulations=10000):
    """
    Calculate championship probabilities using Monte Carlo simulation.
    
    Args:
        league (str): League name
        season (str): Season
        n_simulations (int): Number of Monte Carlo simulations to run
    """
    engine = get_pg_engine()
    
    # Get remaining matches and current standings
    query = """
    WITH team_points AS (
        SELECT 
            home_team as team,
            SUM(CASE 
                WHEN actual_result = 1 THEN 3  -- Home win
                WHEN actual_result = 0 THEN 1  -- Draw
                ELSE 0  -- Home loss
            END) as points,
            COUNT(*) as matches_played
        FROM match_predictions
        WHERE league = :league 
        AND season = :season
        AND actual_result IS NOT NULL
        GROUP BY home_team
        
        UNION ALL
        
        SELECT 
            away_team as team,
            SUM(CASE 
                WHEN actual_result = 2 THEN 3  -- Away win
                WHEN actual_result = 0 THEN 1  -- Draw
                ELSE 0  -- Away loss
            END) as points,
            COUNT(*) as matches_played
        FROM match_predictions
        WHERE league = :league 
        AND season = :season
        AND actual_result IS NOT NULL
        GROUP BY away_team
    ),
    current_standings AS (
        SELECT 
            team,
            SUM(points) as points,
            SUM(matches_played) as matches_played
        FROM team_points
        GROUP BY team
    ),
    remaining_matches AS (
        SELECT 
            home_team,
            away_team,
            home_win_prob,
            draw_prob,
            away_win_prob
        FROM match_predictions
        WHERE league = :league 
        AND season = :season
        AND actual_result IS NULL
    )
    SELECT 
        cs.team,
        cs.points,
        cs.matches_played,
        rm.home_team,
        rm.away_team,
        rm.home_win_prob,
        rm.draw_prob,
        rm.away_win_prob
    FROM current_standings cs
    LEFT JOIN remaining_matches rm ON cs.team = rm.home_team OR cs.team = rm.away_team
    ORDER BY cs.points DESC, cs.team;
    """
    
    try:
        df = pd.read_sql_query(
            text(query),
            engine,
            params={'league': league, 'season': season}
        )
    except Exception as e:
        print(f"Error querying database: {str(e)}")
        return None
    
    if df.empty:
        print(f"No data found for {league} {season}")
        return None
    
    # Get unique teams and their current points
    teams = df['team'].unique()
    current_points = df.groupby('team')['points'].first()
    matches_played = df.groupby('team')['matches_played'].first()
    
    # Initialize results
    championship_counts = {team: 0 for team in teams}
    total_points = {team: [] for team in teams}
    
    # Run Monte Carlo simulations
    for sim in tqdm(range(n_simulations), desc=f"Running Monte Carlo simulations for {league} {season}"):
        # Initialize points for this simulation
        sim_points = current_points.copy()
        
        # Simulate each remaining match
        remaining_matches = df[df['home_team'].notna()].drop_duplicates(['home_team', 'away_team'])
        for _, match in remaining_matches.iterrows():
            # Generate random outcome based on probabilities
            try:
                probs = [match['home_win_prob'], match['draw_prob'], match['away_win_prob']]
                # Normalize probabilities to ensure they sum to 1
                probs = np.array(probs) / sum(probs)
                outcome = np.random.choice(
                    ['home_win', 'draw', 'away_win'],
                    p=probs
                )
                
                # Update points based on outcome
                if outcome == 'home_win':
                    sim_points[match['home_team']] += 3
                elif outcome == 'draw':
                    sim_points[match['home_team']] += 1
                    sim_points[match['away_team']] += 1
                else:  # away_win
                    sim_points[match['away_team']] += 3
            except Exception as e:
                print(f"Error in simulation for match {match['home_team']} vs {match['away_team']}: {str(e)}")
                continue
        
        # Record final points for this simulation
        for team in teams:
            total_points[team].append(sim_points[team])
        
        # Find champion(s) for this simulation
        max_points = sim_points.max()
        champions = sim_points[sim_points == max_points].index
        for champion in champions:
            championship_counts[champion] += 1
    
    # Calculate probabilities and prepare results
    results = []
    for team in teams:
        championship_prob = championship_counts[team] / n_simulations
        avg_points = np.mean(total_points[team])
        max_possible_points = current_points[team] + (38 - matches_played[team]) * 3
        
        # Calculate form based on last 5 matches
        form_query = """
        WITH last_5_matches AS (
            SELECT 
                CASE 
                    WHEN home_team = :team AND actual_result = 1 THEN 'W'
                    WHEN away_team = :team AND actual_result = 2 THEN 'W'
                    WHEN actual_result = 0 THEN 'D'
                    ELSE 'L'
                END as result
            FROM match_predictions
            WHERE (home_team = :team OR away_team = :team)
            AND league = :league 
            AND season = :season
            AND actual_result IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 5
        )
        SELECT STRING_AGG(result, '') as form
        FROM last_5_matches;
        """
        
        try:
            form_result = pd.read_sql_query(
                text(form_query),
                engine,
                params={'team': team, 'league': league, 'season': season}
            )
            form = form_result['form'].iloc[0] if not form_result.empty else ''
        except:
            form = ''
        
        results.append({
            'team_name': team,
            'championship_probability': championship_prob,
            'points': current_points[team],
            'form': form,
            'league': league,
            'season': season,
            'remaining_matches': 38 - matches_played[team],
            'max_possible_points': max_possible_points,
            'prediction_model': 'monte_carlo'
        })
    
    # Sort by championship probability
    results.sort(key=lambda x: x['championship_probability'], reverse=True)
    
    # Add rank
    for i, result in enumerate(results, 1):
        result['rank'] = i
    
    # Insert results into database
    try:
        with engine.begin() as conn:
            # Clear existing results for this league and season
            delete_query = """
            DELETE FROM championship_probabilities 
            WHERE league = :league AND season = :season
            """
            conn.execute(text(delete_query), {'league': league, 'season': season})
            
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
    except Exception as e:
        print(f"Error inserting results into database: {str(e)}")
        return None
    
    return results 