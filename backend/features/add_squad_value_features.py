import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy import text as sqlalchemy_text

def get_pg_engine():
    """Get PostgreSQL engine from utils"""
    from backend.features.utils import get_pg_engine
    return get_pg_engine()

def load_data():
    """Load necessary data from CSV files and database"""
    print("Loading data...")
    
    # Load player values data
    player_values_path = os.path.join('backend', 'data', 'all_player_values.csv')
    player_values = pd.read_csv(player_values_path)
    
    matches_path = os.path.join('backend', 'data', 'match_results.csv')
    matches = pd.read_csv(matches_path)
    
    return matches, player_values

def clean_player_name(name):
    """Clean and standardize player names for matching"""
    if not isinstance(name, str):
        return ""
    
    # Remove accents and special characters
    import unicodedata
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    
    # Convert to lowercase and remove non-alphabetic characters
    name = re.sub(r'[^a-zA-Z\s]', '', name).lower().strip()
    
    return name

def prepare_player_values_data(player_values):
    """Prepare and clean player values data"""
    print("Preparing player values data...")
    
    # Clean player names
    player_values['clean_name'] = player_values['player_name'].apply(clean_player_name)
    
    # Convert value string to numeric value
    def convert_value_to_eur(value_str):
        if not isinstance(value_str, str):
            return 0
        
        try:
            # Remove currency symbol and convert to lowercase
            value_str = value_str.lower().strip()
            
            # Handle "m" (million) and "k" (thousand) suffixes
            if 'm' in value_str:
                value = float(value_str.replace('m', '').replace('€', '').strip()) * 1000000
            elif 'k' in value_str:
                value = float(value_str.replace('k', '').replace('€', '').strip()) * 1000
            else:
                value = float(value_str.replace('€', '').strip())
            
            return value
        except:
            return 0
    
    player_values['market_value_eur'] = player_values['market_value'].apply(convert_value_to_eur)
    
    # Extract season from the data if available, or create a mapping
    if 'season' not in player_values.columns:
        # If season is not in the dataset, we need to map players to their values by season
        # This is a simplified approach, in reality this should be more sophisticated
        unique_seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', 
                          '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
        
        # Group by player and team, assuming values are chronological by player
        player_seasons = {}
        
        for player, team in zip(player_values['clean_name'], player_values['team']):
            if (player, team) not in player_seasons:
                player_seasons[(player, team)] = 0
            else:
                player_seasons[(player, team)] += 1
        
        # Assign seasons based on the order
        season_map = []
        for i, row in player_values.iterrows():
            player_key = (row['clean_name'], row['team'])
            idx = min(player_seasons[player_key], len(unique_seasons) - 1)
            season_map.append(unique_seasons[idx])
        
        player_values['season'] = season_map
    
    # Create a unique index for player-team-season
    player_values['player_team_season'] = player_values['clean_name'].astype(str) + "_" + \
                                          player_values['season'].astype(str)
        
    return player_values

def process_lineups(matches, player_values):
    """Process lineups and calculate squad values"""
    print("Processing lineups and calculating squad values...")
    
    # Create value lookup dict for faster access
    value_lookup = {}
    for _, row in player_values.iterrows():
        value_lookup[row['player_team_season']] = row['market_value_eur']
    
    
    # Function to extract players from lineup string
    def extract_players(lineup_str):
        if not isinstance(lineup_str, str):
            print(f"Warning: lineup_str is not a string, received: {lineup_str}")
            return []
        
        try:
            players_part = lineup_str.split(':', 1)[1].strip() if ':' in lineup_str else lineup_str
        except IndexError:
            print(f"Warning: Invalid lineup format, received: {lineup_str}")
            return []
        
        players = [player.strip() for player in players_part.split(',')]
        return [p for p in players if p]
    
    # Function to calculate squad value
    def calculate_squad_value(lineup_str, team, season):
        players = extract_players(lineup_str)
        total_value = 0
        season = season.split('-')[0]
        
        for player in players:
            # Try exact match with team and season
            key = f"{player.lower()}_{season}"
            if key in value_lookup:
                total_value += value_lookup[key]
            else:
                # Try fuzzy matching or alternative approach
                # This is a simplified approach, a more robust solution would use 
                # fuzzy matching or a more sophisticated player identification system
                for k in value_lookup:
                    if player in k and team in k and season in k:
                        total_value += value_lookup[k]
                        break
        
        return total_value
    
    # Calculate squad values
    home_values = []
    away_values = []
    
    for _, match in matches.iterrows():
        print("Caculate for match: " + match['home_team'] + " vs " + match['away_team'])
        home_value = calculate_squad_value(match['home_lineup'], match['home_team'], match['season'])
        away_value = calculate_squad_value(match['away_lineup'], match['away_team'], match['season'])
        
        print(home_value)
        home_values.append(home_value)
        away_values.append(away_value)
    
    # Add new columns to matches
    matches['home_squad_value'] = home_values
    matches['away_squad_value'] = away_values
    
    return matches

def update_database(matches_with_values):
    """Update matches_featured table with new squad value columns"""
    print("Updating database with squad values...")
    
    # Create engine
    engine = get_pg_engine()
    
    # Mở kết nối
    with engine.connect() as connection:
        # Add columns to the table if they don't exist
        connection.execute(sqlalchemy_text("""
        ALTER TABLE matches_featured 
        ADD COLUMN IF NOT EXISTS home_squad_value NUMERIC DEFAULT 0
        """))
        
        connection.execute(sqlalchemy_text("""
        ALTER TABLE matches_featured 
        ADD COLUMN IF NOT EXISTS away_squad_value NUMERIC DEFAULT 0
        """))
        
        # Update values for each match
        for _, match in matches_with_values.iterrows():
            update_query = sqlalchemy_text("""
                UPDATE matches_featured
                SET home_squad_value = :home_value, away_squad_value = :away_value
                WHERE match_date = :match_date AND home_team = :home_team AND away_team = :away_team
            """)
            connection.execute(update_query, {
                'home_value': match['home_squad_value'],
                'away_value': match['away_squad_value'],
                'match_date': match['date'],
                'home_team': match['home_team'],
                'away_team': match['away_team']
            })
        
        connection.commit()
    
    print(f"Updated {len(matches_with_values)} matches with squad values")

def main():
    """Main function to add squad value features"""
    # Load data
    matches, player_values = load_data()
    
    # Prepare player values data
    player_values = prepare_player_values_data(player_values)
    
    # Process lineups and calculate squad values
    matches_with_values = process_lineups(matches, player_values)
    
    # Update database
    update_database(matches_with_values)
    
    print("Squad value features added successfully!")

if __name__ == "__main__":
    main() 