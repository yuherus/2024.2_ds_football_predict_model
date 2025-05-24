import pandas as pd
from sqlalchemy import create_engine
from backend.features.utils import get_pg_engine

def load_data_from_db():
    """Load training data from PostgreSQL"""
    engine = get_pg_engine()
    query = """
    SELECT * FROM matches_featured 
    WHERE match_result IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    return df

def prepare_features_targets(df):
    """Prepare features and target variables"""
    # Define which columns to use as features
    feature_cols = [
        'home_points_last_5', 'away_points_last_5', 'home_standing', 'away_standing',
        'standing_diff', 'home_points', 'away_points', 'points_diff',
        'home_goals_scored_last_5', 'home_goals_conceded_last_5',
        'away_goals_scored_last_5', 'away_goals_conceded_last_5',
        'home_goal_diff', 'away_goal_diff', 'home_home_win_rate', 'away_away_win_rate',
        'home_win_streak', 'away_win_streak', 'home_loss_streak', 'away_loss_streak',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals_avg', 'h2h_away_goals_avg',
        'is_early_season', 'is_mid_season', 'is_late_season',
        'home_possession_avg', 'away_possession_avg', 'home_shots_on_target_avg',
        'away_shots_on_target_avg', 'home_corners_avg', 'away_corners_avg',
        'home_squad_value', 'away_squad_value'
    ]
    
    # Target variable
    target = 'match_result'
    
    # Drop rows with missing values in important features
    df_clean = df.dropna(subset=[target] + feature_cols)
    
    X = df_clean[feature_cols]
    y = df_clean[target]
    
    return X, y, df_clean

def split_data(X, y, test_size=0.2, val_size=0.1):
    """Split data into training, validation, and test sets"""
    from sklearn.model_selection import train_test_split
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: training vs validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_data_by_time(data, features, target):
    """
    Split data by time across seasons (time-based split):
    - Train: First 7 seasons
    - Validation: Next 2 seasons
    - Test: Final season
    
    Parameters:
    -----------
    data : DataFrame
        The full dataset with 'season' column
    features : list
        List of feature column names
    target : str
        Target variable column name
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, seasons_info)
    """
    # Make sure we have the season column
    if 'season' not in data.columns:
        raise ValueError("Dataset must contain a 'season' column for time-based splitting")
    
    # Sort seasons chronologically
    all_seasons = sorted(data['season'].unique())
    
    if len(all_seasons) < 10:
        raise ValueError(f"Need at least 10 seasons for time-based split, but only found {len(all_seasons)}")
    
    # Split seasons
    train_seasons = all_seasons[:7]  # First 7 seasons
    val_seasons = all_seasons[7:9]   # Next 2 seasons
    test_seasons = all_seasons[9:10] # Final season
    
    # Create masks
    train_mask = data['season'].isin(train_seasons)
    val_mask = data['season'].isin(val_seasons)
    test_mask = data['season'].isin(test_seasons)
    
    # Split the data
    X_train = data.loc[train_mask, features]
    y_train = data.loc[train_mask, target]
    
    X_val = data.loc[val_mask, features]
    y_val = data.loc[val_mask, target]
    
    X_test = data.loc[test_mask, features]
    y_test = data.loc[test_mask, target]
    
    # Create info dictionary about the split
    seasons_info = {
        'train_seasons': train_seasons,
        'val_seasons': val_seasons,
        'test_seasons': test_seasons,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, seasons_info

def split_data_by_time_and_league(data, features, target):
    """
    Split data by time across seasons for each league separately (time-based split):
    - Train: First 7 seasons of each league
    - Validation: Next 2 seasons of each league
    - Test: Final season of each league
    
    Parameters:
    -----------
    data : DataFrame
        The full dataset with 'season' and 'league' columns
    features : list
        List of feature column names
    target : str
        Target variable column name
        
    Returns:
    --------
    dict
        Dictionary with results for each league
    """
    # Make sure we have the required columns
    if 'season' not in data.columns or 'league' not in data.columns:
        raise ValueError("Dataset must contain 'season' and 'league' columns for time-based splitting by league")
    
    # Get unique leagues
    leagues = data['league'].unique()
    
    # Initialize results dictionary
    results = {}
    
    # Process each league separately
    for league in leagues:
        print(f"Processing {league}...")
        
        # Filter data for current league
        league_data = data[data['league'] == league]
        
        # Sort seasons chronologically
        all_seasons = sorted(league_data['season'].unique())
        
        if len(all_seasons) < 10:
            print(f"Warning: {league} has fewer than 10 seasons ({len(all_seasons)}). Skipping.")
            continue
        
        # Split seasons
        train_seasons = all_seasons[:7]  # First 7 seasons
        val_seasons = all_seasons[7:9]   # Next 2 seasons
        test_seasons = all_seasons[9:10] # Final season
        
        # Create masks
        train_mask = league_data['season'].isin(train_seasons)
        val_mask = league_data['season'].isin(val_seasons)
        test_mask = league_data['season'].isin(test_seasons)
        
        # Split the data
        X_train = league_data.loc[train_mask, features]
        y_train = league_data.loc[train_mask, target]
        
        X_val = league_data.loc[val_mask, features]
        y_val = league_data.loc[val_mask, target]
        
        X_test = league_data.loc[test_mask, features]
        y_test = league_data.loc[test_mask, target]
        
        # Create info dictionary about the split
        seasons_info = {
            'train_seasons': train_seasons,
            'val_seasons': val_seasons,
            'test_seasons': test_seasons,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        # Store results for this league
        results[league] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'seasons_info': seasons_info
        }
    
    return results

def prepare_lstm_sequences(df, seq_length=5, target_col='match_result'):
    """
    Prepare enhanced sequence data for LSTM model with more features and normalization
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing match data, sorted by date
    seq_length : int
        Number of previous matches to include in each sequence
    target_col : str
        Column name for the target variable
        
    Returns:
    --------
    tuple
        (X_sequences, y_targets, match_info) as numpy arrays and DataFrame
        match_info contains information about the next match for each sequence
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Assuming data is sorted by date and team
    sequences = []
    targets = []
    match_info = []  # Thêm danh sách để lưu thông tin về trận đấu tiếp theo
    
    # Define features to include in sequences
    sequence_features = [
        'home_points', 'away_points', 'home_score', 'away_score',
        'home_points_last_5', 'away_points_last_5', 'home_standing', 'away_standing',
        'standing_diff', 'points_diff', 'home_goal_diff', 'away_goal_diff',
        'home_win_streak', 'away_win_streak', 'home_loss_streak', 'away_loss_streak',
        'home_possession_avg', 'away_possession_avg', 'home_shots_on_target_avg',
        'away_shots_on_target_avg', 'home_corners_avg', 'away_corners_avg'
    ]
    
    # Check which features are actually available in the dataframe
    available_features = [f for f in sequence_features if f in df.columns]
    
    if len(available_features) < 4:
        raise ValueError(f"Not enough features available in dataframe. Found only: {available_features}")
    
    print(f"Using {len(available_features)} features for LSTM sequences: {available_features}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler on all available feature data
    feature_data = df[available_features].values
    scaler.fit(feature_data)
    
    teams = df['home_team'].unique()
    
    for team in teams:
        # Get all matches involving this team
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('match_date')
        
        if len(team_matches) < seq_length + 1:
            continue
            
        for i in range(len(team_matches) - seq_length):
            # Get sequence of matches
            seq = team_matches.iloc[i:i+seq_length]
            next_match = team_matches.iloc[i+seq_length]
            
            # Extract and normalize features
            seq_features = seq[available_features].values
            seq_features_scaled = scaler.transform(seq_features)
            
            # Add team position indicator (1 for home, 0 for away in each match)
            is_home = np.array([1 if match['home_team'] == team else 0 for _, match in seq.iterrows()]).reshape(-1, 1)
            
            # Combine normalized features with team position indicator
            seq_features_final = np.hstack([seq_features_scaled, is_home])
            
            # Extract target for next match
            if next_match['home_team'] == team:
                target_val = next_match[target_col]
                is_home_next = 1
            else:
                # Invert the target if team is away
                if next_match[target_col] == 0:  # home loss = away win
                    target_val = 2  
                elif next_match[target_col] == 2:  # home win = away loss
                    target_val = 0  
                else:  # Draw remains draw
                    target_val = 1
                is_home_next = 0
            
            # Add the sequence and target to our lists
            sequences.append(seq_features_final)
            targets.append(target_val)
            
            # Lưu thông tin về trận đấu tiếp theo
            match_info.append({
                'match_date': next_match['match_date'],
                'home_team': next_match['home_team'],
                'away_team': next_match['away_team'],
                'league': next_match['league'] if 'league' in next_match else None,
                'season': next_match['season'] if 'season' in next_match else None,
                'round': next_match['round'] if 'round' in next_match else None,
                'team_focus': team,
                'is_home': is_home_next
            })
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    # Convert match_info to DataFrame
    match_info_df = pd.DataFrame(match_info)
    
    print(f"Created {len(X)} sequences with shape {X.shape}")
    print(f"Match info DataFrame shape: {match_info_df.shape}")
    
    return X, y, match_info_df
