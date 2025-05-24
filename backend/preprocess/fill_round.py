import pandas as pd
from collections import Counter
input_path = "backend/data/raw/match_raw.csv"
output_path = "backend/data/processed/match_results.csv"

def fill_round_by_near_dates(df, days_window=7):
    df['date'] = pd.to_datetime(df['date'])
    df_filled = df.copy()
    missing_idx = df_filled[df_filled['round'].isna()].index
    
    for idx in missing_idx:
        row = df_filled.loc[idx]
        league = row['league']
        season = row['season']
        match_date = row['date']
        group = df_filled[(df_filled['league'] == league) & (df_filled['season'] == season)]
        nearby = group[
            (group['round'].notna()) &
            (group['date'] >= match_date - pd.Timedelta(days=days_window)) &
            (group['date'] <= match_date + pd.Timedelta(days=days_window))
        ]
        if not nearby.empty:
            most_common_round = Counter(nearby['round']).most_common(1)[0][0]
            df_filled.at[idx, 'round'] = most_common_round
            
    return df_filled

df = pd.read_csv(input_path)
df_filled = fill_round_by_near_dates(df)
df_filled.to_csv(output_path, index=False)

