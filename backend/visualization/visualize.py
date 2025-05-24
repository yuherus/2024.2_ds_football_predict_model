import pandas as pd
from utils_plot import *
import os

def get_match_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['home_score'] < row['away_score']:
        return 'Away Win'
    else:
        return 'Draw'

def main():
    os.makedirs("figures/before_features", exist_ok=True)

    match_results = pd.read_csv("data/match_results.csv")
    standings = pd.read_csv("data/standings.csv")
    player_values = pd.read_csv("data/all_player_values.csv")

    # Thêm cột kết quả trận đấu
    match_results['result'] = match_results.apply(get_match_result, axis=1)

    numerical_features = [
        'home_score', 'away_score', 'home_possession', 'away_possession',
        'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
        'home_pass_completion', 'away_pass_completion', 'home_red_cards', 'away_red_cards',
        'home_yellow_cards', 'away_yellow_cards', 'home_fouls', 'away_fouls'
    ]

    # Vẽ biểu đồ
    plot_label_distribution(match_results, 'result', "figures/before_features")
    plot_numerical_distributions(match_results, numerical_features, "figures/before_features")
    plot_correlation_matrix(match_results, numerical_features, "figures/before_features")
    plot_result_by_season(match_results, 'season', 'result', "figures/before_features")

    team_stats_cols = ['wins', 'draws', 'losses', 'goals_for', 'goals_against', 'points']
    plot_team_stats(standings, 'club', team_stats_cols, "figures/before_features")

    plot_squad_value_distribution(player_values, 'market_value', "figures/before_features")

    plot_missing_values(match_results, "figures/before_features")

if __name__ == "__main__":
    main()
