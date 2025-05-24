import pandas as pd
from utils_plot import *
import psycopg2
import os

DB_CONFIG = {
    'host': 'localhost',
    'port': '5431',
    'dbname': 'football_prediction',
    'user': 'postgres',
    'password': 'password'
}

def get_connection():
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['dbname'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    return conn

def fetch_features():
    conn = get_connection()
    query = "SELECT * FROM matches_featured"  # Tên bảng dữ liệu đã chứa features
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def map_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['home_score'] < row['away_score']:
        return 'Away Win'
    else:
        return 'Draw'

def main():
    os.makedirs("figures/after_features", exist_ok=True)
    df = fetch_features()

    # Nếu chưa có cột result, tạo thêm
    if 'result' not in df.columns:
        df['result'] = df.apply(map_result, axis=1)

    numerical_features = [
        'home_standing', 'away_standing', 'home_points', 'away_points', 'standing_diff', 'points_diff',
        'home_points_last_5', 'away_points_last_5',
        'home_goals_scored_last_5', 'home_goals_conceded_last_5',
        'away_goals_scored_last_5', 'away_goals_conceded_last_5',
        'home_goal_diff', 'away_goal_diff', 'home_home_win_rate', 'away_away_win_rate',
        'home_win_streak', 'away_win_streak', 'home_loss_streak', 'away_loss_streak'
    ]

    # Vẽ biểu đồ
    plot_label_distribution(df, 'result', "figures/after_features")
    plot_numerical_distributions(df, numerical_features, "figures/after_features")
    plot_correlation_matrix(df, numerical_features, "figures/after_features")

    # Giả sử bảng có cột season
    if 'season' in df.columns:
        plot_result_by_season(df, 'season', 'result', "figures/after_features")

    # Không có bảng standings riêng, nên bỏ phần thống kê theo đội ở đây
    # Nếu có bảng đội bóng riêng, có thể load và plot tương tự

    plot_missing_values(df, "figures/after_features")

if __name__ == "__main__":
    main()
