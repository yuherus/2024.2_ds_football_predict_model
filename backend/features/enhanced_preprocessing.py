import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from backend.features.utils import get_pg_engine
from backend.features.config import MATCHES_FEATURED_COLUMN_ORDER, FEATURED_FLOAT_COLS, FEATURED_INT_COLS


def fill_missing_rounds(df):
    """
    Điền các giá trị round còn thiếu dựa trên giá trị xung quanh.
    Sử dụng phương pháp tiền xử lý dựa trên ngày tháng và thứ tự trận đấu.
    """
    print("Điền các giá trị round còn thiếu...")
    
    # Xác định cột ngày tháng
    date_column = None
    if 'match_date' in df.columns:
        date_column = 'match_date'
    elif 'date' in df.columns:
        date_column = 'date'
    else:
        print("  Không tìm thấy cột ngày tháng để xử lý round")
        return df
    
    # Chuyển đổi date thành datetime nếu chưa phải
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Xử lý theo từng mùa giải và giải đấu
    for (season, league), group in df.groupby(['season', 'league']):
        mask = (df['season'] == season) & (df['league'] == league) & df['round'].isna()
        if not mask.any():
            continue
        
        # Sắp xếp theo ngày tháng
        sorted_group = group.sort_values(date_column)
        
        # Tìm các giá trị round đã biết
        known_rounds = sorted_group.dropna(subset=['round'])[[date_column, 'round']]
        
        if len(known_rounds) < 2:
            print(f"  Không đủ dữ liệu đã biết cho mùa {season}, giải {league}")
            continue
        
        # Nội suy round dựa trên ngày tháng
        rounds_interp = np.interp(
            x=pd.to_datetime(df.loc[mask, date_column]).map(lambda x: x.timestamp()),
            xp=pd.to_datetime(known_rounds[date_column]).map(lambda x: x.timestamp()),
            fp=known_rounds['round']
        )
        
        # Làm tròn và chuyển về integer
        rounded_values = np.round(rounds_interp).astype(int)
        df.loc[mask, 'round'] = rounded_values
        
        print(f"  Đã điền {mask.sum()} giá trị round cho mùa {season}, giải {league}")
    
    return df


def fill_team_stats_with_median(df):
    """
    Điền các thông số trận đấu còn thiếu bằng giá trị trung vị của đội bóng đó trong mùa giải.
    """
    print("Điền các thông số trận đấu còn thiếu bằng giá trị trung vị...")
    
    stats_columns = [
        'home_possession', 'away_possession', 
        'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_pass_completion', 'away_pass_completion',
        'home_red_cards', 'away_red_cards', 
        'home_yellow_cards', 'away_yellow_cards',
        'home_saves', 'away_saves', 
        'home_fouls', 'away_fouls',
        'home_corners', 'away_corners'
    ]
    
    # Xác định cột nào thực sự tồn tại trong DataFrame
    valid_stats_columns = [col for col in stats_columns if col in df.columns]
    
    # Xử lý theo từng mùa giải
    for season, season_df in df.groupby('season'):
        print(f"  Xử lý mùa giải {season}...")
        
        # Lấy các thông số trung vị cho mỗi đội sân nhà
        home_team_medians = {}
        for team, team_df in season_df.groupby('home_team'):
            team_stats = {}
            for col in valid_stats_columns:
                if col.startswith('home_'):
                    # Chỉ lấy giá trị khác NA để tính trung vị
                    valid_values = team_df[col].dropna()
                    if len(valid_values) > 0:
                        team_stats[col] = valid_values.median()
            home_team_medians[team] = team_stats
        
        # Lấy các thông số trung vị cho mỗi đội sân khách
        away_team_medians = {}
        for team, team_df in season_df.groupby('away_team'):
            team_stats = {}
            for col in valid_stats_columns:
                if col.startswith('away_'):
                    valid_values = team_df[col].dropna()
                    if len(valid_values) > 0:
                        team_stats[col] = valid_values.median()
            away_team_medians[team] = team_stats
        
        # Điền các giá trị còn thiếu
        for index, row in season_df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            for col in valid_stats_columns:
                if pd.isna(df.loc[index, col]):
                    if col.startswith('home_') and home_team in home_team_medians and col in home_team_medians[home_team]:
                        df.loc[index, col] = home_team_medians[home_team][col]
                    elif col.startswith('away_') and away_team in away_team_medians and col in away_team_medians[away_team]:
                        df.loc[index, col] = away_team_medians[away_team][col]
    
    # Tính giá trị trung vị theo giải đấu và mùa giải cho các giá trị vẫn còn thiếu
    for (season, league), group in df.groupby(['season', 'league']):
        print(f"  Xử lý các giá trị còn thiếu cho mùa {season}, giải {league}...")
        
        # Tính giá trị trung vị cho từng cột trong giải đấu và mùa giải này
        league_season_medians = {}
        for col in valid_stats_columns:
            valid_values = group[col].dropna()
            if len(valid_values) > 0:
                league_season_medians[col] = valid_values.median()
        
        # Áp dụng giá trị trung vị của giải đấu và mùa giải cho các giá trị còn thiếu
        for index, row in group.iterrows():
            for col in valid_stats_columns:
                if pd.isna(df.loc[index, col]) and col in league_season_medians:
                    df.loc[index, col] = league_season_medians[col]
    
    # Điền các giá trị vẫn còn thiếu bằng 0
    for col in valid_stats_columns:
        if df[col].isna().any():
            print(f"  Điền các giá trị còn thiếu trong cột {col} bằng 0")
            df[col] = df[col].fillna(0)
    
    # Đảm bảo tất cả các giá trị là số nguyên
    for col in valid_stats_columns:
        df[col] = np.round(df[col]).astype(int)
    
    return df


def normalize_features(df):
    """
    Chuẩn hóa các đặc trưng số trong DataFrame.
    Sử dụng RobustScaler để giảm ảnh hưởng của outliers.
    """
    print("Chuẩn hóa các đặc trưng số...")
    
    numeric_cols = []
    
    # Xác định các cột số
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != 'match_result' and 'id' not in col.lower():
            numeric_cols.append(col)
    
    # Dùng RobustScaler để chuẩn hóa
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Lưu scaler để sử dụng sau này
    df.attrs['scaler'] = scaler
    df.attrs['numeric_cols'] = numeric_cols
    
    return df


def detect_and_handle_outliers(df):
    """
    Phát hiện và xử lý các giá trị ngoại lai (outliers).
    """
    print("Phát hiện và xử lý outliers...")
    
    # Chọn các cột số để phát hiện outliers
    numeric_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) 
                   and 'id' not in col.lower()
                   and col not in ['match_result', 'season', 'round']]
    
    # Xử lý outliers dựa trên IQR
    for col in numeric_cols:
        # Điền NaN trước khi tính toán
        df[col] = df[col].fillna(0)
        
        # Thay thế các giá trị inf
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Thay thế outliers bằng bound values
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Đảm bảo không có giá trị NaN hoặc inf trước khi chuyển về integer
        df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
        df[col] = np.round(df[col]).astype(int)
    
    print(f"  Đã xử lý outliers trong {len(numeric_cols)} cột")
    
    return df


def convert_all_numeric_to_int(df):
    """
    Chuyển tất cả các cột số về dạng integer.
    """
    print("Chuyển đổi tất cả các cột số về dạng integer...")
    
    # Danh sách các cột không nên chuyển về integer
    exclude_cols = ['match_date', 'date']  # Các cột datetime
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # Nếu là cột số và không phải string
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Xử lý các giá trị đặc biệt trước khi chuyển đổi
                df[col] = df[col].fillna(0)  # Điền NaN bằng 0
                df[col] = df[col].replace([np.inf, -np.inf], 0)  # Thay thế inf bằng 0
                
                # Đảm bảo không có giá trị âm cho các cột thống kê cụ thể
                if any(keyword in col.lower() for keyword in ['shots', 'saves', 'corners', 'cards']):
                    df[col] = df[col].abs()
                
                # Làm tròn và chuyển về integer
                df[col] = np.round(df[col]).astype(int)
                print(f"  Đã chuyển {col} sang integer")
            except Exception as e:
                print(f"  Không thể chuyển {col} sang integer: {e}")
                # Nếu không thể chuyển, ít nhất cũng điền NaN
                df[col] = df[col].fillna(0)
    
    return df


def enhanced_preprocessing_pipeline(matches_df=None):
    """
    Pipeline tiền xử lý dữ liệu nâng cao.
    """
    if matches_df is None:
        try:
            matches_df = pd.read_csv('backend/data/match_results.csv')
        except Exception as e2:
            print(f"Lỗi khi load dữ liệu từ file CSV: {e2}")
            return None
    
    # Backup dữ liệu gốc
    original_df = matches_df.copy()
    
    # Chuẩn hóa tên cột
    if 'date' in matches_df.columns and 'match_date' not in matches_df.columns:
        matches_df.rename(columns={'date': 'match_date'}, inplace=True)
    
    # 1. Điền các giá trị round còn thiếu
    matches_df = fill_missing_rounds(matches_df)
    
    # 2. Phát hiện và xử lý outliers
    matches_df = detect_and_handle_outliers(matches_df)
    
    # 3. Điền các thông số trận đấu còn thiếu bằng giá trị trung vị
    matches_df = fill_team_stats_with_median(matches_df)
    
    # 4. Thêm cột match_result nếu có home_score và away_score và chưa có match_result
    if 'match_result' not in matches_df.columns and 'home_score' in matches_df.columns and 'away_score' in matches_df.columns:
        # Chuyển đổi pandas Series thành numpy arrays để tránh lỗi
        home_scores = matches_df['home_score'].to_numpy()
        away_scores = matches_df['away_score'].to_numpy()
        
        # Tạo mảng điều kiện sử dụng numpy arrays
        conditions = [
            home_scores < away_scores,  # home_loss
            home_scores == away_scores,  # draw
            home_scores > away_scores  # home_win
        ]
        choices = [0, 1, 2]  # home_loss, draw, home_win
        
        # Sử dụng np.select với numpy arrays
        result_values = np.select(conditions, choices, default=0)
        
        # Chuyển kết quả thành integer
        matches_df['match_result'] = result_values.astype(int)
    
    # 5. Chuyển tất cả các cột số về dạng integer
    matches_df = convert_all_numeric_to_int(matches_df)
    
    # 6. Chuẩn hóa các đặc trưng số (optional - comment out nếu không cần)
    # matches_df = normalize_features(matches_df)
    
    return matches_df


def save_preprocessed_data(df, output_path=None):
    """
    Lưu dữ liệu đã tiền xử lý với đảm bảo kiểu dữ liệu integer.
    """
    if output_path is None:
        output_path = 'backend/data/preprocessed_matches.csv'
    
    try:
        # Tạo một bản sao để xử lý trước khi lưu
        df_to_save = df.copy()
        
        # Đảm bảo tất cả các cột số là integer trước khi lưu
        exclude_cols = ['match_date', 'date', 'home_team', 'away_team', 'league']  # Các cột không phải số
        
        for col in df_to_save.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_to_save[col]):
                try:
                    # Đảm bảo không có NaN
                    df_to_save[col] = df_to_save[col].fillna(0)
                    # Chuyển về integer
                    df_to_save[col] = df_to_save[col].astype(int)
                except Exception as e:
                    print(f"Lỗi khi chuyển đổi cột {col}: {e}")
        
        # Lưu file
        df_to_save.to_csv(output_path, index=False)
        
        print(f"Đã lưu dữ liệu đã tiền xử lý vào {output_path}")
        
        # Kiểm tra kiểu dữ liệu sau khi lưu
        try:
            test_df = pd.read_csv(output_path, nrows=5)
            print("Kiểm tra kiểu dữ liệu sau khi lưu:")
            for col in test_df.columns:
                if pd.api.types.is_numeric_dtype(test_df[col]):
                    print(f"  {col}: {test_df[col].dtype}")
        except Exception as e:
            print(f"Lỗi khi kiểm tra file: {e}")
        
        # Lưu thông tin bổ sung
        import pickle
        
        # Lưu scaler nếu có
        if 'scaler' in df.attrs:
            with open(output_path.replace('.csv', '_scaler.pkl'), 'wb') as f:
                pickle.dump(df.attrs['scaler'], f)
    
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")


def main():
    """
    Hàm chính để chạy pipeline tiền xử lý dữ liệu nâng cao.
    """
    print("Bắt đầu pipeline tiền xử lý dữ liệu nâng cao...")
    
    # Load và tiền xử lý dữ liệu
    preprocessed_df = enhanced_preprocessing_pipeline()
    
    if preprocessed_df is not None:
        # Lưu dữ liệu đã tiền xử lý
        save_preprocessed_data(preprocessed_df)
        
        # Thêm vào database nếu cần
        try:
            engine = get_pg_engine()
            
            # Tạo một bản sao để tránh ảnh hưởng đến dữ liệu gốc
            df_to_db = preprocessed_df.copy()
            
            # Đảm bảo tất cả các cột số là integer trước khi lưu vào database
            exclude_cols = ['match_date', 'date', 'home_team', 'away_team', 'league']
            
            for col in df_to_db.columns:
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_to_db[col]):
                    try:
                        df_to_db[col] = df_to_db[col].fillna(0).astype(int)
                    except Exception as e:
                        print(f"Lỗi khi chuyển đổi cột {col} cho database: {e}")
            
            df_to_db.to_sql('matches_preprocessed', engine, if_exists='replace', index=False)
            print("Đã lưu dữ liệu đã tiền xử lý vào database")
        except Exception as e:
            print(f"Lỗi khi lưu vào database: {e}")
    
    print("Hoàn thành pipeline tiền xử lý dữ liệu nâng cao!")


if __name__ == "__main__":
    main()