# backend/models/model_training.py
from backend.models.data_preparation import (
    load_data_from_db, prepare_features_targets, 
    split_data, prepare_lstm_sequences,
    split_data_by_time_and_league
)
from backend.models.xgboost_model import train_xgboost, evaluate_xgboost
from backend.models.lstm_model import create_lstm_model, train_lstm, evaluate_lstm
from backend.models.model_storage import save_xgboost_model, save_lstm_model, save_model_metadata, save_match_predictions
from backend.models.model_evaluation import calculate_metrics, compare_models, plot_confusion_matrix, plot_roc_curves, plot_learning_curves
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

def train_league_season_models(data, league, season, save_dir="models"):
    """
    Train models for a specific league and season
    
    Parameters:
    -----------
    data : DataFrame
        The full dataset
    league : str
        League name (e.g., 'premierleague')
    season : str
        Season (e.g., '2019-2020')
    save_dir : str
        Directory to save models
        
    Returns:
    --------
    dict
        Results of model training
    """
    print(f"\n=== Training models for {league} {season} ===")
    
    # Prepare directory for this league-season
    league_season_dir = os.path.join(save_dir, f"{league}_{season}")
    os.makedirs(league_season_dir, exist_ok=True)
    
    # Filter data for current league and season
    current_data = data[(data['league'] == league) & (data['season'] == season)]
    
    if len(current_data) < 100:
        print(f"Too few matches ({len(current_data)}) for {league} {season}. Skipping.")
        return None
    
    print(f"Using {len(current_data)} matches for training")
    
    # Historical data from previous seasons of this league (for sequence models)
    historical_data = data[(data['league'] == league) & 
                          (data['season'] < season)].sort_values('match_date')
    
    print(f"Found {len(historical_data)} historical matches for {league} before {season}")
    
    # Prepare data for XGBoost
    X, y, df_clean = prepare_features_targets(current_data)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)
    
    # ==== XGBoost Training ====
    print(f"\nTraining XGBoost for {league} {season}...")
    xgb_start_time = time.time()
    
    # Sử dụng GridSearchCV để tìm tham số tốt nhất
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, use_grid_search=True)
    
    xgb_training_time = time.time() - xgb_start_time
    print(f"XGBoost training completed in {xgb_training_time:.2f} seconds")
    
    # Evaluate XGBoost
    print(f"\nEvaluating XGBoost for {league} {season}...")
    xgb_metrics = evaluate_xgboost(xgb_model, X_test, y_test)
    
    # Save XGBoost model and plots
    xgb_model_path = os.path.join(league_season_dir, "xgboost_model.pkl")
    save_xgboost_model(xgb_model, xgb_model_path)
    
    # Plot confusion matrix
    cm = xgb_metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, save_path=os.path.join(league_season_dir, "xgb_confusion_matrix.png"))
    
    # ==== LSTM Training ====
    print(f"\nTraining LSTM for {league} {season}...")
    
    # Combine historical and current data for sequence preparation
    combined_data = pd.concat([historical_data, current_data]).sort_values('match_date')
    
    # Prepare sequences using historical + current data
    X_seq, y_seq, match_info_seq = prepare_lstm_sequences(combined_data)
    
    if len(X_seq) < 50:
        print(f"Not enough sequence data for LSTM training for {league} {season}. Skipping LSTM.")
        lstm_model = None
        lstm_metrics = None
        lstm_history = None
    else:
        # Split into train (historical), val+test (current season)
        # Find cutoff index between historical and current
        historical_end_date = historical_data['match_date'].max() if not historical_data.empty else pd.Timestamp.min
        
        # Split train (historical) and test (current season)
        train_seq_mask = []
        current_seq_mask = []
        
        for i in range(len(X_seq)):
            # Check if sequence contains only matches before current season
            if i < len(combined_data) - len(current_data):
                train_seq_mask.append(True)
                current_seq_mask.append(False)
            else:
                train_seq_mask.append(False)
                current_seq_mask.append(True)
        
        X_train_seq = X_seq[train_seq_mask]
        y_train_seq = y_seq[train_seq_mask]
        match_info_train = match_info_seq[train_seq_mask].reset_index(drop=True)
        
        X_current_seq = X_seq[current_seq_mask]
        y_current_seq = y_seq[current_seq_mask]
        match_info_current = match_info_seq[current_seq_mask].reset_index(drop=True)
        
        # Further split current season data into val and test
        from sklearn.model_selection import train_test_split
        X_val_seq, X_test_seq, y_val_seq, y_test_seq, val_indices, test_indices = train_test_split(
            X_current_seq, y_current_seq, np.arange(len(X_current_seq)), 
            test_size=0.5, random_state=42, 
            stratify=y_current_seq if len(set(y_current_seq)) > 1 else None
        )
        
        # Split match info accordingly
        match_info_val = match_info_current.iloc[val_indices].reset_index(drop=True)
        match_info_test = match_info_current.iloc[test_indices].reset_index(drop=True)
        
        print(f"LSTM training data: {X_train_seq.shape[0]} sequences")
        print(f"LSTM validation data: {X_val_seq.shape[0]} sequences")
        print(f"LSTM test data: {X_test_seq.shape[0]} sequences")
        
        # Skip LSTM if not enough data in any split
        if len(X_train_seq) < 30 or len(X_val_seq) < 10 or len(X_test_seq) < 10:
            print(f"Insufficient data for LSTM training/validation/testing. Skipping LSTM for {league} {season}.")
            lstm_model = None
            lstm_metrics = None
            lstm_history = None
        else:
            lstm_start_time = time.time()
            
            # Create and train LSTM model
            seq_length, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
            
            lstm_model, lstm_history = train_lstm(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                batch_size=64,
                epochs=100
            )
            
            lstm_training_time = time.time() - lstm_start_time
            print(f"LSTM training completed in {lstm_training_time:.2f} seconds")
            
            # Evaluate LSTM
            print(f"\nEvaluating LSTM for {league} {season}...")
            lstm_metrics = evaluate_lstm(lstm_model, X_test_seq, y_test_seq)
            
            # Save LSTM model and plots
            lstm_model_path = os.path.join(league_season_dir, "lstm_model.h5")
            if hasattr(lstm_model, 'save'):
                lstm_model.save(lstm_model_path)
            
            # Plot learning curves with more metrics
            plot_learning_curves(
                lstm_history, 
                metrics=['accuracy', 'loss'],
                save_path=os.path.join(league_season_dir, "lstm_learning_curves.png")
            )
            
            # Plot additional metrics if available
            if 'f1_score' in lstm_metrics and 'roc_auc' in lstm_metrics:
                # Lưu các metrics bổ sung vào file
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'F1 Score', 'ROC AUC'],
                    'Value': [
                        lstm_metrics['accuracy'],
                        lstm_metrics['f1_score'],
                        lstm_metrics['roc_auc'] if lstm_metrics['roc_auc'] is not None else np.nan
                    ]
                })
                metrics_df.to_csv(os.path.join(league_season_dir, "lstm_metrics.csv"), index=False)
                
                # Vẽ biểu đồ các metrics
                plt.figure(figsize=(10, 6))
                plt.bar(metrics_df['Metric'], metrics_df['Value'])
                plt.title(f'LSTM Model Metrics for {league} {season}')
                plt.ylim(0, 1)
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(os.path.join(league_season_dir, "lstm_metrics_chart.png"))
                plt.close()
            
            # Nếu có huấn luyện LSTM, lưu dự đoán LSTM
            if lstm_metrics is not None:
                # Với mô hình LSTM, chúng ta cần dữ liệu chuỗi từ tập test
                # Giả sử X_test_seq và y_test_seq đã được chuẩn bị
                if 'X_test_seq' in locals() and 'y_test_seq' in locals() and 'match_info_test' in locals():
                    save_test_predictions(
                        match_info_test, 
                        X_test_seq,
                        y_test_seq, 
                        lstm_model, 
                        model_type="lstm"
                    )
    
    # Save model comparison if both models were trained
    if lstm_metrics is not None:
        model_results = {
            'XGBoost': xgb_metrics,
            'LSTM': lstm_metrics
        }
        
        comparison_df = compare_models(
            model_results,
            save_path=os.path.join(league_season_dir, "model_comparison.png")
        )
        
        # Lưu so sánh vào CSV
        comparison_df.to_csv(os.path.join(league_season_dir, "model_comparison.csv"), index=False)
    
    return {
        'league': league,
        'season': season,
        'xgboost': {
            'model': xgb_model,
            'metrics': xgb_metrics,
            'path': xgb_model_path if 'xgb_model_path' in locals() else None
        },
        'lstm': {
            'model': lstm_model,
            'metrics': lstm_metrics,
            'history': lstm_history,
            'path': lstm_model_path if 'lstm_model_path' in locals() and lstm_model is not None else None
        }
    }

def prepare_prediction_dataframe(test_data, X_test, y_pred_proba, y_pred, y_test=None):
    """
    Chuẩn bị DataFrame để lưu kết quả dự đoán vào bảng match_predictions
    
    Parameters:
    -----------
    test_data : DataFrame
        DataFrame gốc chứa dữ liệu test
    X_test : DataFrame hoặc ndarray
        Các đặc trưng của tập test
    y_pred_proba : ndarray
        Ma trận xác suất dự đoán (số hàng = số lượng mẫu, số cột = 3 cho kết quả)
    y_pred : ndarray
        Kết quả dự đoán (0=away win, 1=draw, 2=home win)
    y_test : ndarray, optional
        Kết quả thực tế nếu có
    
    Returns:
    --------
    DataFrame
        DataFrame đã chuẩn bị với thông tin cho việc lưu dự đoán
    """
    # Tạo DataFrame mới cho các dự đoán
    predictions_df = pd.DataFrame()
    
    # Đảm bảo y_pred_proba và test_data có cùng số lượng mẫu
    n_samples = len(y_pred_proba)
    
    # Xử lý khác nhau tùy thuộc vào loại X_test
    if isinstance(X_test, pd.DataFrame):
        # Nếu X_test là DataFrame, sử dụng index để truy xuất dữ liệu gốc
        test_indices = X_test.index
        
        # Đảm bảo chỉ sử dụng số lượng dòng tương đương với số lượng dự đoán
        if len(test_indices) > n_samples:
            print(f"Cảnh báo: Số lượng test_indices ({len(test_indices)}) lớn hơn số lượng dự đoán ({n_samples})")
            test_indices = test_indices[:n_samples]
        elif len(test_indices) < n_samples:
            print(f"Cảnh báo: Số lượng test_indices ({len(test_indices)}) nhỏ hơn số lượng dự đoán ({n_samples})")
            n_samples = len(test_indices)
            y_pred_proba = y_pred_proba[:n_samples]
            if y_test is not None:
                y_test = y_test[:n_samples]
        
        # Lấy thông tin cần thiết từ dữ liệu gốc
        predictions_df['match_date'] = test_data.loc[test_indices, 'match_date'].values
        predictions_df['round'] = test_data.loc[test_indices, 'round'].values
        predictions_df['home_team'] = test_data.loc[test_indices, 'home_team'].values
        predictions_df['away_team'] = test_data.loc[test_indices, 'away_team'].values
        
        # Nếu có các thông tin khác
        if 'league' in test_data.columns:
            predictions_df['league'] = test_data.loc[test_indices, 'league'].values
        if 'season' in test_data.columns:
            predictions_df['season'] = test_data.loc[test_indices, 'season'].values
        if 'venue' in test_data.columns:
            predictions_df['venue'] = test_data.loc[test_indices, 'venue'].values
    else:
        # Nếu X_test là ndarray, giả định rằng thứ tự hàng trong test_data khớp với X_test
        # Đảm bảo test_data có đủ dòng
        if len(test_data) < n_samples:
            print(f"Cảnh báo: test_data ({len(test_data)} hàng) có ít hàng hơn dự đoán ({n_samples} hàng)")
            n_samples = min(n_samples, len(test_data))
            y_pred_proba = y_pred_proba[:n_samples]
            if y_test is not None:
                y_test = y_test[:n_samples]
        elif len(test_data) > n_samples:
            print(f"Cảnh báo: test_data ({len(test_data)} hàng) có nhiều hàng hơn dự đoán ({n_samples} hàng)")
        
        # Lấy n_samples đầu tiên từ test_data
        test_subset = test_data.iloc[:n_samples]
        
        # Lấy thông tin cần thiết
        predictions_df['match_date'] = test_subset['match_date'].values
        predictions_df['round'] = test_subset['round'].values
        predictions_df['home_team'] = test_subset['home_team'].values
        predictions_df['away_team'] = test_subset['away_team'].values
        
        # Nếu có các thông tin khác
        if 'league' in test_data.columns:
            predictions_df['league'] = test_subset['league'].values
        if 'season' in test_data.columns:
            predictions_df['season'] = test_subset['season'].values
        if 'venue' in test_data.columns:
            predictions_df['venue'] = test_subset['venue'].values
    
    # Thêm kết quả dự đoán - chuyển đổi mảng numpy thành danh sách python
    probabilities_list = []
    for i in range(len(predictions_df)):
        if i < len(y_pred_proba):
            probabilities_list.append(y_pred_proba[i].tolist())  # Chuyển đổi ndarray thành list
        else:
            # Trường hợp thiếu dự đoán
            probabilities_list.append([0.0, 0.0, 0.0])  # Giá trị mặc định
    
    predictions_df['probabilities'] = probabilities_list
    
    # Thêm kết quả thực tế nếu có
    if y_test is not None:
        if len(y_test) >= len(predictions_df):
            predictions_df['actual_result'] = y_test[:len(predictions_df)]
        else:
            # Nếu không đủ kết quả thực tế, điền None cho phần thiếu
            actual_results = list(y_test) + [None] * (len(predictions_df) - len(y_test))
            predictions_df['actual_result'] = actual_results
    
    return predictions_df

def save_test_predictions(test_data, X_test, y_test, model, model_type="xgboost"):
    """
    Lưu dự đoán cho tập test vào database
    
    Parameters:
    -----------
    test_data : DataFrame
        DataFrame gốc chứa dữ liệu test hoặc DataFrame match_info cho LSTM
    X_test : DataFrame hoặc ndarray
        Các đặc trưng của tập test
    y_test : ndarray
        Kết quả thực tế
    model : model object
        Mô hình đã được huấn luyện
    model_type : str
        Loại mô hình ("xgboost" hoặc "lstm")
    
    Returns:
    --------
    int
        Số lượng dự đoán đã lưu
    """
    print(f"Lưu dự đoán {model_type} cho {len(X_test)} mẫu...")
    
    # Thực hiện dự đoán
    if model_type.lower() == "xgboost":
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Tạo DataFrame dự đoán cho XGBoost
        predictions_df = prepare_prediction_dataframe(test_data, X_test, y_pred_proba, y_pred, y_test)
        
    elif model_type.lower() == "lstm":
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Đối với LSTM, test_data nên là match_info_df
        if isinstance(test_data, pd.DataFrame) and 'match_date' in test_data.columns and 'home_team' in test_data.columns:
            match_info_df = test_data
            print(f"Sử dụng match_info_df với {len(match_info_df)} trận đấu cho dự đoán LSTM")
            
            # Đảm bảo số lượng dự đoán khớp với số lượng trận đấu
            if len(y_pred_proba) > len(match_info_df):
                print(f"Cảnh báo: Số lượng dự đoán ({len(y_pred_proba)}) nhiều hơn số lượng trận đấu ({len(match_info_df)})")
                y_pred_proba = y_pred_proba[:len(match_info_df)]
                y_pred = y_pred[:len(match_info_df)]
            elif len(y_pred_proba) < len(match_info_df):
                print(f"Cảnh báo: Số lượng dự đoán ({len(y_pred_proba)}) ít hơn số lượng trận đấu ({len(match_info_df)})")
                match_info_df = match_info_df.iloc[:len(y_pred_proba)]
            
            # Tạo DataFrame dự đoán từ match_info_df
            predictions_df = pd.DataFrame()
            predictions_df['match_date'] = match_info_df['match_date']
            predictions_df['home_team'] = match_info_df['home_team']
            predictions_df['away_team'] = match_info_df['away_team']
            
            if 'round' in match_info_df:
                predictions_df['round'] = match_info_df['round']
            if 'league' in match_info_df:
                predictions_df['league'] = match_info_df['league']
            if 'season' in match_info_df:
                predictions_df['season'] = match_info_df['season']
            
            # Thêm kết quả dự đoán
            probabilities_list = [probs.tolist() for probs in y_pred_proba]
            predictions_df['probabilities'] = probabilities_list
            
            # Thêm kết quả thực tế nếu có
            if y_test is not None and len(y_test) >= len(predictions_df):
                predictions_df['actual_result'] = y_test[:len(predictions_df)]
        else:
            # Fallback nếu test_data không phải là match_info_df
            print("Cảnh báo: test_data không chứa thông tin trận đấu cần thiết cho LSTM")
            predictions_df = prepare_prediction_dataframe(test_data, 
                                                        np.arange(len(y_pred)).reshape(-1, 1), 
                                                        y_pred_proba, y_pred, y_test)
    else:
        raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")
    
    # Lưu vào database
    return save_match_predictions(predictions_df, model_name=model_type)

def train_all_models(save_dir="models"):
    """
    Training pipeline for both XGBoost and LSTM models using time-based split
    for each league separately
    """
    # Create directory for models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data from database...")
    data = load_data_from_db()
    
    if data.empty:
        print("No data found in database. Please run the data processing pipeline first.")
        return
    
    print(f"Loaded {len(data)} matches from database")
    
    # Run the squad value feature creation if needed
    if 'home_squad_value' not in data.columns or 'away_squad_value' not in data.columns:
        print("Squad value features not found in the data. Creating them now...")
        import sys
        import importlib.util
        
        # Dynamically import and run the squad value creation script
        spec = importlib.util.spec_from_file_location(
            "add_squad_value_features", 
            os.path.join("backend", "features", "add_squad_value_features.py")
        )
        squad_value_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(squad_value_module)
        
        # Run the main function
        squad_value_module.main()
        
        # Reload data with new features
        data = load_data_from_db()
        print("Data reloaded with squad value features")
    
    # Prepare features and target
    _, _, df_clean = prepare_features_targets(data)
    
    # Get feature columns
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
    target_col = 'match_result'
    
    # Scale features before split
    scaler = StandardScaler()
    data_scaled = df_clean.copy()
    data_scaled[feature_cols] = scaler.fit_transform(data_scaled[feature_cols])
    
    # Split data by time for each league
    print("Splitting data by time for each league...")
    league_data = split_data_by_time_and_league(data_scaled, feature_cols, target_col)
    
    # Train models for each league
    league_results = {}
    league_summaries = []
    
    for league, split_data in league_data.items():
        print(f"\n==== Training models for {league} ====")
        league_dir = os.path.join(save_dir, league)
        os.makedirs(league_dir, exist_ok=True)
        
        # Save split information
        split_info = split_data['seasons_info']
        pd.DataFrame([split_info]).to_csv(os.path.join(league_dir, "split_info.csv"), index=False)
        
        print(f"Train set: {split_info['train_size']} samples")
        print(f"Validation set: {split_info['val_size']} samples")
        print(f"Test set: {split_info['test_size']} samples")
        
        # Train XGBoost
        print(f"\nTraining XGBoost model for {league}...")
        xgb_start_time = time.time()
        
        # Sử dụng GridSearchCV để tìm tham số tốt nhất
        xgb_model = train_xgboost(
            split_data['X_train'], split_data['y_train'],
            split_data['X_val'], split_data['y_val'],
            use_grid_search=True
        )
        
        xgb_training_time = time.time() - xgb_start_time
        print(f"XGBoost training completed in {xgb_training_time:.2f} seconds")
        
        # Evaluate XGBoost
        print(f"\nEvaluating XGBoost model for {league}...")
        xgb_metrics = evaluate_xgboost(xgb_model, split_data['X_test'], split_data['y_test'])
        
        # Save XGBoost model and plots
        xgb_model_path = os.path.join(league_dir, "xgboost_model.pkl")
        save_xgboost_model(xgb_model, xgb_model_path)
        
        # Plot confusion matrix
        cm = xgb_metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(cm, save_path=os.path.join(league_dir, "xgb_confusion_matrix.png"))
        
        # Lưu dự đoán XGBoost vào database
        # Lấy dữ liệu test gốc (không scaled) cho thông tin trận đấu
        test_data_original = df_clean[df_clean['season'].isin(split_data['seasons_info']['test_seasons']) & 
                                     (df_clean['league'] == league)]
        save_test_predictions(
            test_data_original, 
            split_data['X_test'], 
            split_data['y_test'], 
            xgb_model, 
            model_type="xgboost"
        )
        
        # Train LSTM model
        print(f"\nPreparing sequence data for LSTM model for {league}...")
        
        # Get league-specific data and sort by date
        league_specific_data = df_clean[df_clean['league'] == league].sort_values('match_date')
        
        # Prepare sequences
        X_seq, y_seq, match_info_seq = prepare_lstm_sequences(league_specific_data)
        
        if len(X_seq) < 100:
            print(f"Not enough sequence data for LSTM training for {league}. Skipping LSTM.")
            lstm_model = None
            lstm_metrics = None
            lstm_history = None
        else:
            # Create masks based on dates to match the time-based split
            train_seasons = split_data['seasons_info']['train_seasons']
            val_seasons = split_data['seasons_info']['val_seasons']
            test_seasons = split_data['seasons_info']['test_seasons']
            
            # Tạo mask dựa trên thông tin mùa giải trong match_info_seq
            train_mask = match_info_seq['season'].isin(train_seasons)
            val_mask = match_info_seq['season'].isin(val_seasons)
            test_mask = match_info_seq['season'].isin(test_seasons)
            
            # Split sequences
            X_train_seq = X_seq[train_mask]
            y_train_seq = y_seq[train_mask]
            match_info_train = match_info_seq[train_mask].reset_index(drop=True)
            
            X_val_seq = X_seq[val_mask]
            y_val_seq = y_seq[val_mask]
            match_info_val = match_info_seq[val_mask].reset_index(drop=True)
            
            X_test_seq = X_seq[test_mask]
            y_test_seq = y_seq[test_mask]
            match_info_test = match_info_seq[test_mask].reset_index(drop=True)
            
            print(f"LSTM training data: {len(X_train_seq)} sequences")
            print(f"LSTM validation data: {len(X_val_seq)} sequences")
            print(f"LSTM test data: {len(X_test_seq)} sequences")
            
            # Skip LSTM if not enough data in any split
            if len(X_train_seq) < 30 or len(X_val_seq) < 10 or len(X_test_seq) < 10:
                print(f"Insufficient data for LSTM training/validation/testing. Skipping LSTM for {league}.")
                lstm_model = None
                lstm_metrics = None
                lstm_history = None
            else:
                lstm_start_time = time.time()
                
                # Create and train LSTM model
                seq_length, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
                
                lstm_model, lstm_history = train_lstm(
                    X_train_seq, y_train_seq,
                    X_val_seq, y_val_seq,
                    batch_size=64,
                    epochs=100
                )
                
                lstm_training_time = time.time() - lstm_start_time
                print(f"LSTM training completed in {lstm_training_time:.2f} seconds")
                
                # Evaluate LSTM
                print(f"\nEvaluating LSTM model for {league}...")
                lstm_metrics = evaluate_lstm(lstm_model, X_test_seq, y_test_seq)
                
                # Save LSTM model and plots
                lstm_model_path = os.path.join(league_dir, "lstm_model.h5")
                if hasattr(lstm_model, 'save'):
                    lstm_model.save(lstm_model_path)
                
                # Plot learning curves with more metrics
                plot_learning_curves(
                    lstm_history, 
                    metrics=['accuracy', 'loss'],
                    save_path=os.path.join(league_dir, "lstm_learning_curves.png")
                )
                
                # Plot additional metrics if available
                if 'f1_score' in lstm_metrics and 'roc_auc' in lstm_metrics:
                    # Lưu các metrics bổ sung vào file
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'F1 Score', 'ROC AUC'],
                        'Value': [
                            lstm_metrics['accuracy'],
                            lstm_metrics['f1_score'],
                            lstm_metrics['roc_auc'] if lstm_metrics['roc_auc'] is not None else np.nan
                        ]
                    })
                    metrics_df.to_csv(os.path.join(league_dir, "lstm_metrics.csv"), index=False)
                    
                    # Vẽ biểu đồ các metrics
                    plt.figure(figsize=(10, 6))
                    plt.bar(metrics_df['Metric'], metrics_df['Value'])
                    plt.title(f'LSTM Model Metrics for {league}')
                    plt.ylim(0, 1)
                    plt.grid(axis='y', alpha=0.3)
                    plt.savefig(os.path.join(league_dir, "lstm_metrics_chart.png"))
                    plt.close()
                
                # Nếu có huấn luyện LSTM, lưu dự đoán LSTM
                if lstm_metrics is not None:
                    # Với mô hình LSTM, chúng ta cần dữ liệu chuỗi từ tập test
                    # Giả sử X_test_seq và y_test_seq đã được chuẩn bị
                    if 'X_test_seq' in locals() and 'y_test_seq' in locals() and 'match_info_test' in locals():
                        save_test_predictions(
                            match_info_test, 
                            X_test_seq,
                            y_test_seq, 
                            lstm_model, 
                            model_type="lstm"
                        )
        
        # Save model comparison if both models were trained
        if lstm_metrics is not None:
            model_results = {
                'XGBoost': xgb_metrics,
                'LSTM': lstm_metrics
            }
            
            comparison_df = compare_models(
                model_results,
                save_path=os.path.join(league_dir, "model_comparison.png")
            )
            
            # Lưu so sánh vào CSV
            comparison_df.to_csv(os.path.join(league_dir, "model_comparison.csv"), index=False)
        
        # Create summary for this league
        league_summary = {
            'League': league,
            'Method': 'time-based',
            'Train_size': split_info['train_size'],
            'Val_size': split_info['val_size'],
            'Test_size': split_info['test_size'],
            'XGBoost_Accuracy': xgb_metrics.get('accuracy', float('nan')),
            'XGBoost_F1': xgb_metrics.get('f1', float('nan')),
            'LSTM_Accuracy': lstm_metrics.get('accuracy', float('nan')) if lstm_metrics else float('nan'),
            'LSTM_F1': lstm_metrics.get('f1', float('nan')) if lstm_metrics else float('nan'),
            'Better_Model': 'XGBoost' if lstm_metrics is None or 
                             xgb_metrics.get('accuracy', 0) > lstm_metrics.get('accuracy', 0) else 'LSTM'
        }
        
        # Save league summary
        pd.DataFrame([league_summary]).to_csv(os.path.join(league_dir, "model_summary.csv"), index=False)
        league_summaries.append(league_summary)
        
        # Store league results
        league_results[league] = {
            'xgboost': {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'path': xgb_model_path
            },
            'lstm': {
                'model': lstm_model,
                'metrics': lstm_metrics,
                'history': lstm_history,
                'path': lstm_model_path if 'lstm_model_path' in locals() and lstm_model is not None else None
            },
            'split_info': split_info
        }
    
    # Create and save overall summary
    all_leagues_summary = pd.DataFrame(league_summaries)
    all_leagues_summary.to_csv(os.path.join(save_dir, "all_leagues_summary.csv"), index=False)
    
    # Create league comparison visualization
    create_league_comparison(league_results, save_dir)
    
    return {
        'method': 'time-based-by-league',
        'league_results': league_results,
        'summary': all_leagues_summary
    }

def create_summary_dataframe(results):
    """
    Create a summary DataFrame from training results
    """
    summary_rows = []
    
    for result in results:
        league = result['league']
        season = result['season']
        
        # XGBoost metrics
        xgb_metrics = result['xgboost']['metrics']
        xgb_accuracy = xgb_metrics.get('accuracy', float('nan'))
        xgb_f1 = xgb_metrics.get('f1', float('nan'))
        
        # LSTM metrics
        lstm_row = {}
        if result['lstm']['metrics'] is not None:
            lstm_metrics = result['lstm']['metrics']
            lstm_accuracy = lstm_metrics.get('accuracy', float('nan'))
            lstm_f1 = lstm_metrics.get('f1', float('nan'))
        else:
            lstm_accuracy = float('nan')
            lstm_f1 = float('nan')
        
        # Better model determination
        better_model = 'None'
        if not np.isnan(xgb_accuracy) and not np.isnan(lstm_accuracy):
            better_model = 'XGBoost' if xgb_accuracy > lstm_accuracy else 'LSTM'
        elif not np.isnan(xgb_accuracy):
            better_model = 'XGBoost'
        elif not np.isnan(lstm_accuracy):
            better_model = 'LSTM'
        
        row = {
            'League': league,
            'Season': season,
            'XGBoost_Accuracy': xgb_accuracy,
            'XGBoost_F1': xgb_f1,
            'LSTM_Accuracy': lstm_accuracy,
            'LSTM_F1': lstm_f1,
            'Better_Model': better_model
        }
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)

def create_league_comparison(league_results, save_dir):
    """
    Create league comparison visualizations
    """
    # Create plots directory
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Prepare data for plotting
    leagues = list(league_results.keys())
    
    # Model accuracy by league
    plt.figure(figsize=(12, 6))
    
    league_names = []
    xgb_accuracies = []
    lstm_accuracies = []
    
    for league, result in league_results.items():
        league_names.append(league)
        
        # XGBoost accuracy
        xgb_metrics = result['xgboost']['metrics']
        xgb_accuracies.append(xgb_metrics.get('accuracy', float('nan')))
        
        # LSTM accuracy
        if result['lstm']['metrics'] is not None:
            lstm_metrics = result['lstm']['metrics']
            lstm_accuracies.append(lstm_metrics.get('accuracy', float('nan')))
        else:
            lstm_accuracies.append(float('nan'))
    
    x = np.arange(len(league_names))
    width = 0.35
    
    plt.bar(x - width/2, xgb_accuracies, width, label='XGBoost')
    plt.bar(x + width/2, lstm_accuracies, width, label='LSTM')
    
    plt.title('Model Accuracy by League (Time-Based Split)')
    plt.xlabel('League')
    plt.ylabel('Accuracy')
    plt.xticks(x, league_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, "league_accuracy_comparison.png"))
    plt.close()
    
    # Dataset sizes
    plt.figure(figsize=(12, 6))
    
    train_sizes = []
    val_sizes = []
    test_sizes = []
    
    for league, result in league_results.items():
        split_info = result['split_info']
        train_sizes.append(split_info['train_size'])
        val_sizes.append(split_info['val_size'])
        test_sizes.append(split_info['test_size'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(leagues))
    
    p1 = ax.bar(leagues, train_sizes, label='Training')
    bottom += train_sizes
    
    p2 = ax.bar(leagues, val_sizes, bottom=bottom, label='Validation')
    bottom += val_sizes
    
    p3 = ax.bar(leagues, test_sizes, bottom=bottom, label='Test')
    
    ax.set_title('Dataset Sizes by League')
    ax.set_xlabel('League')
    ax.set_ylabel('Number of Matches')
    ax.legend()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, "league_dataset_sizes.png"))
    plt.close()
    
    return

if __name__ == "__main__":
    results = train_all_models("models")
    print("Model training completed!")