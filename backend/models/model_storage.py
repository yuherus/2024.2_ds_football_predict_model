"""Utilities for saving and loading trained models"""

import os
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import StandardScaler
from backend.models.config import MODEL_SAVE_DIR, XGBOOST_MODEL_FILENAME, LSTM_MODEL_FILENAME, SCALER_FILENAME
from backend.features.utils import get_pg_engine
from sqlalchemy import text as sqlalchemy_text

def ensure_model_dir():
    """Ensure the model save directory exists"""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    return MODEL_SAVE_DIR

def save_xgboost_model(model, model_path):
    """Save XGBoost model to disk"""
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Lưu model bằng pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"XGBoost model saved to {model_path}")
    return model_path

def load_xgboost_model(version=None):
    """Load XGBoost model from disk"""
    dir_path = ensure_model_dir()
    if version:
        filename = f"xgboost_football_predict_v{version}.pkl"
    else:
        filename = XGBOOST_MODEL_FILENAME
    
    model_path = os.path.join(dir_path, filename)
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Tải model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"XGBoost model loaded from {model_path}")
    return model

def save_lstm_model(model, model_path):
    """Save LSTM model to disk"""
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Lưu model
    model.save(model_path)
    
    print(f"LSTM model saved to {model_path}")
    return model_path

def load_lstm_model(version=None):
    """Load LSTM model from disk"""
    dir_path = ensure_model_dir()
    if version:
        filename = f"lstm_football_predict_v{version}.h5"
    else:
        filename = LSTM_MODEL_FILENAME
    
    model_path = os.path.join(dir_path, filename)
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Tải model
    model = load_model(model_path)
    
    print(f"LSTM model loaded from {model_path}")
    return model

def save_scaler(scaler):
    """Save the feature scaler to disk"""
    dir_path = ensure_model_dir()
    scaler_path = os.path.join(dir_path, SCALER_FILENAME)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Feature scaler saved to {scaler_path}")
    return scaler_path

def load_scaler():
    """Load the feature scaler from disk"""
    dir_path = ensure_model_dir()
    scaler_path = os.path.join(dir_path, SCALER_FILENAME)
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file {scaler_path} not found")
    
    # Tải scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Feature scaler loaded from {scaler_path}")
    return scaler

def save_model_metadata(model_type, metrics, hyperparams, version=None):
    """Save model metadata and evaluation metrics"""
    dir_path = ensure_model_dir()
    
    if version:
        filename = f"{model_type}_metadata_v{version}.json"
    else:
        filename = f"{model_type}_metadata.json"
    
    metadata_path = os.path.join(dir_path, filename)
    
    # Chuyển đổi các giá trị numpy/pandas thành dạng có thể lưu JSON
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    # Tạo metadata
    metadata = {
        'model_type': model_type,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {k: convert_to_serializable(v) for k, v in metrics.items()},
        'hyperparameters': hyperparams
    }
    
    # Lưu metadata dưới dạng JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model metadata saved to {metadata_path}")
    return metadata_path

def load_model_metadata(model_type, version=None):
    """Load model metadata"""
    dir_path = ensure_model_dir()
    
    if version:
        filename = f"{model_type}_metadata_v{version}.json"
    else:
        filename = f"{model_type}_metadata.json"
    
    metadata_path = os.path.join(dir_path, filename)
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} not found")
    
    # Tải metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def save_match_predictions(predictions_df, model_name="xgboost"):
    """
    Lưu kết quả dự đoán trận đấu vào bảng match_predictions
    
    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame chứa thông tin trận đấu và kết quả dự đoán
        Cần có các cột: match_date, round, home_team, away_team, 
                        probabilities (mảng numpy chứa xác suất cho 3 kết quả),
                        league, season, venue (nếu có)
    model_name : str
        Tên mô hình sử dụng để dự đoán
        
    Returns:
    --------
    int
        Số lượng dự đoán đã lưu vào cơ sở dữ liệu
    """
    if predictions_df.empty:
        print("Không có dự đoán nào để lưu")
        return 0
    
    # Tạo bản sao để không ảnh hưởng đến DataFrame gốc
    df_to_save = predictions_df.copy()
    
    # Tạo các cột mới cho định dạng phần trăm
    df_to_save['home_win_prob'] = df_to_save['probabilities'].apply(lambda x: float(x[2]) if isinstance(x, list) and len(x) > 2 else 0.0)
    df_to_save['draw_prob'] = df_to_save['probabilities'].apply(lambda x: float(x[1]) if isinstance(x, list) and len(x) > 1 else 0.0)
    df_to_save['away_win_prob'] = df_to_save['probabilities'].apply(lambda x: float(x[0]) if isinstance(x, list) and len(x) > 0 else 0.0)
    
    # Định dạng phần trăm
    df_to_save['home_win_pct'] = df_to_save['home_win_prob'].apply(lambda x: f"{x*100:.1f}%")
    df_to_save['draw_pct'] = df_to_save['draw_prob'].apply(lambda x: f"{x*100:.1f}%")
    df_to_save['away_win_pct'] = df_to_save['away_win_prob'].apply(lambda x: f"{x*100:.1f}%")
    
    # Thêm cột predicted_result (kết quả dự đoán: 0=away win, 1=draw, 2=home win)
    df_to_save['predicted_result'] = df_to_save.apply(
        lambda row: int(np.argmax([row['away_win_prob'], row['draw_prob'], row['home_win_prob']])), axis=1
    )
    
    # Thêm cột model
    df_to_save['prediction_model'] = model_name
    
    # Chuẩn bị dữ liệu để lưu vào DB
    db_columns = [
        'match_date', 'round', 'home_team', 'away_team',
        'home_win_prob', 'draw_prob', 'away_win_prob',
        'home_win_pct', 'draw_pct', 'away_win_pct',
        'league', 'season', 'venue', 'prediction_model',
        'predicted_result', 'actual_result'
    ]
    
    # Đảm bảo tất cả các cột tồn tại
    for col in db_columns:
        if col not in df_to_save.columns:
            if col == 'actual_result':
                # Đặt giá trị này là NULL khi chưa biết kết quả thực tế
                df_to_save[col] = None
            elif col in ['league', 'season', 'venue']:
                # Các thông tin bổ sung này có thể không bắt buộc
                df_to_save[col] = None
            else:
                raise ValueError(f"Cột {col} bắt buộc phải có trong DataFrame")
    
    # Loại bỏ cột probabilities vì không lưu vào DB
    if 'probabilities' in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=['probabilities'])
    
    # Chọn các cột cần thiết cho DB
    predictions_to_db = df_to_save[db_columns]
    
    # Bỏ qua hàng có giá trị null trong các cột quan trọng
    for col in ['match_date', 'home_team', 'away_team']:
        predictions_to_db = predictions_to_db[predictions_to_db[col].notna()]
    
    if predictions_to_db.empty:
        print("Không có dự đoán hợp lệ để lưu sau khi lọc giá trị null")
        return 0
    
    # Lưu vào database
    engine = get_pg_engine()
    try:
        # Xử lý trùng lặp bằng cách cập nhật nếu đã tồn tại
        # (dựa trên ràng buộc unique_prediction)
        predictions_to_db.to_sql('match_predictions', engine, if_exists='append', index=False)
        print(f"Đã lưu {len(predictions_to_db)} dự đoán vào database")
        return len(predictions_to_db)
    except Exception as e:
        print(f"Lỗi khi lưu dự đoán: {e}")
        
        # Xử lý trường hợp vi phạm ràng buộc unique
        # Thử cập nhật từng dòng một
        rows_updated = 0
        for _, row in predictions_to_db.iterrows():
            try:
                # Chuyển đổi row thành dict để tránh lỗi "List argument must consist only of tuples or dictionaries"
                row_dict = row.to_dict()
                
                # Xóa bản ghi cũ (nếu có)
                delete_query = sqlalchemy_text("""
                DELETE FROM match_predictions 
                WHERE match_date = :match_date AND home_team = :home_team 
                AND away_team = :away_team AND prediction_model = :prediction_model
                """)
                
                with engine.connect() as conn:
                    conn.execute(delete_query, {
                        'match_date': row_dict['match_date'],
                        'home_team': row_dict['home_team'],
                        'away_team': row_dict['away_team'],
                        'prediction_model': row_dict['prediction_model']
                    })
                    conn.commit()
                    
                # Thêm bản ghi mới - chuyển đổi thành DataFrame một hàng
                row_df = pd.DataFrame([row_dict])
                row_df.to_sql('match_predictions', engine, if_exists='append', index=False)
                rows_updated += 1
            except Exception as row_error:
                print(f"Không thể cập nhật dự đoán cho trận {row['home_team']} vs {row['away_team']}: {row_error}")
        
        print(f"Đã cập nhật {rows_updated} dự đoán")
        return rows_updated
