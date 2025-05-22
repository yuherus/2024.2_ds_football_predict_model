"""Configuration settings for machine learning models"""

# Đường dẫn cho việc lưu trữ model
MODEL_SAVE_DIR = "backend/models/saved_models"
XGBOOST_MODEL_FILENAME = "xgboost_football_predict.pkl"
LSTM_MODEL_FILENAME = "lstm_football_predict.h5"
SCALER_FILENAME = "feature_scaler.pkl"

# Cấu hình XGBoost
XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,  # 0=away win, 1=draw, 2=home win
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'merror',
    'seed': 42
}

# Số vòng lặp tối đa và cơ chế early stopping cho XGBoost
XGBOOST_NUM_ROUNDS = 1000
XGBOOST_EARLY_STOPPING = 50

# Cấu hình LSTM
LSTM_SEQUENCE_LENGTH = 5  # Số trận đấu trong một chuỗi
LSTM_FEATURES = ['home_points', 'away_points', 'home_score', 'away_score', 
                 'home_possession', 'away_possession', 'home_shots_on_target', 'away_shots_on_target']

# Tham số training cho LSTM
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_PATIENCE = 10  # Early stopping patience

# Tham số đánh giá
EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
CLASS_LABELS = ['Away Win', 'Draw', 'Home Win']

# Tỷ lệ chia tập dữ liệu
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Random seed để đảm bảo tính nhất quán
RANDOM_SEED = 42
