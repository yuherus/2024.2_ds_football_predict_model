# backend/models/xgboost_model.py
import xgboost as xgb
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

def find_best_params(X_train, y_train, X_val=None, y_val=None, cv=3):
    """
    Sử dụng GridSearchCV để tìm tham số tốt nhất cho XGBoost
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like, optional
        Validation features (nếu cung cấp, sẽ được sử dụng để đánh giá)
    y_val : array-like, optional
        Validation labels
    cv : int
        Số lượng fold cho cross-validation
        
    Returns:
    --------
    dict
        Tham số tốt nhất tìm được
    """
    print("Bắt đầu tìm tham số tốt nhất cho XGBoost...")
    start_time = time.time()
    
    # Kết hợp dữ liệu train và validation nếu có
    if X_val is not None and y_val is not None:
        X_combined = np.vstack((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))
    else:
        X_combined = X_train
        y_combined = y_train
    
    # Tạo mô hình XGBoost với sklearn API
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Định nghĩa grid tham số để tìm kiếm
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'n_estimators': [100, 200, 300]
    }
    
    # Sử dụng GridSearchCV để tìm tham số tốt nhất
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Huấn luyện GridSearchCV
    grid_search.fit(X_combined, y_combined)
    
    # Lấy tham số tốt nhất
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Tính thời gian chạy
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Tìm tham số tốt nhất hoàn thành trong {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Tham số tốt nhất: {best_params}")
    print(f"Điểm số tốt nhất (accuracy): {best_score:.4f}")
    
    # Lưu kết quả tìm kiếm vào file
    results_df = pd.DataFrame(grid_search.cv_results_)
    os.makedirs("models/grid_search_results", exist_ok=True)
    results_df.to_csv("models/grid_search_results/xgboost_grid_search_results.csv", index=False)
    
    # Vẽ biểu đồ so sánh các tham số
    plot_param_importance(grid_search)
    
    # Chuyển đổi tham số cho xgboost.train API
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': best_params['learning_rate'],
        'max_depth': best_params['max_depth'],
        'min_child_weight': best_params['min_child_weight'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'gamma': best_params['gamma'],
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    return xgb_params

def plot_param_importance(grid_search):
    """
    Vẽ biểu đồ so sánh tầm quan trọng của các tham số
    """
    # Lấy kết quả từ grid search
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Lấy tham số tốt nhất
    best_params = grid_search.best_params_
    
    # Tạo thư mục để lưu biểu đồ
    os.makedirs("models/grid_search_results/plots", exist_ok=True)
    
    # Vẽ biểu đồ cho từng tham số
    for param_name in best_params.keys():
        plt.figure(figsize=(10, 6))
        
        # Lọc kết quả chỉ cho tham số hiện tại
        param_results = results.copy()
        
        # Tạo cột mới để nhóm theo tham số hiện tại
        param_results['param_value'] = param_results[f'param_{param_name}'].astype(str)
        
        # Tính giá trị trung bình và độ lệch chuẩn cho mỗi giá trị tham số
        param_summary = param_results.groupby('param_value')['mean_test_score'].agg(['mean', 'std']).reset_index()
        
        # Vẽ biểu đồ
        plt.errorbar(
            x=param_summary['param_value'],
            y=param_summary['mean'],
            yerr=param_summary['std'],
            fmt='o-',
            capsize=5
        )
        
        plt.title(f'Effect of {param_name} on Accuracy')
        plt.xlabel(param_name)
        plt.ylabel('Mean Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Đánh dấu giá trị tốt nhất
        best_value = str(best_params[param_name])
        best_score = param_summary[param_summary['param_value'] == best_value]['mean'].values[0]
        plt.scatter([best_value], [best_score], color='red', s=100, marker='*')
        plt.annotate(f'Best: {best_value}', 
                    (best_value, best_score), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
        
        # Lưu biểu đồ
        plt.savefig(f"models/grid_search_results/plots/xgboost_{param_name}_comparison.png")
        plt.close()
    
    # Tạo biểu đồ heatmap cho các cặp tham số quan trọng
    if 'max_depth' in best_params and 'learning_rate' in best_params:
        plt.figure(figsize=(10, 8))
        
        # Tạo pivot table cho cặp tham số max_depth và learning_rate
        heatmap_data = results.pivot_table(
            values='mean_test_score', 
            index='param_max_depth', 
            columns='param_learning_rate'
        )
        
        # Vẽ heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.4f')
        plt.title('Accuracy for Different max_depth and learning_rate Values')
        plt.ylabel('max_depth')
        plt.xlabel('learning_rate')
        
        # Lưu biểu đồ
        plt.savefig("models/grid_search_results/plots/xgboost_depth_lr_heatmap.png")
        plt.close()

def train_xgboost(X_train, y_train, X_val, y_val, params=None, use_grid_search=True):
    """
    Train an XGBoost model for match prediction
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    params : dict, optional
        XGBoost parameters
    use_grid_search : bool, default=True
        Whether to use grid search to find the best parameters
        
    Returns:
    --------
    xgb.Booster
        Trained XGBoost model
    """
    if use_grid_search and params is None:
        # Tìm tham số tốt nhất bằng GridSearchCV
        params = find_best_params(X_train, y_train, X_val, y_val)
    elif params is None:
        # Sử dụng tham số mặc định
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # 0=away win, 1=draw, 2=home win
            'learning_rate': 0.01,
            'max_depth': 4,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': 42
        }
    
    # Thêm scale_pos_weight nếu cần
    if 'scale_pos_weight' not in params:
        # Tính toán scale_pos_weight dựa trên phân phối lớp
        class_counts = np.bincount(y_train)
        if len(class_counts) == 3:
            # Tính tỷ lệ giữa các lớp
            scale_pos_weight = [class_counts[0]/class_counts[0], 
                               class_counts[0]/class_counts[1], 
                               class_counts[0]/class_counts[2]]
            params['scale_pos_weight'] = scale_pos_weight
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    # Lấy n_estimators từ grid search nếu có
    num_boost_round = params.pop('n_estimators', 2000) if 'n_estimators' in params else 2000
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    return model

def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate XGBoost model performance
    """
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_xgboost_model(model, filepath="models/xgboost_model.pkl"):
    """
    Save the trained XGBoost model to disk
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_xgboost_model(filepath="models/xgboost_model.pkl"):
    """
    Load a trained XGBoost model from disk
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model