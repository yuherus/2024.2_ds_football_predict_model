# backend/models/xgboost_model.py
import xgboost as xgb
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train an XGBoost model for match prediction
    """
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # 0=away win, 1=draw, 2=home win
            'learning_rate': 0.01,
            'max_depth': 4,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'scale_pos_weight': [1, 3, 1],
            'seed': 42
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
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