# backend/models/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def create_lstm_model(seq_length, n_features, n_classes=3):
    """
    Create LSTM model architecture
    """
    model = Sequential([
        LSTM(64, input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, model_path="models/lstm_model.h5"):
    """
    Train LSTM model
    """
    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    model = create_lstm_model(seq_length, n_features)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_lstm(model, X_test, y_test):
    """
    Evaluate LSTM model
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_lstm_model(model, filepath="models/lstm_model.h5"):
    """
    Save LSTM model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"LSTM model saved to {filepath}")

def load_lstm_model(filepath="models/lstm_model.h5"):
    """
    Load LSTM model
    """
    return load_model(filepath)