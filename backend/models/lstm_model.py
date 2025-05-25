# backend/models/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, regularizers
import os
import matplotlib.pyplot as plt

def create_lstm_model(seq_length, n_features, n_classes=3):
    """
    Create LSTM model architecture with regularization to prevent overfitting
    """
    # Sử dụng Functional API
    inputs = Input(shape=(seq_length, n_features))
    
    # First Bidirectional LSTM layer with L2 regularization
    x = Bidirectional(LSTM(64, return_sequences=True, 
                          dropout=0.3,
                          recurrent_dropout=0.3,
                          kernel_regularizer=regularizers.l2(0.001),
                          recurrent_regularizer=regularizers.l2(0.001)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Increased dropout
    
    # Second Bidirectional LSTM layer (reduced complexity)
    x = Bidirectional(LSTM(24, return_sequences=False,
                          dropout=0.4,
                          recurrent_dropout=0.4,
                          kernel_regularizer=regularizers.l2(0.001),
                          recurrent_regularizer=regularizers.l2(0.001)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)  # Increased dropout
    
    # Dense layers with L2 regularization
    x = Dense(16, activation='relu', 
             kernel_regularizer=regularizers.l2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Increased dropout
    
    # Output layer
    outputs = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=0.0003,
                     beta_1= 0.9,
                     beta_2= 0.999,
                     epsilon= 1e-08,
                     clipnorm=1.0 
                    )  # Lower learning rate
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm(X_train, y_train, X_val, y_val, epochs=30, batch_size=64, model_path="models/lstm_model.h5"):
    """
    Train LSTM model with improved training parameters to prevent overfitting
    """
    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    model = create_lstm_model(seq_length, n_features)
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=12,  # Increased patience
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005
    )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',  # Changed to monitor val_loss instead of val_accuracy
        save_best_only=True,
        mode='min',  # Changed to min mode for loss
        verbose=1
    )
    
    # Learning rate scheduler with more gradual reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More gradual reduction
        patience=6,  # Increased patience
        min_lr=0.000001,
        verbose=1
    )
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in zip(classes, class_weights)}
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,  # Smaller batch size
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return model, history

def evaluate_lstm(model, X_test, y_test, league=None, season=None):
    """
    Evaluate LSTM model with enhanced metrics
    
    Parameters:
    -----------
    model : keras.Model
        Trained LSTM model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    league : str, optional
        Tên giải đấu (nếu có)
    season : str, optional
        Mùa giải (nếu có)
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
    
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate F1 score (macro average)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Calculate ROC AUC for multi-class
    try:
        roc_auc = roc_auc_score(
            tf.keras.utils.to_categorical(y_test), 
            y_pred_proba, 
            multi_class='ovr'
        )
    except:
        roc_auc = None
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Lưu classification report vào file
    from backend.models.model_evaluation import save_classification_report
    report_path = save_classification_report(y_test, y_pred, model_type='lstm', league=league, season=season)
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'classification_report': report,
        'classification_report_path': report_path,
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