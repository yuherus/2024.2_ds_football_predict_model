# backend/models/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import os

def create_lstm_model(seq_length, n_features, n_classes=3):
    """
    Create enhanced LSTM model architecture with attention mechanism
    """
    # Sử dụng Functional API thay vì Sequential để có thể thêm attention
    inputs = Input(shape=(seq_length, n_features))
    
    # Bidirectional LSTM layer with more units
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(
        key_dim=64, num_heads=2, dropout=0.3
    )(x, x)
    x = tf.keras.layers.Add()([attention, x])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Final LSTM layer
    x = LSTM(32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    # Output layer
    outputs = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_path="models/lstm_model.h5"):
    """
    Train LSTM model with improved training parameters
    """
    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    model = create_lstm_model(seq_length, n_features)
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Learning rate scheduler
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model with class weights to handle imbalance
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
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return model, history

def evaluate_lstm(model, X_test, y_test):
    """
    Evaluate LSTM model with enhanced metrics
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
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'f1_score': f1,
        'roc_auc': roc_auc,
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