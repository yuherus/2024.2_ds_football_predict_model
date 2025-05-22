"""Utilities for evaluating model performance"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from backend.models.config import CLASS_LABELS, EVALUATION_METRICS

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate common classification metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for each class
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (weighted average across all classes)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Class-specific metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics['class_report'] = class_report
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC AUC if probability predictions are provided
    if y_pred_proba is not None:
        # For multi-class, compute OvR ROC AUC
        n_classes = y_pred_proba.shape[1]
        metrics['roc_auc'] = {}
        
        for i in range(n_classes):
            class_label = CLASS_LABELS[i]
            # Convert to binary classification for this class
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            try:
                # ROC AUC for this class
                metrics['roc_auc'][class_label] = auc(
                    *roc_curve(y_true_binary, y_score)[:2]
                )
            except:
                metrics['roc_auc'][class_label] = np.nan
        
        # Average ROC AUC across all classes
        metrics['roc_auc_avg'] = np.mean([v for v in metrics['roc_auc'].values() if not np.isnan(v)])
    
    return metrics

def plot_confusion_matrix(confusion_mat, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    confusion_mat : array-like
        Confusion matrix to plot
    save_path : str, optional
        Path to save the plot image
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curves for multi-class classification
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for each class
    save_path : str, optional
        Path to save the plot image
    """
    n_classes = y_pred_proba.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        class_label = CLASS_LABELS[i]
        
        # Convert to binary classification for this class
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(
            fpr, tpr,
            label=f'{class_label} (AUC = {roc_auc:.2f})'
        )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curves plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_learning_curves(history, save_path=None):
    """
    Plot learning curves for neural network training
    
    Parameters:
    -----------
    history : History
        Keras History object from model.fit
    save_path : str, optional
        Path to save the plot image
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def compare_models(model_results, save_path=None):
    """
    Compare multiple models based on their evaluation metrics
    
    Parameters:
    -----------
    model_results : dict
        Dictionary mapping model names to their evaluation metrics
    save_path : str, optional
        Path to save the comparison plot
    """
    # Extract common metrics for comparison
    model_names = list(model_results.keys())
    comparison_metrics = EVALUATION_METRICS
    
    # Prepare data for plotting
    metrics_data = []
    
    for model_name in model_names:
        model_metrics = model_results[model_name]
        
        # Extract metrics
        row = {'Model': model_name}
        for metric in comparison_metrics:
            if metric in model_metrics:
                if isinstance(model_metrics[metric], dict):
                    # For metrics like ROC AUC that might be dictionaries
                    row[metric] = np.mean([v for v in model_metrics[metric].values() 
                                         if not (isinstance(v, float) and np.isnan(v))])
                else:
                    row[metric] = model_metrics[metric]
            else:
                row[metric] = np.nan
        
        metrics_data.append(row)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(metrics_data)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(comparison_df, id_vars=['Model'], 
                       value_vars=comparison_metrics,
                       var_name='Metric', value_name='Value')
    
    # Create a grouped bar chart
    sns.barplot(x='Metric', y='Value', hue='Model', data=melted_df)
    
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Most metrics are between 0 and 1
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Also return the comparison DataFrame
    return comparison_df

def evaluate_model_on_test_data(model, X_test, y_test, model_type='xgboost'):
    """
    Evaluate a trained model on test data
    
    Parameters:
    -----------
    model : trained model
        The trained model to evaluate
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_type : str, default='xgboost'
        Type of model being evaluated ('xgboost' or 'lstm')
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Get predictions
    if model_type.lower() == 'xgboost':
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    elif model_type.lower() == 'lstm':
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Print summary
    print(f"\n{model_type.upper()} Model Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if 'roc_auc_avg' in metrics:
        print(f"ROC AUC (avg): {metrics['roc_auc_avg']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Plot ROC curves if probabilities are available
    if 'roc_auc' in metrics:
        plot_roc_curves(y_test, y_pred_proba)
    
    return metrics
