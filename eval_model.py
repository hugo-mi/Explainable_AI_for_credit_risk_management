import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from train_model import predict_model

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix as a heatmap."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_probs):
    """Plot the ROC-AUC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_lift_curve(y_true, y_probs):
    # Sort by predicted probabilities
    data = pd.DataFrame({'true': y_true, 'probs': y_probs})
    data.sort_values(by='probs', ascending=False, inplace=True)
    
    # Calculate cumulative rate of positives
    data['cumulative_data'] = np.arange(1, len(data) + 1)
    data['cumulative_positives'] = data['true'].cumsum()
    
    total_positives = data['true'].sum()
    total_data = len(data)
    
    # Calculate lift
    data['lift'] = (data['cumulative_positives'] / data['cumulative_data']) / (total_positives / total_data)
    
    # Plot lift curve
    plt.figure(figsize=(10, 7))
    plt.plot(data['cumulative_data'] / total_data, data['lift'], label='Lift Curve')
    plt.axhline(1, color='red', linestyle='--', label='Baseline')
    plt.xlabel('Proportion of Data')
    plt.ylabel('Lift')
    plt.title('Lift Curve')
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate the model performance."""
    y_preds, y_probs = predict_model(model, X_test)
    
    y_preds = (y_probs[:,1] > threshold).astype(int)

    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_preds, target_names=['Non-Default', 'Default']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_preds)
    plot_confusion_matrix(cm, class_names=['Non-Default', 'Default'])
    
    # ROC-AUC curve
    plot_roc_curve(y_test, y_probs[:, 1])
    
    # Lift curve
    plot_lift_curve(y_test, y_probs[:, 1])

# Example usage:
# Assuming df, X_train, y_train, X_test, y_test are already defined.
# model = train_model(df, X_train, y_train, imbalance=True)
# evaluate_model(model, X_test, y_test)
