import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate_model_isotonic(fitted_model, X_test, y_test):
    """
    fitted_model: use of fitted mode (i.e pre trained model on train set)
    """
    # Calibrate the model using isotonic regression
    calibrated_model = CalibratedClassifierCV(fitted_model, method='isotonic', cv='prefit')
    
    # Fit the calibrated model
    calibrated_model.fit(X_test, y_test)
    
    # Predict probabilities of default
    y_probs_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    y_probs_uncalibrated = fitted_model.predict_proba(X_test)[:, 1]
    
    # Calibration curve
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_probs_calibrated, n_bins=40)
    prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(y_test, y_probs_uncalibrated, n_bins=40)

    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='Calibrated model')
    plt.plot(prob_pred_uncalibrated, prob_true_uncalibrated, marker='o', label='Uncalibrated model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    return calibrated_model


def find_best_threshold_recall(X_test, y_test, y_probs):
    
    # Initialize lists for metrics
    thresholds = np.arange(0.0, 1.1, 0.01)
    non_default_recall_scores = []
    default_recall_scores = []
    accuracy_scores = []
    
    # Calculate metrics for different thresholds
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        non_default_recall_scores.append(recall_score(y_test, y_pred, pos_label=0))
        default_recall_scores.append(recall_score(y_test, y_pred, pos_label=1))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, non_default_recall_scores, label='Non-Default Recall')
    plt.plot(thresholds, default_recall_scores, label='Default Recall')
    plt.plot(thresholds, accuracy_scores, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find the best balanced threshold
    differences = np.abs(np.array(non_default_recall_scores) - np.array(default_recall_scores))
    best_idx = np.argmin(differences)
    best_threshold = thresholds[best_idx]
    best_non_default_recall = non_default_recall_scores[best_idx]
    best_default_recall = default_recall_scores[best_idx]
    best_accuracy = accuracy_scores[best_idx]
    
    print(f'Best Balanced Threshold: {best_threshold}')
    print(f'Non-Default Recall: {best_non_default_recall}')
    print(f'Default Recall: {best_default_recall}')
    print(f'Accuracy: {best_accuracy}')
    
    return best_threshold

def find_best_threshold_precision(X_test, y_test, y_probs):

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate accuracy for each threshold
    accuracy = [(y_probs >= t).astype(int) == y_test.values for t in thresholds]
    accuracy = [accuracy_score(y_test, pred) for pred in accuracy]

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, accuracy, 'r-', label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Precision, Recall, and Accuracy vs. Threshold')
    plt.show()

    # Find the best threshold based on precision
    differences = np.abs(np.array(precision) - np.array(recall))
    best_idx = np.argmin(differences)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_accuracy = accuracy[best_idx]

    print(f'Best Balanced Threshold: {best_threshold}')
    print(f'Default Recall: {best_recall}')
    print(f'Default Precision: {best_precision}')
    print(f'Accuracy: {best_accuracy}')
    
    return best_threshold