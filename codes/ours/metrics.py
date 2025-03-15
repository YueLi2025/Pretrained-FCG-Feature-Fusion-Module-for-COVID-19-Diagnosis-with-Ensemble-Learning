"""
Evaluation Metrics for COVID-19 Classification

This module provides functions to calculate performance metrics from confusion matrices
for evaluating the COVID-19 classification model.

The main metrics include:
- Per-class recall (sensitivity)
- Overall accuracy
"""

import numpy as np


def acc_from_cm(cm: np.ndarray):
    """
    Compute accuracy metrics from a confusion matrix.
    
    This function calculates both per-class recall (sensitivity) values and
    the overall accuracy from a confusion matrix. For medical image classification,
    per-class recall is particularly important as it shows the model's ability
    to correctly identify each disease category.
    
    Args:
        cm (np.ndarray): Confusion matrix of shape (n_classes, n_classes)
                         where rows represent true classes and columns represent
                         predicted classes.
    
    Returns:
        list: A list containing per-class recall values followed by the overall accuracy.
              Format: [recall_class0, recall_class1, ..., recall_classN, overall_accuracy]
              For COVID-19 classification, this would be:
              [recall_covid, recall_lung_opacity, recall_normal, recall_viral_pneumonia, overall_accuracy]
    """
    # Verify the confusion matrix is square (same number of true and predicted classes)
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1], "Confusion matrix must be square"

    # Calculate correctly classified samples (sum of diagonal elements)
    correct = 0
    all_correct = np.zeros(len(cm), dtype=int)
    for i in range(len(cm)):
        correct += cm[i, i]  # Add diagonal elements (correct predictions)
        all_correct[i] += cm[i, i]  # Store correct predictions for each class

    # Calculate total number of samples
    total = np.sum(cm)

    # Calculate per-class recall (sensitivity): correctly classified / total samples in that class
    all_acc = all_correct / np.sum(cm, axis=1)
    
    # Convert to list and append overall accuracy
    all_acc = list(all_acc)
    all_acc.append(correct / total)  # Overall accuracy as the last element

    return all_acc
