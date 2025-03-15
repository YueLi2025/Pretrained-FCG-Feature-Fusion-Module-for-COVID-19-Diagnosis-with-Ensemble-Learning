import numpy as np
from typing import List

def acc_from_cm(cm: np.ndarray) -> List[float]:
    """
    Compute the accuracy from a confusion matrix.

    Args:
        cm (np.ndarray): A square confusion matrix of shape (N, N),
                         where N is the number of classes.

    Returns:
        List[float]: A list containing class-wise accuracy and overall accuracy.
                     The last element in the list is the overall accuracy.
    """
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1], "Confusion matrix must be square."

    # Correctly classified samples per class
    per_class_correct = np.diag(cm)  # Extract the diagonal (true positives for each class)
    
    # Total samples per class (actual class counts)
    per_class_total = cm.sum(axis=1)

    # Compute class-wise accuracy (avoid division by zero)
    per_class_accuracy = np.divide(per_class_correct, per_class_total, 
                                   out=np.zeros_like(per_class_correct, dtype=float), 
                                   where=per_class_total != 0)

    # Compute overall accuracy
    overall_accuracy = per_class_correct.sum() / cm.sum()

    return list(per_class_accuracy) + [overall_accuracy]

