import numpy as np


def acc_from_cm(cm: np.ndarray):
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    all_correct = np.zeros(len(cm), dtype=int)
    for i in range(len(cm)):
        correct += cm[i, i]
        all_correct[i] += cm[i, i]

    total = np.sum(cm)

    all_acc = all_correct / np.sum(cm, axis=1)
    all_acc = list(all_acc)
    all_acc.append(correct / total)

    return all_acc
