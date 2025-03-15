"""
Mixup Data Augmentation Implementation

This module provides functions for performing Mixup augmentation, 
computing the Mixup loss, and evaluating correctness under Mixup.

Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
"""

import torch
from numpy.random import beta


def mixup_inputs(inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.1):
    """
    Applies Mixup augmentation to input images and labels.

    Args:
        inputs (torch.Tensor): Batch of input images (BxC×H×W).
        labels (torch.Tensor): Corresponding class labels (B).
        alpha (float): Mixup interpolation parameter (default: 0.1).

    Returns:
        tuple: (Mixed inputs, original labels, shuffled labels, lambda coefficient)
    """
    lam = beta(alpha, alpha)
    mix_index = torch.randperm(inputs.size(0)).to(inputs.device)

    # Perform Mixup
    mixed_inputs = lam * inputs + (1 - lam) * inputs[mix_index, :]
    mixed_labels = labels[mix_index]

    return mixed_inputs, labels, mixed_labels, lam


def mixup_loss(criterion, outputs: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor, lam: float):
    """
    Computes Mixup loss as a weighted combination of two labels.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss).
        outputs (torch.Tensor): Model predictions (logits).
        labels_a (torch.Tensor): Original labels.
        labels_b (torch.Tensor): Mixed labels.
        lam (float): Lambda coefficient for Mixup.

    Returns:
        torch.Tensor: Computed Mixup loss.
    """
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def mixup_correct(predicted: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor, lam: float):
    """
    Computes the number of correctly classified samples under Mixup.

    This method estimates correctness as a **weighted sum** of classification 
    accuracy for both labels (not directly comparable to validation accuracy).

    Args:
        predicted (torch.Tensor): Model predictions (class indices).
        labels_a (torch.Tensor): Original labels.
        labels_b (torch.Tensor): Mixed labels.
        lam (float): Lambda coefficient for Mixup.

    Returns:
        float: Approximate count of correctly classified samples.
    """
    return (
        lam * (predicted == labels_a).cpu().sum().float() +
        (1 - lam) * (predicted == labels_b).cpu().sum().float()
    )


#--------------------------------------------------------
# Example Usage of Mixup in Training Loop:

# for i, (inputs, labels) in enumerate(train_loader):
#     inputs, labels = inputs.to(device), labels.to(device)

#     # Apply Mixup
#     if mix_up:
#         inputs, labels_a, labels_b, lam = mixup_inputs(inputs, labels, alpha)

#     outputs = model(inputs)

#     # Compute Loss
#     if mix_up:
#         loss = mixup_loss(criterion, outputs, labels_a, labels_b, lam)
#     else:
#         loss = criterion(outputs, labels)

#     # Backpropagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Compute Accuracy
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     if mix_up:
#         correct += mixup_correct(predicted, labels_a, labels_b, lam)
#     else:
#         correct += (predicted == labels).cpu().sum().item()
