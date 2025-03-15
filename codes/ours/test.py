"""
Test Script for COVID-19 Classification Model

This script evaluates a trained deep learning model on a test dataset of chest X-ray/CT images.
It calculates and reports various performance metrics including:
- Loss
- Overall accuracy
- Per-class recall (sensitivity)
- ROC AUC scores (both one-vs-rest and one-vs-one)
- Confusion matrix

These metrics provide a comprehensive assessment of the model's performance
in classifying images into four categories: COVID-19, lung opacity, normal, and viral pneumonia.
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from dataset import MyDataset
from model import MyModel
from metrics import acc_from_cm
from config import parser_config, logger_config, seed_config
import os, sys
import torch.nn.functional as F
from utils.enhance_trans import *
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings

def eval():
    """
    Evaluate the model on the test dataset.
    
    This function:
    1. Iterates through the test data loader
    2. Calculates test metrics including:
       - Loss
       - Accuracy (overall and per-class)
       - ROC AUC scores (one-vs-rest and one-vs-one)
    3. Logs detailed performance metrics and confusion matrix
    
    ROC AUC scores are calculated in two ways:
    - One-vs-Rest (OVR): Each class against all others
    - One-vs-One (OVO): Each class pair against each other
    
    These metrics are particularly important for medical diagnosis tasks
    where understanding model performance across different disease categories is critical.
    """
    loss_val = 0.0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        cm = np.zeros((args.num_classes, args.num_classes), dtype=int)  # Confusion matrix
        all_labels = []  # Store all true labels for AUC calculation
        all_preds = []   # Store all predicted probabilities for AUC calculation
        
        # Iterate through test data
        for j, data in enumerate(valid_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Convert logits to probabilities for AUC calculation
            outputs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            all_labels.append(labels.to('cpu'))  # Store true labels
            all_preds.append(outputs.to('cpu'))  # Store predicted probabilities

            # Get class predictions
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate test loss
            loss_val += loss.item()

            # Update confusion matrix
            for y_pred, y_true in zip(predicted, labels):
                cm[y_true][y_pred] += 1

        # Calculate average test loss and accuracy metrics
        loss_val = loss_val / len(valid_loader)
        val_acc = acc_from_cm(cm)

        # Calculate ROC AUC scores
        all_labels = torch.cat(all_labels)  # Concatenate all batches of labels
        all_preds = torch.cat(all_preds)    # Concatenate all batches of predictions
        
        # Calculate AUC with two different averaging strategies
        auc_ovr = roc_auc_score(all_labels, all_preds, multi_class='ovr')  # One-vs-Rest
        auc_ovo = roc_auc_score(all_labels, all_preds, multi_class='ovo')  # One-vs-One

        # Log test results with detailed metrics
        logger.info(
            "Valid - Loss: {:.4f}, Acc_mean: {:.2%}, AUC_ovr: {:.2%}, AUC_ovo: {:.2%}, Recall: ({:.2%}, {:.2%}, {:.2%}, {:.2%})\n".format(
                loss_val,
                val_acc[-1],  # Overall accuracy
                auc_ovr,      # One-vs-Rest AUC
                auc_ovo,      # One-vs-One AUC
                val_acc[0],   # COVID-19 recall
                val_acc[1],   # Lung opacity recall
                val_acc[2],   # Normal recall
                val_acc[3],   # Viral pneumonia recall
            )
        )
        # Log confusion matrix for detailed error analysis
        logger.info("cm:\n{}".format(cm))


if __name__ == "__main__":
    # Parse command-line arguments
    args = parser_config()
    
    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    logger = logger_config(os.path.join("logs", f"{args.backbone}_{args.log_file}"))
    
    # Set random seed for reproducibility
    seed_config(args.seed)
    
    # Log configuration
    logger.info(args)
    logger.info("Random seed: {}".format(args.seed))
    
    # Set device (GPU/CPU)
    if not args.device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Running on device: {}".format(device))

    # Define data transformations for testing
    # Note: For testing, we use minimal transformations to preserve image content
    valid_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),      # Resize to standard size
            transforms.RandomCrop((224, 224)),  # Center crop to model input size
            transforms.ToTensor(),              # Convert to tensor
        ]
    )

    # Create test dataset
    valid_data = MyDataset(
        data_dir=args.val_dir,
        train=False,
        transform=valid_transform,
        device=args.device,
    )

    # Create test data loader
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size)

    # Initialize model with the same architecture used during training
    model = MyModel(args.num_classes, args)

    # Define loss function for evaluation
    criterion = F.cross_entropy

    # Load trained model weights from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    logger.info("Loaded checkpoint from {}.".format(args.checkpoint))
    
    # Set model to evaluation mode
    model.eval()
    
    # Run evaluation
    eval()