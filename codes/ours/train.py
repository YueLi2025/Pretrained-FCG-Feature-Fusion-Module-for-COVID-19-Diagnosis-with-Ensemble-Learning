"""
Training Script for COVID-19 Classification Model

This script implements the training and validation pipeline for a deep learning model
designed to classify chest X-ray/CT images into four categories:
- COVID-19
- Lung Opacity
- Normal
- Viral Pneumonia

The script supports:
- Training from scratch
- Fine-tuning from a pretrained model
- Periodic validation and model checkpointing
- Comprehensive logging of training progress
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


def train():
    """
    Train the model for one epoch.
    
    This function:
    1. Iterates through the training data loader
    2. Performs forward and backward passes
    3. Updates model weights
    4. Calculates and logs training metrics (loss, accuracy, per-class recall)
    
    The function maintains both batch-level metrics (for logging during the epoch)
    and epoch-level metrics (for final epoch summary).
    """
    loss_mean = 0.0  # Average loss for logging during the epoch
    loss_mean_epoch = 0.0  # Average loss for the entire epoch
    cm = np.zeros((args.num_classes, args.num_classes), dtype=int)  # Confusion matrix for logging
    cm_epoch = np.zeros((args.num_classes, args.num_classes), dtype=int)  # Confusion matrix for the entire epoch

    for i, data in enumerate(train_loader):

        # Forward pass
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # Backward pass
        optimizer.zero_grad()  # Zero the parameter gradients
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagate gradients

        # Update weights
        optimizer.step()

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update confusion matrices for accuracy calculation
        for y_pred, y_true in zip(predicted, labels):
            cm[y_true][y_pred] += 1  # Update batch-level confusion matrix
            cm_epoch[y_true][y_pred] += 1  # Update epoch-level confusion matrix

        # Accumulate loss values
        loss_mean += loss.item()
        loss_mean_epoch += loss.item()

        # Log training progress at regular intervals
        if (i + 1) % args.log_interval == 0:
            loss_mean = loss_mean / args.log_interval  # Calculate average loss
            train_acc = acc_from_cm(cm)  # Calculate accuracy metrics from confusion matrix
            
            # Log detailed training progress
            logger.info(
                "Train - Epoch: {}/{} Iter: {}/{} Loss: {:.4f} Acc_mean: {:.2%} Recall: ({:.2%}, {:.2%}, {:.2%}, {:.2%})".format(
                    epoch,
                    args.max_epoch,
                    i + 1,
                    len(train_loader),
                    loss_mean,
                    train_acc[-1],  # Overall accuracy
                    train_acc[0],   # COVID-19 recall
                    train_acc[1],   # Lung opacity recall
                    train_acc[2],   # Normal recall
                    train_acc[3],   # Viral pneumonia recall
                )
            )
            # Reset metrics for next logging interval
            loss_mean = 0.0
            cm = np.zeros((args.num_classes, args.num_classes), dtype=int)

    # Calculate and log epoch-level metrics
    loss_mean_epoch = loss_mean_epoch / len(train_loader)
    train_acc_epoch = acc_from_cm(cm_epoch)
    logger.info(
        "Finish training Epoch {}, Loss: {:.4f}, Acc_mean: {:.2%}, Recall: ({:.2%}, {:.2%}, {:.2%}, {:.2%})\n".format(
            epoch,
            loss_mean_epoch,
            train_acc_epoch[-1],  # Overall accuracy
            train_acc_epoch[0],   # COVID-19 recall
            train_acc_epoch[1],   # Lung opacity recall
            train_acc_epoch[2],   # Normal recall
            train_acc_epoch[3],   # Viral pneumonia recall
        )
    )


def eval(max_acc=None, reached=None, eval_only=None):
    """
    Evaluate the model on the validation dataset.
    
    This function:
    1. Sets the model to evaluation mode
    2. Iterates through the validation data loader
    3. Calculates validation metrics (loss, accuracy, per-class recall)
    4. Saves the best model checkpoint based on validation accuracy
    
    Args:
        max_acc (list, optional): List of maximum accuracies achieved so far
        reached (int, optional): Epoch at which maximum accuracy was reached
        eval_only (bool, optional): If True, only perform evaluation without saving checkpoints
        
    Returns:
        tuple: (max_acc, reached) - Updated maximum accuracies and epoch reached
               Only returned if eval_only is False
    """
    loss_val = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        cm = np.zeros((args.num_classes, args.num_classes), dtype=int)  # Confusion matrix
        
        # Iterate through validation data
        for j, data in enumerate(valid_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate validation loss
            loss_val += loss.item()

            # Update confusion matrix
            for y_pred, y_true in zip(predicted, labels):
                cm[y_true][y_pred] += 1

        # Calculate average validation loss and accuracy metrics
        loss_val = loss_val / len(valid_loader)
        val_acc = acc_from_cm(cm)

        # Log validation results
        logger.info(
            "Valid - Epoch: {}, Loss: {:.4f}, Acc_mean: {:.2%}, Recall: ({:.2%}, {:.2%}, {:.2%}, {:.2%})\n".format(
                epoch,
                loss_val,
                val_acc[-1],  # Overall accuracy
                val_acc[0],   # COVID-19 recall
                val_acc[1],   # Lung opacity recall
                val_acc[2],   # Normal recall
                val_acc[3],   # Viral pneumonia recall
            )
        )
        logger.info("cm:\n{}".format(cm))  # Log confusion matrix

        # Save best model checkpoint if not in evaluation-only mode
        if not eval_only:
            if val_acc[-1] > max_acc[-1]:  # If current accuracy is better than previous best
                max_acc = val_acc  # Update maximum accuracy
                reached = epoch    # Update epoch at which maximum was reached
                
                # Save model checkpoint
                torch.save(
                    {"args": args, "model": model.state_dict()},
                    os.path.join(args.save_dir, f"{args.backbone}_best.ckpt"),
                )
                logger.info(
                    "Saved checkpoint to {}.".format(
                        os.path.join(args.save_dir, f"{args.backbone}_best.ckpt")
                    )
                )

            return max_acc, reached


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

    # Define data transformations for training
    # These transformations include data augmentation techniques to improve model generalization
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to standard size
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),    # Random vertical flip
            transforms.RandomRotation(90),      # Random 90-degree rotation
            transforms.RandomRotation(270),     # Random 270-degree rotation
            transforms.RandomCrop((224, 224)),  # Random crop to model input size
            transforms.ToTensor(),              # Convert to tensor
        ]
    )

    # Define data transformations for validation (minimal preprocessing)
    valid_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),      # Resize to standard size
            transforms.RandomCrop((224, 224)),  # Center crop to model input size
            transforms.ToTensor(),              # Convert to tensor
        ]
    )

    # Create datasets
    train_data = MyDataset(
        data_dir=args.train_dir,
        train=True,
        transform=train_transform,
        device=args.device,
    )
    valid_data = MyDataset(
        data_dir=args.val_dir,
        train=False,
        transform=valid_transform,
        device=args.device,
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size)

    # Initialize model
    model = MyModel(args.num_classes, args.backbone)

    # Define loss function and optimizer
    criterion = F.cross_entropy  # Cross-entropy loss for multi-class classification
    # optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.85, weight_decay=1e-5
    )
    
    # Learning rate scheduler - gradually reduces learning rate
    lr_lambda = lambda epoch: 1.0 - pow((epoch / args.max_epoch), 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load pretrained model if specified
    if args.pretrain_model:
        ckpt = torch.load(args.pretrain_model, map_location="cpu")
        # Temporarily modify classifier to match pretrained model's output size
        model.backbone.classifier = torch.nn.Linear(1024, 15, bias=True)
        # Load weights, allowing for mismatches in the classifier layer
        model.load_state_dict(ckpt["model"], strict=False)
        # Restore classifier to target size
        model.backbone.classifier = torch.nn.Linear(1024, args.num_classes, bias=True)
        logger.info("Loaded pretrained weights from {}.".format(args.pretrain_model))

    # Evaluation-only mode
    if args.eval:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        logger.info("Loaded checkpoint from {}.".format(args.checkpoint))
        model.eval()
        epoch = 0
        eval(eval_only=True)
        os._exit(0)

    # Move model to device (GPU/CPU)
    model.to(device)
    logger.info(model)

    # Start training
    logger.info("Training start!\n")
    start = time.time()
    max_acc = [0.0 for i in range(args.num_classes)]  # Initialize maximum accuracy for each class
    reached = 0  # Epoch at which maximum accuracy was reached

    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        model.train()  # Set model to training mode
        train()        # Train for one epoch

        # Validate the model at regular intervals
        if epoch % args.val_interval == 0:
            model.eval()  # Set model to evaluation mode
            max_acc, reached = eval(max_acc, reached)

        # Save model checkpoint at regular intervals
        if epoch % args.save_interval == 0:
            torch.save(
                {"args": args, "model": model.state_dict()},
                os.path.join(args.save_dir, f"{args.backbone}_epoch_{epoch}.ckpt"),
            )
            logger.info(
                "Saved checkpoint to {}.".format(
                    os.path.join(args.save_dir, f"{args.backbone}_epoch_{epoch}.ckpt")
                )
            )

        # Update learning rate
        scheduler.step()

    # Log training summary
    logger.info(
        "Training finish, the time consumption of {} epochs is {}s\n".format(
            args.max_epoch, round(time.time() - start)
        )
    )
    logger.info(
        "Best validation accuracy {:.2%} reached at epoch {}".format(
            max(max_acc), reached
        )
    )
