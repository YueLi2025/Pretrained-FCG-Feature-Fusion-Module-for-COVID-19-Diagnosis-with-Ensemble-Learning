"""
Configuration Module for COVID-19 Classification

This module provides configuration utilities for the COVID-19 classification project:
1. Command-line argument parsing
2. Logging configuration
3. Random seed setting for reproducibility

These utilities ensure consistent configuration across training, validation, and testing.
"""

import logging
import argparse
import random
import numpy as np
import torch


def parser_config():
    """
    Parse command-line arguments for model training and evaluation.
    
    This function defines all configurable parameters for the COVID-19 classification
    model, including training hyperparameters, data paths, model architecture options,
    and evaluation settings.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="COVID-19 Classification Model Configuration")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training and evaluation")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--max_epoch", type=int, default=80, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (L2 penalty)")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging training progress (iterations)")
    parser.add_argument("--val_interval", type=int, default=1, help="Interval for validation (epochs)")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints (epochs)")
    
    # Data paths
    parser.add_argument("--train_dir", type=str, default="../datasets/covid19/train", help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default="../datasets/covid19/test", help="Path to validation/test data directory")
    
    # Model configuration
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes to classify (COVID-19, lung opacity, normal, viral pneumonia)")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Backbone architecture for the model")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_file", type=str, default="log.log", help="Filename for logging")
    parser.add_argument("--pretrain_model", type=str, default="", help="Path to pretrained model for fine-tuning")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for resuming training or evaluation")
    parser.add_argument("--transform", type=str, default="clahe", help="Image transformation method (clahe, gamma, etc.)")

    # Hardware configuration
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (cuda:0, cpu, etc.)")
    
    # Evaluation mode
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (no training)")

    args = parser.parse_args()
    return args


def logger_config(file_name=None):
    """
    Configure logging for the project.
    
    Sets up a logger that outputs to both console and a file (if specified).
    This ensures that training progress and results are properly recorded.
    
    Args:
        file_name (str, optional): Path to log file. If None, only console logging is enabled.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel("INFO")
    
    # Define log format
    basic_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(basic_format, date_format)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()  # output to console
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    
    # File handler for persistent logging
    if file_name:
        file_handler = logging.FileHandler(file_name, mode="w")  # output to file
        file_handler.setFormatter(formatter)
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)
        
    logger.addHandler(console_handler)

    return logger


def seed_config(seed=1):
    """
    Set random seeds for reproducibility.
    
    This function sets random seeds for Python's random module, NumPy, and PyTorch
    to ensure that experiments are reproducible across runs.
    
    Args:
        seed (int, optional): Random seed value. Defaults to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Set CUDA seed for GPU operations
    
    # You can uncomment the following for even more deterministic behavior
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
