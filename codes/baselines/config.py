import logging
import argparse
import random
import numpy as np
import torch
from typing import Optional


def parse_config() -> argparse.Namespace:
    """Parses command-line arguments for model training configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="COVID-19 Classification Training Configuration")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--max_epoch", type=int, default=80, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")

    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10, help="Log training metrics every N batches")
    parser.add_argument("--val_interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model checkpoint every N epochs")
    parser.add_argument("--log_file", type=str, default="log.log", help="Path to log file")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")

    # Dataset & Model settings
    parser.add_argument("--train_dir", type=str, default="../datasets/covid19/train", help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, default="../datasets/covid19/test", help="Path to validation dataset")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes in dataset")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Backbone model architecture")
    parser.add_argument("--pretrain_model", type=str, default="", help="Path to pre-trained model file")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint file")

    # Device settings
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run training on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--eval", action="store_true", help="Run model in evaluation mode")

    args = parser.parse_args()
    return args


def configure_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Configures and returns a logger for training.

    Args:
        log_file (Optional[str]): Path to log file. If None, logs only to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 1):
    """Sets the random seed for reproducibility.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    return logger


def seed_config(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
