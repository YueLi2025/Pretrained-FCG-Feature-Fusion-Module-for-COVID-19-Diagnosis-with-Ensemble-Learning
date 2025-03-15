import logging
import argparse
import random
import numpy as np
import torch


def parser_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="../datasets/chestX-ray14/images")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_file", type=str, default="log.out")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()
    return args


def logger_config(file_name=None):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    basic_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(basic_format, date_format)
    console_handler = logging.StreamHandler()  # output to console
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    if file_name:
        file_handler = logging.FileHandler(file_name, mode="w")  # output to file
        file_handler.setFormatter(formatter)
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def seed_config(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
