import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import os
import warnings

from dataset import MyDataset
from model import MyModel
from metrics import acc_from_cm
from config import parser_config, logger_config, seed_config

# Suppress warnings
warnings.filterwarnings('ignore')


def evaluate(model, valid_loader, criterion, device, logger, args):
    """Evaluates the model on the validation set and logs performance metrics."""
    
    model.eval()
    loss_val = 0.0
    cm = np.zeros((args.num_classes, args.num_classes), dtype=int)
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Move data to appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Compute AUC
            outputs_softmax = F.softmax(outputs, dim=1)  # Ensure softmax along class dimension
            all_labels.append(labels.cpu())
            all_preds.append(outputs_softmax.cpu())

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            loss_val += loss.item()

            for y_pred, y_true in zip(predicted, labels):
                cm[y_true, y_pred] += 1

    # Compute average loss
    avg_loss = loss_val / len(valid_loader)

    # Compute accuracy metrics
    val_acc = acc_from_cm(cm)

    # Compute AUC
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    auc_ovr = roc_auc_score(all_labels.numpy(), all_preds.numpy(), multi_class='ovr')
    auc_ovo = roc_auc_score(all_labels.numpy(), all_preds.numpy(), multi_class='ovo')

    # Log evaluation results
    logger.info(
        f"Valid - Loss: {avg_loss:.4f}, Acc_mean: {val_acc[-1]:.2%}, "
        f"AUC_ovr: {auc_ovr:.2%}, AUC_ovo: {auc_ovo:.2%}, Recall: {val_acc[:-1]}"
    )
    logger.info(f"Confusion Matrix:\n{cm}")


def main():
    """Main function to set up the evaluation process."""
    
    args = parser_config()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Set up logger and random seed
    logger = logger_config(os.path.join("logs", f"{args.backbone}_{args.log_file}"))
    seed_config(args.seed)

    # Determine device
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # Define validation data transformations
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])

    # Load validation dataset
    valid_data = MyDataset(
        data_dir=args.val_dir,
        train=False,
        transform=valid_transform,
        device=args.device,
    )
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size)

    # Initialize model
    model = MyModel(args.num_classes, args.backbone)

    # Define loss function
    criterion = F.cross_entropy

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    logger.info(f"Loaded checkpoint from {ckpt_path}.")
    
    # Evaluate the model
    evaluate(model, valid_loader, criterion, device, logger, args)


if __name__ == "__main__":
    main()
