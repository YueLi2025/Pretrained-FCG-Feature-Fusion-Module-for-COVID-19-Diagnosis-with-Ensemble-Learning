import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from dataset import MyDataset
from model import MyModel
from metrics import acc_from_cm
from config import parser_config, logger_config, seed_config


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, logger, args):
    """Trains the model for one epoch."""
    
    model.train()
    loss_mean, loss_mean_epoch = 0.0, 0.0
    cm, cm_epoch = np.zeros((args.num_classes, args.num_classes), dtype=int), np.zeros((args.num_classes, args.num_classes), dtype=int)
    
    for i, (inputs, labels) in enumerate(train_loader):
        # Move data to the appropriate device (CPU/GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy for this batch
        _, predicted = torch.max(outputs, 1)
        for y_pred, y_true in zip(predicted, labels):
            cm[y_true, y_pred] += 1
            cm_epoch[y_true, y_pred] += 1

        # Track loss
        loss_mean += loss.item()
        loss_mean_epoch += loss.item()

        # Logging every 'log_interval' iterations
        if (i + 1) % args.log_interval == 0:
            avg_loss = loss_mean / args.log_interval
            train_acc = acc_from_cm(cm)
            logger.info(
                f"Train - Epoch {epoch}/{args.max_epoch} | Iter {i+1}/{len(train_loader)} "
                f"| Loss: {avg_loss:.4f} | Acc: {train_acc[-1]:.2%} | Recall: {train_acc[:-1]}"
            )
            loss_mean = 0.0
            cm.fill(0)  # Reset confusion matrix for next interval

    # Epoch-wise metrics
    avg_epoch_loss = loss_mean_epoch / len(train_loader)
    train_acc_epoch = acc_from_cm(cm_epoch)
    logger.info(
        f"Finish Training - Epoch {epoch} | Loss: {avg_epoch_loss:.4f} | "
        f"Acc: {train_acc_epoch[-1]:.2%} | Recall: {train_acc_epoch[:-1]}\n"
    )


def evaluate(model, valid_loader, criterion, epoch, device, logger, args, max_acc=None, reached=None, eval_only=False):
    """Evaluates the model on the validation set and saves the best model checkpoint."""
    
    model.eval()
    loss_val = 0.0
    cm = np.zeros((args.num_classes, args.num_classes), dtype=int)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            for y_pred, y_true in zip(predicted, labels):
                cm[y_true, y_pred] += 1

            loss_val += loss.item()

    avg_loss = loss_val / len(valid_loader)
    val_acc = acc_from_cm(cm)

    logger.info(
        f"Valid - Epoch {epoch} | Loss: {avg_loss:.4f} | "
        f"Acc: {val_acc[-1]:.2%} | Recall: {val_acc[:-1]}\n"
    )
    logger.info(f"Confusion Matrix:\n{cm}")

    if not eval_only and val_acc[-1] > max_acc[-1]:
        max_acc = val_acc
        reached = epoch
        save_path = os.path.join(args.save_dir, f"{args.backbone}_best.ckpt")
        torch.save({"args": args, "model": model.state_dict()}, save_path)
        logger.info(f"Saved best checkpoint to {save_path}.")

    return max_acc, reached


def main():
    args = parser_config()
    
    # Create directories for saving models & logs
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize logger and random seed
    logger = logger_config(os.path.join("logs", f"{args.backbone}_{args.log_file}"))
    seed_config(args.seed)

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # Define Data Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([90, 270]),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])

    # Load Dataset
    train_data = MyDataset(args.train_dir, train=True, transform=train_transform, device=args.device)
    valid_data = MyDataset(args.val_dir, train=False, transform=valid_transform, device=args.device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    # Initialize Model
    model = MyModel(args.num_classes, args.backbone).to(device)

    # Define Loss Function and Optimizer
    criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.85, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - pow((epoch / args.max_epoch), 0.9))

    # Load Pretrained Model if provided
    if args.pretrain_model:
        ckpt = torch.load(args.pretrain_model, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        logger.info(f"Loaded pretrained weights from {args.pretrain_model}.")

    # Evaluation Mode
    if args.eval:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded checkpoint from {args.checkpoint}.")
        model.eval()
        evaluate(model, valid_loader, criterion, epoch=0, device=device, logger=logger, args=args, eval_only=True)
        os._exit(0)

    logger.info("Training started!\n")
    start_time = time.time()
    max_acc, reached = [0.0] * args.num_classes, 0  # Track best accuracy and epoch

    for epoch in range(1, args.max_epoch + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, logger, args)

        # Validate model periodically
        if epoch % args.val_interval == 0:
            max_acc, reached = evaluate(model, valid_loader, criterion, epoch, device, logger, args, max_acc, reached)

        # Save checkpoints periodically
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"{args.backbone}_epoch_{epoch}.ckpt")
            torch.save({"args": args, "model": model.state_dict()}, save_path)
            logger.info(f"Checkpoint saved: {save_path}")

        lr_scheduler.step()

    elapsed_time = round(time.time() - start_time)
    logger.info(f"Training complete! Time taken: {elapsed_time}s for {args.max_epoch} epochs.")
    logger.info(f"Best validation accuracy: {max_acc[-1]:.2%}, Recall: {max_acc[:-1]}, achieved at epoch {reached}.\n")


if __name__ == "__main__":
    main() 
