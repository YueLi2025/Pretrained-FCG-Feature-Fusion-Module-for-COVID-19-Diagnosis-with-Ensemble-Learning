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
import os
import torch.nn.functional as F


def train():
    loss_mean = 0.0
    loss_mean_epoch = 0.0
    cm = np.zeros((args.num_classes, args.num_classes), dtype=int)
    cm_epoch = np.zeros((args.num_classes, args.num_classes), dtype=int)

    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # results
        _, predicted = torch.max(outputs.data, 1)

        # calculate the accuracy of this training iteration
        for y_pred, y_true in zip(predicted, labels):
            cm[y_true][y_pred] += 1
            cm_epoch[y_true][y_pred] += 1

        loss_mean += loss.item()
        loss_mean_epoch += loss.item()

        if (i + 1) % args.log_interval == 0:
            loss_mean = loss_mean / args.log_interval
            train_acc = acc_from_cm(cm)
            logger.info(
                "Train - Epoch: {}/{} Iter: {}/{} Loss: {:.4f} Acc_mean: {:.2%}".format(
                    epoch,
                    args.max_epoch,
                    i + 1,
                    len(train_loader),
                    loss_mean,
                    train_acc[-1],
                )
            )
            loss_mean = 0.0
            cm = np.zeros((args.num_classes, args.num_classes), dtype=int)

    loss_mean_epoch = loss_mean_epoch / len(train_loader)
    train_acc_epoch = acc_from_cm(cm_epoch)
    logger.info(
        "Finish training Epoch {}, Loss: {:.4f}, Acc_mean: {:.2%}\n".format(
            epoch,
            loss_mean_epoch,
            train_acc_epoch[-1],
        )
    )


def eval(max_acc=None, reached=None):
    loss_val = 0.0
    with torch.no_grad():
        cm = np.zeros((args.num_classes, args.num_classes), dtype=int)
        for j, data in enumerate(valid_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            loss_val += loss.item()

            for y_pred, y_true in zip(predicted, labels):
                cm[y_true][y_pred] += 1

        loss_val = loss_val / len(valid_loader)
        val_acc = acc_from_cm(cm)

        logger.info(
            "Valid - Epoch: {}, Loss: {:.4f}, Acc_mean: {:.2%}\n".format(
                epoch,
                loss_val,
                val_acc[-1],
            )
        )
        logger.info("cm:\n{}".format(cm))

        if val_acc[-1] > max_acc[-1]:
            max_acc = val_acc
            reached = epoch
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
    args = parser_config()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger = logger_config(os.path.join("logs", f"{args.backbone}_{args.log_file}"))
    seed_config(args.seed)
    logger.info(args)
    logger.info("Random seed: {}".format(args.seed))
    if not args.device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Running on device: {}".format(device))

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomRotation(270),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_data = MyDataset(
        data_dir=args.data_dir,
        train=True,
        transform=train_transform,
        device=args.device,
        split_file='train_val_list.txt'
    )
    valid_data = MyDataset(
        data_dir=args.data_dir,
        train=False,
        transform=valid_transform,
        device=args.device,
        split_file='test_list.txt'
    )

    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size)

    model = MyModel(args.num_classes, args.backbone)

    criterion = F.cross_entropy
    # optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.85, weight_decay=1e-5
    )
    lr_lambda = lambda epoch: 1.0 - pow((epoch / args.max_epoch), 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.eval:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        logger.info("Loaded checkpoint from {}.".format(args.checkpoint))
        model.eval()
        eval()

    model.to(device)
    logger.info(model)

    logger.info("Training start!\n")
    start = time.time()
    max_acc = [0.0 for i in range(args.num_classes)]  # accuracys
    reached = 0  # which epoch reached the max accuracy

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        train()

        # validate the model
        if epoch % args.val_interval == 0:
            model.eval()
            max_acc, reached = eval(max_acc, reached)

        scheduler.step()

    logger.info(
        "Training finish, the time consumption of {} epochs is {}s\n".format(
            args.max_epoch, round(time.time() - start)
        )
    )
    logger.info(
        "The best validation accuracy is: {:.2%}, reached at epoch {}.\n".format(
            max_acc[-1], reached
        )
    )
