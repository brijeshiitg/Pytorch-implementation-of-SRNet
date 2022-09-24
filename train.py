"""This module is use to train the Srnet model."""

import logging
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import dataset
from opts.options import arguments
from model.model import Srnet
from utils.utils import (
    latest_checkpoint,
    adjust_learning_rate,
    weights_init,
    saver,
)

opt = arguments()

logging.basicConfig(
    filename="training.log",
    format="%(asctime)s %(message)s",
    level=logging.DEBUG,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    train_data = dataset.DatasetLoad(
        opt.cover_path,
        opt.stego_path,
        opt.train_size,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=90),
                transforms.ToTensor(),
            ]
        ),
    )

    val_data = dataset.DatasetLoad(
        opt.valid_cover_path,
        opt.valid_stego_path,
        opt.val_size,
        transform=transforms.ToTensor(),
    )

    # Creating training and validation loader.
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        val_data, batch_size=opt.batch_size, shuffle=False
    )

    # model creation and initialization.
    model = Srnet()
    model.to(device)
    model = model.apply(weights_init)

    # Loss function and Optimizer
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    check_point = latest_checkpoint()
    if not check_point:
        START_EPOCH = 1
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        print("No checkpoints found!!, Retraining started... ")
    else:
        pth = opt.checkpoints_dir + "net_" + str(check_point) + ".pt"
        ckpt = torch.load(pth)
        START_EPOCH = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        print("Model Loaded from epoch " + str(START_EPOCH) + "..")

    for epoch in range(START_EPOCH, opt.num_epochs + 1):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        test_accuracy = []

        # Training
        model.train()
        st_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        for i, train_batch in enumerate(train_loader):
            images = torch.cat((train_batch["cover"], train_batch["stego"]), 0)
            labels = torch.cat(
                (train_batch["label"][0], train_batch["label"][1]), 0
            )
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            training_loss.append(loss.item())
            prediction = outputs.data.max(1)[1]
            accuracy = (
                prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
            )
            training_accuracy.append(accuracy.item())

            sys.stdout.write(
                f"\r Epoch:{epoch}/{opt.num_epochs}"
                f" Batch:{i+1}/{len(train_loader)}"
                f" Loss:{training_loss[-1]:.4f}"
                f" Acc:{training_accuracy[-1]:.2f}"
                f" LR:{optimizer.param_groups[0]['lr']:.4f}"
            )

        end_time = time.time()

        # Validation
        model.eval()
        with torch.no_grad():

            for i, val_batch in enumerate(valid_loader):
                images = torch.cat((val_batch["cover"], val_batch["stego"]), 0)
                labels = torch.cat(
                    (val_batch["label"][0], val_batch["label"][1]), 0
                )

                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)

                loss = loss_fn(outputs, labels)
                validation_loss.append(loss.item())
                prediction = outputs.data.max(1)[1]
                accuracy = (
                    prediction.eq(labels.data).sum()
                    * 100.0
                    / (labels.size()[0])
                )
                validation_accuracy.append(accuracy.item())

        avg_train_loss = sum(training_loss) / len(training_loss)
        avg_valid_loss = sum(validation_loss) / len(validation_loss)

        message = (
            f"Epoch: {epoch}. "
            f"Train Loss:{(sum(training_loss) / len(training_loss)):.5f}. "
            f"Valid Loss:{(sum(validation_loss) / len(validation_loss)):.5f}. "
            "Train"
            f" Acc:{(sum(training_accuracy) / len(training_accuracy)):.2f} "
            "Valid"
            f" Acc:{(sum(validation_accuracy) / len(validation_accuracy)):.2f} "
        )
        print("\n", message)

        logging.info(message)

        state = {
            "epoch": epoch,
            "opt": opt,
            "train_loss": sum(training_loss) / len(training_loss),
            "valid_loss": sum(validation_loss) / len(validation_loss),
            "train_accuracy": sum(training_accuracy) / len(training_accuracy),
            "valid_accuracy": sum(validation_accuracy)
            / len(validation_accuracy),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
        }

        saver(state, opt.checkpoints_dir, epoch)
