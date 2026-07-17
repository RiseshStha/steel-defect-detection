import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.segmentation_dataset import SteelSegmentationDataset
from src.models.unet_resnet18 import UNetResNet18
from src.losses.dice_loss import DiceBCELoss
from src.train.segmentation_trainer import (
    train_one_epoch,
    validate,
)

# ==========================================================
# CONFIG
# ==========================================================

CONFIG = {

    "seed": 42,

    "epochs": 30,

    "batch_size": 4,

    "learning_rate": 1e-4,

    "num_workers": 2,

    "patience": 7,

    "model_type": "baseline",

    # change to "cbam" later

    "checkpoint_path":
        "checkpoints/segmentation_best.pth",

    "log_path":
        "logs/segmentation_training.csv"

}


# ==========================================================
# Seed
# ==========================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ==========================================================
# Main
# ==========================================================

def main():

    seed_everything(CONFIG["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()

    print("Using device :", device)

    if device.type == "cuda":

        print(torch.cuda.get_device_name(0))

    os.makedirs("checkpoints", exist_ok=True)

    os.makedirs("logs", exist_ok=True)

    train_df = pd.read_csv(
        "data/processed/train_split.csv"
    )

    val_df = pd.read_csv(
        "data/processed/val_split.csv"
    )

    train_dataset = SteelSegmentationDataset(
        dataframe=train_df,
        image_dir="data/train_images",
        annotation_file="data/train.csv",
        train=True,
    )

    val_dataset = SteelSegmentationDataset(
        dataframe=val_df,
        image_dir="data/train_images",
        annotation_file="data/train.csv",
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model = UNetResNet18().to(device)

    criterion = DiceBCELoss()

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    best_dice = 0.0

    patience_counter = 0

    history = []

    print()

    print("Train Images :", len(train_dataset))

    print("Validation Images :", len(val_dataset))

    print()

    for epoch in range(CONFIG["epochs"]):

        print("=" * 60)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")

        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(metrics["dice"])

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Val Loss   : {metrics['loss']:.4f}")

        print(f"Dice       : {metrics['dice']:.4f}")

        print(f"IoU        : {metrics['iou']:.4f}")

        print(f"Pixel Acc  : {metrics['pixel_acc']:.4f}")

        print(f"LR         : {optimizer.param_groups[0]['lr']:.6f}")

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_loss": metrics["loss"],

            "dice": metrics["dice"],

            "iou": metrics["iou"],

            "pixel_acc": metrics["pixel_acc"],

            "lr": optimizer.param_groups[0]["lr"]

        })

        if metrics["dice"] > best_dice:

            best_dice = metrics["dice"]

            torch.save(

                model.state_dict(),

                CONFIG["checkpoint_path"]

            )

            patience_counter = 0

            print("✓ Best model saved.")

        else:

            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:

            print()

            print("Early stopping triggered.")

            break

        print()

    pd.DataFrame(history).to_csv(

        CONFIG["log_path"],

        index=False

    )

    print()

    print("Training completed.")

    print(f"Best Dice : {best_dice:.4f}")


if __name__ == "__main__":

    main()