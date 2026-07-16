import os
import time
import warnings
import argparse

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.transforms import (
    get_train_classification_transform,
    get_val_classification_transform,
)

from src.losses.focal_loss import FocalLoss
from src.models.resnet18_classifier import ResNet18Classifier
from src.models.cbam_resnet18 import CBAMResNet18

from src.train.trainer import train_one_epoch, validate
from src.train.callbacks import EarlyStopping, CheckpointSaver
from src.train.utils import seed_everything

warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

CONFIG = {

    "seed": 42,

    "epochs": 20,

    "batch_size": 32,

    "learning_rate": 1e-4,

    "patience": 5,

    "num_workers": 2,

    # Choose which model to train
    "model_type": "cbam",      # "baseline" or "cbam"

    # Automatically generated paths
    "checkpoint_path": None,
    "log_path": None,
}


CONFIG["checkpoint_path"] = (
    f"checkpoints/{CONFIG['model_type']}_seed{CONFIG['seed']}.pth"
)

CONFIG["log_path"] = (
    f"logs/{CONFIG['model_type']}_seed{CONFIG['seed']}.csv"
)


# ==========================================================
# Main
# ==========================================================

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=CONFIG["model_type"],
        choices=["baseline", "cbam"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"],
    )

    args = parser.parse_args()

    CONFIG["model_type"] = args.model
    CONFIG["seed"] = args.seed

    seed_everything(CONFIG["seed"])
    
    set_seed(CONFIG["seed"])


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device : {device}")

    if device.type == "cuda":

        print(torch.cuda.get_device_name(0))

    os.makedirs("checkpoints", exist_ok=True)

    os.makedirs("logs", exist_ok=True)

    # -----------------------------------------
    # Read CSV
    # -----------------------------------------

    train_df = pd.read_csv(
        "data/processed/train_split.csv"
    )

    val_df = pd.read_csv(
        "data/processed/val_split.csv"
    )

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    train_dataset = ClassificationDataset(
        train_df,
        "data/train_images",
        transform=get_train_classification_transform()
    )

    val_dataset = ClassificationDataset(
        val_df,
        "data/train_images",
        transform=get_val_classification_transform()
    )

    # -----------------------------------------
    # DataLoader
    # -----------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------

    if CONFIG["model_type"] == "baseline":

        model = ResNet18Classifier(
            num_classes=4
        )

    elif CONFIG["model_type"] == "cbam":

        model = CBAMResNet18(
            pretrained=True,
            num_classes=4
        )

    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")

    model = model.to(device)

    criterion = FocalLoss()

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = GradScaler(
        enabled=torch.cuda.is_available()
    )

    early_stopping = EarlyStopping(
        patience=CONFIG["patience"]
    )

    saver = CheckpointSaver(
        CONFIG["checkpoint_path"]
    )

    print("\nEverything initialized successfully.")

    print(f"Train Images : {len(train_dataset)}")

    print(f"Validation Images : {len(val_dataset)}")

    print(f"Batch Size : {CONFIG['batch_size']}")

    print(f"Epochs : {CONFIG['epochs']}")

    print(f"Learning Rate : {CONFIG['learning_rate']}")

    # -------------------------------------------------
    # CSV Logger
    # -------------------------------------------------

    import csv

    with open(CONFIG["log_path"], "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "precision",
            "recall",
            "f1_macro",
            "f1_micro",
            "roc_auc",
            "lr",
            "epoch_time"
        ])

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(CONFIG["epochs"]):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        print("=" * 60)

        train_loss, epoch_time = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        saver.save(model, val_loss)

        early_stopping(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Precision  : {metrics['precision']:.4f}")
        print(f"Recall     : {metrics['recall']:.4f}")
        print(f"Macro F1   : {metrics['f1_macro']:.4f}")
        print(f"Micro F1   : {metrics['f1_micro']:.4f}")
        print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")

        print(f"LR         : {lr:.6f}")
        print(f"Epoch Time : {epoch_time:.1f} sec")

        with open(CONFIG["log_path"], "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                metrics["precision"],
                metrics["recall"],
                metrics["f1_macro"],
                metrics["f1_micro"],
                metrics["roc_auc"],
                lr,
                epoch_time
            ])

        if early_stopping.stop:

            print("\nEarly stopping triggered.")
            break

    print("\nTraining completed successfully.")
    


if __name__ == "__main__":
    main()