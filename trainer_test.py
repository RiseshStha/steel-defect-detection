import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.unet_resnet18 import UNetResNet18
from src.losses.dice_loss import DiceBCELoss
from src.train.segmentation_trainer import (
    train_one_epoch,
    validate,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNetResNet18().to(device)

criterion = DiceBCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=device.type == "cuda"
)

images = torch.randn(
    8,
    3,
    256,
    256
)

masks = torch.randint(
    0,
    2,
    (
        8,
        4,
        256,
        256
    )
).float()

loader = DataLoader(
    TensorDataset(images, masks),
    batch_size=2
)

train_loss = train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
)

metrics = validate(
    model,
    loader,
    criterion,
    device,
)

print()

print("Train Loss:", train_loss)

print(metrics)