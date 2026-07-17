import torch

from src.losses.dice_loss import (
    DiceLoss,
    DiceBCELoss
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

criterion1 = DiceLoss().to(device)

criterion2 = DiceBCELoss().to(device)

pred = torch.randn(
    2,
    4,
    256,
    256,
    device=device
)

target = torch.randint(
    0,
    2,
    (2, 4, 256, 256),
    device=device
).float()

loss1 = criterion1(pred, target)

loss2 = criterion2(pred, target)

print()

print("Dice Loss :", loss1.item())

print("Combined Loss :", loss2.item())