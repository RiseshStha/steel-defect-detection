import torch

from src.metrics.segmentation_metrics import (
    dice_score,
    iou_score,
    pixel_accuracy,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print()

print("Dice :", dice_score(pred, target))

print("IoU  :", iou_score(pred, target))

print("Pixel Accuracy :", pixel_accuracy(pred, target))