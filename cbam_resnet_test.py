import torch

from src.models.cbam_resnet18 import CBAMResNet18

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CBAMResNet18().to(device)

x = torch.randn(
    2,
    3,
    256,
    256
).to(device)

with torch.no_grad():

    y = model(x)

print()

print(model)

print()

print("Output shape:", y.shape)

print()

print("Parameters:")

print(
    f"{sum(p.numel() for p in model.parameters()):,}"
)