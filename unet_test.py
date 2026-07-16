import torch

from src.models.unet_resnet18 import UNetResNet18


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

model = UNetResNet18(
    num_classes=4,
    pretrained=True,
    use_cbam=True
).to(device)

x = torch.randn(
    2,
    3,
    256,
    256
).to(device)

with torch.no_grad():

    y = model(x)

print()

print("Input :", x.shape)

print("Output:", y.shape)

print()

params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters: {params:,}")