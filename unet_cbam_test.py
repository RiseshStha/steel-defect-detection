import torch

from src.models.unet_cbam import UNetCBAM

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNetCBAM().to(device)

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

print("Input :", x.shape)

print("Output:", y.shape)

print()

print(
    "Parameters:",
    format(
        sum(p.numel() for p in model.parameters()),
        ","
    )
)