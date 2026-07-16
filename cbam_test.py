import torch

from src.models.cbam import CBAM

# -------------------------------------------------------
# Device
# -------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# -------------------------------------------------------
# Dummy Feature Map
# -------------------------------------------------------

x = torch.randn(
    2,
    64,
    56,
    56
).to(device)

# -------------------------------------------------------
# Model
# -------------------------------------------------------

cbam = CBAM(
    in_channels=64
).to(device)

# -------------------------------------------------------
# Forward Pass
# -------------------------------------------------------

with torch.no_grad():

    y = cbam(x)

print("\nInput shape :", x.shape)

print("Output shape:", y.shape)

print("\nShape identical:",
      x.shape == y.shape)

# -------------------------------------------------------
# Parameter Count
# -------------------------------------------------------

params = sum(
    p.numel()
    for p in cbam.parameters()
)

print("\nTotal Parameters:", params)

# -------------------------------------------------------
# Range Check
# -------------------------------------------------------

print("\nInput mean :", x.mean().item())

print("Output mean:", y.mean().item())

print("\nCBAM test completed successfully.")