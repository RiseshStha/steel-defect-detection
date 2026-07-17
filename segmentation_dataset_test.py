import pandas as pd

from src.datasets.segmentation_dataset import SteelSegmentationDataset
from src.datasets.segmentation_transforms import get_train_transforms

df = pd.read_csv("data/train.csv")

dataset = SteelSegmentationDataset(
    dataframe=df,
    image_dir="data/train_images",
    transform=get_train_transforms(),
)

image, mask = dataset[0]

print()

print("Image Shape :", image.shape)

print("Mask Shape  :", mask.shape)

print()

print("Mask dtype:", mask.dtype)

print("Mask max:", mask.max())

print("Mask min:", mask.min())

print()

print("Pixels per class:")

for c in range(4):

    print(
        f"Class {c+1}:",
        mask[c].sum().item()
    )