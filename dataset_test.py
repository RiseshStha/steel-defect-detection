import pandas as pd

from src.datasets.classification_dataset import ClassificationDataset

from src.datasets.transforms import get_train_classification_transform

df = pd.read_csv(
    "data/processed/train_split.csv"
)

dataset = ClassificationDataset(
    df,
    "data/train_images",
    transform=get_train_classification_transform()
)

image, label = dataset[0]

print(image.shape)

print(label)