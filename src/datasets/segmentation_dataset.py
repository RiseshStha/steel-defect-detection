import os

import cv2

import numpy as np

import torch

from torch.utils.data import Dataset

from utils1 import rle_decode


class SegmentationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        train_csv,
        image_dir,
        transform=None,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.train_csv = train_csv

        self.image_dir = image_dir

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row.ImageId

        image = cv2.imread(
            os.path.join(
                self.image_dir,
                image_id
            )
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = np.zeros(
            (
                256,
                1600,
                4
            ),
            dtype=np.uint8
        )

        rows = self.train_csv[
            self.train_csv.ImageId == image_id
        ]

        for _, r in rows.iterrows():

            cls = int(r.ClassId) - 1

            mask[:, :, cls] = rle_decode(
                r.EncodedPixels
            )

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]

            mask = augmented["mask"]

        mask = torch.tensor(
            mask.transpose(2, 0, 1),
            dtype=torch.float32
        )

        return image, mask