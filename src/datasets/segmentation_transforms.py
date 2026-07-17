import albumentations as A


def get_train_transforms():

    return A.Compose(

        [

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.RandomBrightnessContrast(p=0.3),

        ]

    )


def get_val_transforms():

    return A.Compose([])