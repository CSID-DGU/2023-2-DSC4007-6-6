import albumentations as A
from albumentations.pytorch import ToTensorV2


class Augments:
    train_augments=A.Compose([
        ToTensorV2(p=1.0),
    ],p=1.)
    
    valid_augments=A.Compose([
        ToTensorV2(p=1.0),
    ],p=1.)
    