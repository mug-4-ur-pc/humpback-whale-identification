from typing import Dict

import torchvision.transforms as T
from omegaconf import DictConfig


def create_transforms(img_size: int, transform_cfg: DictConfig) -> Dict[str, T.Compose]:
    common = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(transform_cfg.normalize_mean, transform_cfg.normalize_std),
        ]
    )
    train_augs = [
        T.RandomAffine(
            degrees=transform_cfg.degrees,
            scale=transform_cfg.scale,
        ),
        T.ColorJitter(
            brightness=transform_cfg.brightness,
            contrast=transform_cfg.contrast,
            saturation=transform_cfg.saturation,
        ),
    ]
    if transform_cfg.horizontal_flip:
        train_augs.append(T.RandomHorizontalFlip())
    if transform_cfg.vertical_flip:
        train_augs.append(T.RandomVerticalFlip())

    train_augs.extend([*common.transforms])

    return {"train": T.Compose(train_augs), "val": common, "test": common}


def create_infer_transforms(
    img_size: int, transform_cfg: DictConfig
) -> Dict[str, T.Compose]:
    common = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(transform_cfg.normalize_mean, transform_cfg.normalize_std),
        ]
    )
    return common
