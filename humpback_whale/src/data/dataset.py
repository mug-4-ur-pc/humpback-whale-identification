from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image

from humpback_whale.src.model.encoder import LabelEncoder


class HumpBackWhaleDataset(torch.utils.data.Dataset):
    """Humpback whale dataset."""

    def __init__(
        self,
        img_path: str,
        label_encoder: LabelEncoder,
        labels_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        train: bool = False,
    ):
        self.transforms = transforms
        if labels_path is None:
            self.img_paths = list(Path(img_path).iterdir())
        else:
            df = pd.read_csv(labels_path)
            correct_ids = df.Train.values == train
            self.img_paths = [Path(img_path) / p for p in df.Image.values[correct_ids]]
            self.labels = label_encoder.encode(df.Id.values[correct_ids])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img) if self.transforms else img

        if self.labels is None:
            return img
        else:
            return img, self.labels[idx]
