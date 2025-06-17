from typing import Optional

import pytorch_lightning as L
import torch
from omegaconf import DictConfig

from humpback_whale.src.data.dataset import HumpBackWhaleDataset
from humpback_whale.src.data.transforms import create_transforms
from humpback_whale.src.model.encoder import LabelEncoder


class HumpBackWhaleDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.batch_size = cfg.train.batch_size
        self.transforms = create_transforms(cfg.model.image_size, cfg.data.transforms)
        self.data_cfg = cfg.data

    def setup(self, stage: Optional[str] = None):
        self.label_encoder = LabelEncoder.load(self.data_cfg.unique_labels)

        if stage in ("fit", None):
            self.train_data = HumpBackWhaleDataset(
                self.data_cfg.train_img_path,
                self.label_encoder,
                self.data_cfg.train_labels,
                self.transforms["train"],
                train=True,
            )
            self.val_data = HumpBackWhaleDataset(
                self.data_cfg.train_img_path,
                self.label_encoder,
                self.data_cfg.train_labels,
                self.transforms["val"],
                train=False,
            )
        if stage in ("predict", None):
            self.predict_data = HumpBackWhaleDataset(
                self.data_cfg.test_img_path,
                self.label_encoder,
                transforms=self.transforms["test"],
            )

    def _create_dataloader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.data_cfg.loader.num_workers,
            pin_memory=self.data_cfg.loader.pin_memory,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_data, shuffle=self.data_cfg.loader.shuffle
        )

    def val_dataloader(self):
        return self._create_dataloader(self.val_data, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(self.predict_ds, shuffle=False)
