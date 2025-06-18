from typing import Any, Dict

import pytorch_lightning as pl
import timm
import torch
from omegaconf import DictConfig

from humpback_whale.src.model.arc_margin_product import ArcMarginProduct
from humpback_whale.src.model.mean_average_precision import FixedRetrievalMAP


class IdentificationModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.num_predictions = model_cfg.num_predictions
        self._init_model(model_cfg)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_map = FixedRetrievalMAP(top_k=self.num_predictions)
        self.val_map = FixedRetrievalMAP(top_k=self.num_predictions)

        self.train_cfg = train_cfg

    def _init_model(self, cfg: DictConfig):
        self.num_classes = cfg.num_classes

        self.backbone = timm.create_model(cfg.model, pretrained=cfg.pretrained)
        in_features = self.backbone.fc.out_features
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_features),
            torch.nn.Dropout(cfg.head_dropout),
            torch.nn.Linear(in_features, cfg.embedding_size),
            torch.nn.BatchNorm1d(cfg.embedding_size),
        )
        self.margin = ArcMarginProduct(
            cfg.embedding_size, cfg.num_classes, cfg.arc_margin
        )

    def forward(self, images, one_hot_label=None):
        x = self.backbone(images)
        x = self.head(x)
        if self.training:
            output = self.margin(x, one_hot_label)
        else:
            output = self.margin(x, None)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        one_hot_label = self._make_one_hot(y)
        logits = self(x, one_hot_label)

        loss = self.loss_fn(logits, y)
        self.train_map.update(logits, one_hot_label)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        one_hot_label = self._make_one_hot(y)
        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.val_map.update(logits, one_hot_label)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        mean_ap = self.train_map.compute()
        self.log("train_map", mean_ap, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        mean_ap = self.val_map.compute()
        self.log("val_map", mean_ap, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        logits = self(batch, None)
        output = torch.argsort(logits, dim=1)[..., -self.num_predictions :]

        return output

    def _make_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        one_hot_label = torch.zeros(
            labels.shape[0], self.num_classes, device=labels.device
        )
        one_hot_label.scatter_(1, labels.view(-1, 1).long(), 1)
        return one_hot_label

    def _load_element_class(
        self, element_config: DictConfig, possible_classes: Dict[str, Any]
    ):
        if element_config.name not in possible_classes:
            raise ValueError(
                f"Unknown element '{element_config.name}'. Parsing is not implemented"
            )
        element_class = possible_classes[element_config.name]
        element_params = dict(element_config)
        element_params.pop("name")
        return element_class, element_params

    def configure_optimizers(self):
        optimizers = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }

        optim_class, optim_params = self._load_element_class(
            self.train_cfg.optimizer, optimizers
        )
        optimizer = optim_class(self.parameters(), **optim_params)

        schedulers = {
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
            "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        scheduler_class, scheduler_params = self._load_element_class(
            self.train_cfg.scheduler, schedulers
        )
        scheduler = scheduler_class(optimizer, **scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
