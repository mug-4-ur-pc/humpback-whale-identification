from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from humpback_whale.src.model.encoder import LabelEncoder


def create_label_encoder(df: pd.DataFrame, data_cfg: DictConfig):
    label_encoder = LabelEncoder.create(df.Id, data_cfg.unknown_label)
    label_encoder.save(data_cfg.unique_labels)


def train_val_split(df: pd.DataFrame, data_cfg: DictConfig):
    train_size = int(df.shape[0] * data_cfg.train_val_split_ratio)

    rng = np.random.default_rng(data_cfg.split_seed)
    random_vals = rng.random(df.shape[0])
    threshold = np.sort(random_vals)[train_size]
    is_train = random_vals < threshold

    df["Train"] = is_train

    df.to_csv(data_cfg.train_labels)


def data_exists(data_cfg: DictConfig):
    key_paths = data_cfg.train_img_path, data_cfg.train_labels, data_cfg.test_img_path
    return all(Path(p).exists() for p in key_paths)


def load_data(data_cfg: DictConfig):
    if not data_exists(data_cfg):
        fs = dvc.api.DVCFileSystem(".")
        fs.get(data_cfg.root, str(Path(data_cfg.root).parent), recursive=True)


def setup(data_cfg: DictConfig):
    load_data(data_cfg)
    df = pd.read_csv(data_cfg.train_labels)
    create_label_encoder(df, data_cfg)
    train_val_split(df, data_cfg)
