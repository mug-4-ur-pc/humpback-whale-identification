from typing import List

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger, MLFlowLogger

import humpback_whale.src.utils as utils
from humpback_whale.src.data.module import HumpBackWhaleDataModule
from humpback_whale.src.model.identifier import IdentificationModel
from humpback_whale.src.setup import setup


def set_tracking(cfg: DictConfig):
    """
    Set tracking for MLflow.
    """
    if not cfg.log.debug:
        # utils.check_uncommitted_changes()

        if cfg.log.mlflow_uri:
            mlflow.set_tracking_uri(cfg.log.mlflow_uri)
        mlflow.set_experiment(cfg.log.experiment)

    return not cfg.log.debug


def check_if_already_active():
    if mlflow.active_run():
        print(
            f"Run with UUID {mlflow.active_run().info.run_id} is already active. Do you want to stop it? [Y/n]."
        )
        if input().strip() in ["Y", "y"]:
            mlflow.end_run()
        else:
            utils.print_abort(
                "Aborting. You can resume this run with config parameter 'resume=True'."
            )


def load_model(cfg: DictConfig):
    return IdentificationModel(cfg.model, cfg.train)


def load_data(cfg: DictConfig):
    setup(cfg.data)
    return HumpBackWhaleDataModule(cfg)


def load_logger(cfg: DictConfig, run_name: str):
    return MLFlowLogger(
        experiment_name=cfg.log.experiment,
        run_name=run_name,
        tracking_uri=cfg.log.mlflow_uri,
    )


def load_callbacks(callbacks_cfg: DictConfig, run_dir: str):
    callbacks = []
    for cb_cfg in callbacks_cfg.values():
        if "dirpath" in cb_cfg:
            cb_cfg.dirpath = run_dir
        callbacks.append(hydra.utils.instantiate(cb_cfg))
    return callbacks


def init_trainer(cfg: DictConfig, logger: Logger, callbacks: List[Callback]):
    use_gpu = torch.cuda.is_available() and cfg.hardware.devices not in (
        "cpu",
        None,
        [],
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        **({"devices": cfg.hardware.devices} if use_gpu else {}),
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )
    return trainer


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def train(cfg: DictConfig):
    """
    Train a model.

    Args:
        config : DictConfig - configuration of model, training, logging and overall experiment setup.
    """
    tracking = set_tracking(cfg)

    check_if_already_active()

    run_name = utils.get_run_name(cfg)
    run_dir = utils.get_run_dir(cfg, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.hardware.seed)
    torch.cuda.manual_seed_all(cfg.hardware.seed)
    pl.seed_everything(cfg.hardware.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.hardware.matmul)

    print(f"Starting train run: {run_name}")
    print("[0/4] Loading model...")
    model = load_model(cfg)

    print("[1/4] Loading data...")
    data = load_data(cfg)
    data.setup()

    print("[2/4] Loading trainer...")
    logger = load_logger(cfg, run_name) if tracking else False
    callbacks = load_callbacks(cfg.train.callbacks, str(run_dir))
    trainer = init_trainer(cfg, logger, callbacks)

    print("[3/4] Starting training...")
    if tracking:
        run_id = logger.run_id
    try:
        ckpt_path = run_dir / "last.ckpt"
        trainer.fit(model, data, ckpt_path=ckpt_path if ckpt_path.exists() else None)
    finally:
        if tracking:
            mlflow.log_artifact(run_dir, run_dir, run_id=run_id)

    print("[4/4] Finished successfully!")


if __name__ == "__main__":
    train()
