import sys

import fire
import hydra

from humpback_whale.src.export import ModelExporter
from humpback_whale.src.infer import infer
from humpback_whale.src.setup import setup
from humpback_whale.src.train import train


def load_hydra_config(config_dir: str):
    """Load Hydra configuration from a YAML file.

    Args:
        config_dir (str): Path to the directory containing `main.yaml`.
    """
    with hydra.initialize(version_base=None, config_path=config_dir):
        return hydra.compose("main.yaml")


def setup_wrapper(config_dir: str = "../config"):
    """Download and prepare data using Hydra configuration.

    Args:
        config_dir (str): Configuration directory path (default: '../conf').
    """
    cfg = load_hydra_config(config_dir)
    setup(cfg.data)
    print("Data was processed successfully.")


def train_wrapper(config_dir: str = "../config"):
    """Train identification model.

    Args:
        config_dir (str): Configuration directory path (default: '../conf').
    """
    cfg = load_hydra_config(config_dir)
    train(cfg)


def export_wrapper(
    src: str,
    dst: str = None,
    format: str = "pt",
    config_dir: str = "../config",
    **kwargs,
):
    """Export trained model.

    Args:
        src: Model checkpoint.
        dst: Output path (default: None).
        format: Output format (pt, onnx, engine).
        config_dir (str): Configuration directory path (default: '../conf').
    """
    cfg = load_hydra_config(config_dir)
    exporter = ModelExporter(cfg)
    if format == "pt":
        exporter.to_pt(src, dst)
    elif format == "onnx":
        exporter.to_onnx(src, dst, **kwargs)
    elif format == "engine":
        exporter.to_engine(src, dst, **kwargs)
    else:
        raise ValueError(f"Unknown output format: {format}")


def main():
    """Humpback Whale Data Manager CLI.

    Commands:
        setup : Download and prepare data.
    """
    commands = {
        "setup": setup_wrapper,
        "train": train_wrapper,
        "export": export_wrapper,
        "infer": infer,
    }
    try:
        fire.Fire(commands)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
