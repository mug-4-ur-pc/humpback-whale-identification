import sys

import fire
from hydra import compose, initialize

from humpback_whale.src.setup import setup
from humpback_whale.src.train import train


def load_hydra_config(config_dir: str):
    """Load Hydra configuration from a YAML file.

    Args:
        config_dir (str): Path to the directory containing `main.yaml`.
    """
    with initialize(version_base=None, config_path=config_dir):
        return compose("main.yaml")


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


def main():
    """Humpback Whale Data Manager CLI.

    Commands:
        setup : Download and prepare data.
    """
    commands = {
        "setup": setup_wrapper,
        "train": train_wrapper,
    }
    try:
        fire.Fire(commands)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
