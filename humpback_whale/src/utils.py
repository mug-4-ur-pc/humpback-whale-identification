import hashlib
from pathlib import Path

import git
import omegaconf
from omegaconf import DictConfig


def print_abort(message: str):
    """
    Print message and abort.
    """
    print(message)
    print("Aborting...")
    exit(1)


def get_current_commit():
    try:
        repo = git.Repo(search_parent_directories=True)
        commit = repo.head.commit
        return commit.hexsha
    except (git.InvalidGitRepositoryError, git.GitCommandNotFound, git.ValueError):
        return "nogit"


def get_config_hash(cfg: DictConfig):
    cfg_str = omegaconf.OmegaConf.to_yaml(cfg, sort_keys=True)
    return hashlib.md5(cfg_str.encode("utf-8")).hexdigest()


def get_run_name(cfg: DictConfig):
    if cfg.log.debug:
        return "debug"

    commit = get_current_commit()
    cfg_hash = get_config_hash(cfg)

    return f"{commit[:8]}-{cfg_hash[:8]}"


def get_run_dir(cfg: DictConfig, run_name: str):
    experiment_dir = Path(cfg.log.experiment_root_path) / cfg.log.experiment
    return experiment_dir / run_name


def check_uncommitted_changes():
    """
    Check if there are any uncommitted changes in the current git repository.
    Returns True if there are uncommitted changes, False otherwise.
    """
    try:
        repo = git.Repo(search_parent_directories=True)

        untracked_files = repo.untracked_files
        modified_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]

        has_changes = bool(untracked_files or modified_files or staged_files)

        if has_changes:
            print_abort(
                "There are uncommitted changes in the git repository. "
                "Please commit or stash them before running this script."
            )

    except git.InvalidGitRepositoryError:
        raise RuntimeError("Not a git repository")
