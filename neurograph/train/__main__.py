# pragma: no cover
""" Entrypoint for running training """

import os
import os.path as osp
import json
import logging
from pathlib import Path

import git
import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric import seed_everything
import wandb

import neurograph
from neurograph.config import Config, validate_config
from neurograph.train.initialize import dataset_factory, init_wandb
from neurograph.train.train import train

REPO_PATH = Path(neurograph.__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    """Entrypoint function for running training from CLI"""

    if cfg.check_commit:
        repo = git.Repo(REPO_PATH)
        if repo.is_dirty():
            raise RuntimeError("Commit your changes before running training!")

    seed_everything(cfg.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)

    validate_config(cfg)
    logger.info("Config: \n%s", OmegaConf.to_yaml(cfg))
    init_wandb(cfg)

    dataset = dataset_factory(cfg.dataset)
    logger.info(dataset)

    # set checkpoints dir
    checkpoint_path = Path(os.getcwd()) / "checkpoints"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    metrics = train(dataset, cfg, checkpoint_path)
    wandb.finish()

    logger.info("Results saved in: %s", os.getcwd())

    with open(
        osp.join(os.getcwd(), "metrics.json"), "w", encoding="utf-8"
    ) as f_metrics:
        json.dump(metrics, f_metrics)
    OmegaConf.save(cfg, osp.join(os.getcwd(), "config.yaml"))


# pylint: disable=no-value-for-parameter
main()
