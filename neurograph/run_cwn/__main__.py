""" Entrypoint for CWN experiments """

# always import it first!
# pylint: disable=wrong-import-order, unused-import
import graph_tool as gt

import os
import os.path as osp
import json
import logging

import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric import seed_everything
import wandb

from neurograph.config import Config
from .run_cwn import prepate_data, train


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    """Entrypoint function for running training from CLI"""

    seed_everything(cfg.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)

    # validate_config(cfg)

    logging.info("Config: \n%s", OmegaConf.to_yaml(cfg))

    cfg_dict = OmegaConf.to_container(cfg)
    wandb.init(
        project=cfg.log.wandb_project,
        settings=wandb.Settings(start_method="thread"),
        config=cfg_dict,  # type: ignore
        mode=cfg.log.wandb_mode,
        name=cfg.log.wandb_name,
        entity=cfg.log.wandb_entity,
    )
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")

    # load dataset and complex list
    dataset, complex_list = prepate_data(cfg.dataset)

    # run experiments
    metrics = train(complex_list, dataset, cfg)
    wandb.finish()

    logging.info("Results saved in: %s", os.getcwd())

    # save metrics and config
    with open(
        osp.join(os.getcwd(), "metrics.json"), "w", encoding="utf-8"
    ) as f_metrics:
        json.dump(metrics, f_metrics)

    OmegaConf.save(cfg, osp.join(os.getcwd(), "config.yaml"))


# pylint: disable=no-value-for-parameter
main()
