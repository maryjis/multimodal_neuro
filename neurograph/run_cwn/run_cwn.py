""" Module for running a few experiments w/ COBRE datasets and CWN model """

# always import it first!
# pylint: disable=wrong-import-order, unused-import
import graph_tool as gt

from copy import deepcopy
import json
import logging
import pickle
from typing import Any

from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from cwn.mp.models import SparseCIN
from cwn.data.utils import convert_graph_dataset_with_rings
from cwn.data.data_loading import DataLoader as cwnDataLoader

from neurograph.config import get_config
from neurograph.config.dataset import DEFAULT_DATA_PATH
from neurograph.unidata.datasets.cobre import CobreGraphDataset
from neurograph.train.train import (
    train_one_split,
    evaluate,
    get_log_msg,
    agg_fold_metrics,
    metrics_resistry,
)
from neurograph.models.available_modules import (
    available_optimizers,
    available_losses,
    available_schedulers,
)
from neurograph.config import CellularDatasetConfig

logger = logging.getLogger()


def prepate_data(cfg: CellularDatasetConfig):
    """Load graph data via CobreGraphDataset and construct a list of 2-complexes"""

    dataset = CobreGraphDataset(
        root=cfg.data_path,
        pt_thr=cfg.pt_thr,
        no_cache=False,
        experiment_type="fmri",
        atlas="aal",
    )
    datalist, *_ = dataset.load_datalist()

    complex_list, _, _ = convert_graph_dataset_with_rings(
        datalist,
        max_ring_size=cfg.max_ring_size,
        include_down_adj=False,
        init_edges=True,
        init_rings=True,
        n_jobs=cfg.n_jobs,
        top_pt_rings=cfg.top_pt_rings,
        ignore_edge_attr=True,  # ignores `edge_attr` when computing edge features
    )

    cycles = torch.unique(
        complex_list[1].cochains[2].boundary_index[1], return_counts=True
    )[1]
    n_cycles = torch.unique(cycles, return_counts=True)

    logging.info("Final number of cycles: %s", n_cycles)
    # import pdb; pdb.set_trace();

    return dataset, complex_list


def create_cv_loaders(complex_list, folds_idx, batch_size=16, max_dim=2):
    """Create dataloaders from a complex list"""

    train_loaders = []
    for split in folds_idx["train"]:
        train_idx = split["train"]
        valid_idx = split["valid"]

        train_list = [complex_list[i] for i in train_idx]
        valid_list = [complex_list[i] for i in valid_idx]

        train_loaders.append(
            {
                "train": cwnDataLoader(
                    train_list,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    max_dim=max_dim,
                ),
                "valid": cwnDataLoader(
                    valid_list,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    max_dim=max_dim,
                ),
            }
        )
    return train_loaders


def create_test_loader(complex_list, test_idx, batch_size=16, max_dim=2):
    """Create a test dataloader from a complex list"""
    test_list = [complex_list[idx] for idx in test_idx]
    return cwnDataLoader(
        test_list,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        max_dim=max_dim,
    )


def init_model(cfg):
    """initialize a default CIN model"""
    model_cfg = OmegaConf.to_container(cfg.model)
    model_cfg.pop("name")
    return SparseCIN(
        **model_cfg,
    )


def init_model_optim_loss(cfg):
    """Initialize model, optimizer, scheduler and loss function instances"""
    model = init_model(cfg)

    # set optimizer
    optimizer = available_optimizers[cfg.train.optim](
        model.parameters(), **cfg.train.optim_args if cfg.train.optim_args else {}
    )

    # set lr_scheduler
    scheduler = None
    if cfg.train.scheduler is not None:
        scheduler_params: dict[str, Any]
        if cfg.train.scheduler_args is not None:
            scheduler_params = dict(deepcopy(cfg.train.scheduler_args))
        else:
            scheduler_params = {}  # pragma: no cover
        if cfg.train.scheduler == "ReduceLROnPlateau":
            if cfg.train.scheduler_metric is not None:
                scheduler_params["mode"] = metrics_resistry[cfg.train.scheduler_metric]

        scheduler = available_schedulers[cfg.train.scheduler](
            optimizer, **scheduler_params
        )
    # set loss function
    loss_f = available_losses[cfg.train.loss](
        **cfg.train.loss_args if cfg.train.loss_args else {}
    )

    return model, optimizer, scheduler, loss_f


def train(
    complex_list,  # list of cell complexes
    dataset,  # neurograph dataset w/ all meta info
    cfg,  # standart config from neurograph
):
    """Run one experiment w/ CWN: run cross-validation, report metrics on valids and test"""

    logging.info("Model architecture:\n %s", init_model(cfg))

    # get test loader beforehand
    test_loader = create_test_loader(
        complex_list,
        dataset.folds["test"],
        batch_size=cfg.train.valid_batch_size,
    )

    # final metrics per fold
    valid_folds_metrics: list[dict[str, float]] = []
    test_folds_metrics: list[dict[str, float]] = []

    # create cwn loaders for each fold, using indices from neurograph dataset
    loaders_iter = create_cv_loaders(
        complex_list, dataset.folds, batch_size=cfg.train.batch_size
    )
    for fold_i, loaders in enumerate(loaders_iter):
        logging.info("Run training on fold: %s", {fold_i})

        # init model optimizer, (scheduler), loss_f
        model, optimizer, scheduler, loss_f = init_model_optim_loss(cfg)

        # train and return valid metrics on last epoch
        valid_metrics, best_model = train_one_split(
            model,
            loaders,
            optimizer,
            scheduler,
            loss_f=loss_f,
            device=cfg.train.device,
            fold_i=fold_i,
            cfg=cfg,
        )
        # eval on test
        test_metrics = evaluate(best_model, test_loader, loss_f, cfg)
        logging.info(get_log_msg("test", fold_i, None, test_metrics))

        # save valid and test metrics for each fold
        valid_folds_metrics.append(valid_metrics)
        test_folds_metrics.append(test_metrics)

        # just to be sure
        del model, best_model

    # aggregate valid and test metrics for all folds
    final_valid_metrics = agg_fold_metrics(valid_folds_metrics)
    final_test_metrics = agg_fold_metrics(test_folds_metrics)

    wandb.summary["final"] = {"valid": final_valid_metrics, "test": final_test_metrics}

    logging.info(
        "Valid metrics over folds: %s", json.dumps(final_valid_metrics, indent=2)
    )
    logging.info(
        "Test metrics over folds: %s", json.dumps(final_test_metrics, indent=2)
    )

    return {"valid": final_valid_metrics, "test": final_test_metrics}
