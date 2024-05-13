""" Module responsible for initializing all the resources required for training:
    - Initialize wandb client
    - Initialize dataset instance
    - Initialize model, loss, optimizer, sheduler instances
"""

from omegaconf import OmegaConf
import wandb

from neurograph.config import Config, DatasetConfig
from neurograph.unidata.datasets import (
    dense_datasets,
    dense_ts_datasets,
    graph_datasets,
)
from neurograph.unidata.graph import UniGraphDataset

from neurograph.unidata.dense import UniDenseDataset, DenseTimeSeriesDataset
from neurograph.multidata.datasets import multimodal_dense_2,morph_multimodal_dense_2
from neurograph.multidata.graph import MultiGraphDataset


def dataset_factory(
    ds_cfg: DatasetConfig,
) -> UniDenseDataset | UniGraphDataset | MultiGraphDataset | DenseTimeSeriesDataset:
    """Factory function that returns a dataset class instance based on the passed dataset config"""

    if ds_cfg.data_type == "graph":
        return graph_datasets[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            experiment_type=ds_cfg.experiment_type,
            pt_thr=ds_cfg.pt_thr,
            abs_thr=ds_cfg.abs_thr,
            normalize=ds_cfg.normalize,
            time_series_length=ds_cfg.time_series_length,
            random_crop=ds_cfg.random_crop,
            random_crop_strategy=ds_cfg.random_crop_strategy,
        )
    if ds_cfg.data_type == "dense":
        return dense_datasets[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            time_series_length=ds_cfg.time_series_length,
            experiment_type=ds_cfg.experiment_type,
            feature_type=ds_cfg.feature_type,
            random_crop=ds_cfg.random_crop,
            random_crop_strategy=ds_cfg.random_crop_strategy,
            scale=ds_cfg.scale,
            normalize=ds_cfg.normalize,
        )
    if ds_cfg.data_type == "DenseTimeSeriesDataset":
        return dense_ts_datasets[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            experiment_type=ds_cfg.experiment_type,
            fourier_timeseries=ds_cfg.fourier_timeseries,
        )
    if ds_cfg.data_type == "MultiGraphDataset":
        return MultiGraphDataset(
            root=str(ds_cfg.data_path),
            name=ds_cfg.name,
            atlas=ds_cfg.atlas,
            pt_thr=ds_cfg.pt_thr,
            abs_thr=ds_cfg.abs_thr,
            normalize=ds_cfg.normalize,
            fusion=ds_cfg.fusion,
        )
    if ds_cfg.data_type == "multimodal_dense_2":
        return multimodal_dense_2[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            fmri_feature_type=ds_cfg.fmri_feature_type,
            normalize=ds_cfg.normalize,
        )
        
    if ds_cfg.data_type == "morph_multimodal_dense_2":
        return morph_multimodal_dense_2[ds_cfg.name](
            root=str(ds_cfg.data_path),
            atlas=ds_cfg.atlas,
            fmri_feature_type=ds_cfg.fmri_feature_type,
            normalize=ds_cfg.normalize,
        )
    raise ValueError(
        "Unknown dataset data_type! Options: dense, graph, multimodel_dense_2"
    )


def init_wandb(cfg: Config):
    """Initialize wandb according to `Config.log`.
    Basically, a wrapper over `wandb.init`
    """

    cfg_dict = OmegaConf.to_container(cfg)
    if (
        cfg.log.wandb_mode != "disabled" and cfg.log.wandb_project is None
    ):  # pragma: no cover
        raise ValueError(
            "Please, explicitly specify wand project name or turn off wandb by setting `log.wandb_mode=disable`"
        )

    wandb.init(
        project=cfg.log.wandb_project,
        settings=wandb.Settings(start_method="thread"),
        config=cfg_dict,  # type: ignore
        mode=cfg.log.wandb_mode,
        name=cfg.log.wandb_name,
        entity=cfg.log.wandb_entity,
        group=cfg.log.wandb_group,
    )
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")
