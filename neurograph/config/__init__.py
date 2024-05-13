""" Config module """

from hydra import compose, initialize
from omegaconf import OmegaConf
from .config import (
    BrainGATConfig,
    BrainGCNConfig,
    Config,
    MLPConfig,
    MLPlayer,
    ModelConfig,
    MultiModalTransformerConfig,
    PriorTransformerConfig,
    StandartGNNConfig,
    SparseCINConfig,
    TransformerConfig,
    TrainConfig,
    LogConfig,
    validate_config,
    TrainConfig,
    BolTConfig,
)
from .dataset import (
    CellularDatasetConfig,
    DatasetConfig,
    TimeSeriesDenseDatasetConfig,
    MultimodalDatasetConfig,
    MultiGraphDatasetConfig,
    UnimodalDatasetConfig,
)


def get_config(name: str = "config", overrides=[]) -> Config:
    """Get config instance by its name (get default config if not specified"""
    with initialize(version_base=None, config_path="."):
        cfg: Config = OmegaConf.structured(compose(name, overrides=overrides))
    return cfg
