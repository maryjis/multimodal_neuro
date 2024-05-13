from neurograph.config import (
    MultimodalDatasetConfig,
    get_config,
)
from neurograph.train.initialize import dataset_factory
from neurograph.train.train import handle_batch
import pytest
import torch


def test_handle_batch():
    batch = [torch.tensor(0.0) for _ in range(4)]
    with pytest.raises(ValueError):
        handle_batch(batch, "cpu")


def test_dataset_factory():
    cfg = get_config()
    cfg.dataset = MultimodalDatasetConfig(data_type="multimodal_graph_2")
    with pytest.raises(ValueError):
        dataset_factory(cfg.dataset)
