from functools import reduce
import itertools

import pytest
import torch
from torch_geometric import seed_everything

from neurograph.config import get_config
from neurograph.unidata.augmentations import random_crop
from neurograph.unidata.datasets.cobre import (
    CobreDenseDataset,
    CobreGraphDataset,
)

from ..conftest import data_path


def test_crops_w_nested_cv():
    """check that we turn off random crops for"""
    ds = CobreDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="timeseries",
        random_crop=True,
        time_series_length=5,
    )

    # replace data w/ synthetic one
    ds.data = [torch.arange(10).unsqueeze(0) for _ in range(16)]
    ds.y = torch.arange(4).tile(4)

    for fold in ds.get_nested_loaders(batch_size=1, num_folds=4):
        train = fold["train"]
        valid = fold["valid"]
        test = fold["test"]

        assert train.dataset.dataset._random_crop == True
        assert valid.dataset.dataset._random_crop == False
        assert test.dataset.dataset._random_crop == False

        valid_batch, _ = next(iter(valid))
        test_batch, _ = next(iter(test))

        # check that first elements are different for tensors from all train batches
        assert torch.unique(
            torch.cat([x.squeeze(0) for x, _ in train], dim=0)[:, 0]
        ).size() != torch.Size([1])
        assert torch.all(test_batch == torch.tensor([[[0, 1, 2, 3, 4]]]))
        assert torch.all(valid_batch == torch.tensor([[[0, 1, 2, 3, 4]]]))


def test_crop_graph_nested_cv():
    ds = CobreGraphDataset(root=data_path, random_crop=True, time_series_length=100, pt_thr=0.8)
    for fold in ds.get_nested_loaders(batch_size=64, num_folds=3):
        train = fold["train"]
        valid = fold["valid"]
        test = fold["test"]

        # check that we switch `_random_crop` flag during test
        assert train.dataset._random_crop == True
        assert valid.dataset._random_crop == False
        assert test.dataset._random_crop == False

        # check that we don't crop randomly during test
        b_1 = next(iter(test))
        b_2 = next(iter(test))
        assert torch.all(b_1.x == b_2.x)
        assert torch.all(b_1.edge_index == b_2.edge_index)
        assert torch.all(b_1.edge_attr == b_2.edge_attr)

        b_1 = next(iter(valid))
        b_2 = next(iter(valid))
        assert torch.all(b_1.x == b_2.x)
        assert torch.all(b_1.edge_index == b_2.edge_index)
        assert torch.all(b_1.edge_attr == b_2.edge_attr)


def test_random_crop_graph():
    seed_everything(1381)
    pt_thr = 0.3
    ds = CobreGraphDataset(root=data_path, random_crop=True, time_series_length=100, pt_thr=pt_thr)

    g_1 = ds[0]
    g_2 = ds[0]
    assert torch.any(~torch.eq(g_1.x, g_2.x))
    assert g_1.x.shape == (g_1.num_nodes, g_1.num_nodes)

    ds.to_test()
    g_1 = ds[0]
    g_2 = ds[0]
    assert torch.equal(g_1.x, g_2.x)

    # check pt_thr
    assert g_1.num_nodes ** 2 > g_1.edge_index.size(-1)


def test_random_crop_is_turned_off():
    ds = CobreGraphDataset(root=data_path, random_crop=False, time_series_length=100, pt_thr=0.8)
    g_1 = ds[0]
    g_2 = ds[0]
    assert torch.equal(g_1.x, g_2.x)


def test_random_crop_length():
    ds = CobreDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="timeseries",
        random_crop=True,
        time_series_length=100,
    )

    for length in [1, 10, 50, 100]:
        ds.time_series_length = length
        for i in list(range(len(ds)))[::10]:
            x, _ = ds[i]
            assert x.shape[-1] == length


def test_random_crop_per_roi():
    seed_everything(1381)
    x = torch.Tensor(
        [
            [i for i in range(10)],
            [10 + i for i in range(10)],
            [100 + i for i in range(10)],
        ]
    )

    out = torch.Tensor(
        [
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [100.0, 101.0, 102.0, 103.0, 104.0],
        ]
    )
    assert torch.all(random_crop(x, 5, "per_roi") == out)
