from functools import reduce
import itertools

import pytest
import torch

from neurograph.config import get_config
from neurograph.unidata.datasets.hcp import HCPDenseDataset, HCPGraphDataset
from ..conftest import data_path


@pytest.fixture(scope="session")
def hcp_ds_no_thr():
    return HCPGraphDataset(root=data_path, atlas="shen", no_cache=True)


@pytest.fixture(scope="session")
def hcp_ds_abs_thr():
    return HCPGraphDataset(root=data_path, atlas="shen", abs_thr=0.3, no_cache=False)


@pytest.fixture(scope="session")
def hcp_ds_pt_thr():
    return HCPGraphDataset(root=data_path, atlas="shen", pt_thr=0.5, no_cache=False)


@pytest.fixture(scope="session")
def hcp_dense_ts():
    return HCPDenseDataset(
        root=data_path,
        atlas="shen",
        experiment_type="fmri",
        feature_type="timeseries",
        random_crop=True,
        time_series_length=165,
    )


@pytest.fixture(scope="session")
def hcp_dense_connprofile():
    return HCPDenseDataset(
        root=data_path,
        atlas="shen",
        experiment_type="fmri",
        feature_type="conn_profile",
    )


def test_hcp_no_thr(hcp_ds_no_thr):
    g = hcp_ds_no_thr
    assert g[0].edge_index.shape[1] == hcp_ds_no_thr.num_nodes**2


def test_hcp_abs_thr(hcp_ds_abs_thr):
    g = hcp_ds_abs_thr[0]
    assert 0 < g.edge_index.shape[1] < hcp_ds_abs_thr.num_nodes**2
    assert g.edge_attr.abs().min() >= 0.3


def test_hcp_pt_thr(hcp_ds_pt_thr):
    g = hcp_ds_pt_thr[0]
    p = hcp_ds_pt_thr.pt_thr * (hcp_ds_pt_thr.num_nodes**2)
    assert p // 2 < g.edge_index.shape[1]
    assert g.edge_index.shape[1] <= int(p)


def test_hcp_target(hcp_ds_no_thr):
    target, label2idx, idx2label = hcp_ds_no_thr._load_targets()
    target = target[hcp_ds_no_thr.target_col]

    assert len(idx2label) == 7
    assert target.nunique() == 7
    assert target.isnull().sum() == 0

    assert target.index.isnull().sum() == 0


@pytest.mark.parametrize("ds", ["hcp_ds_no_thr", "hcp_dense_ts"])
def test_hcp_folds(ds, request):
    # workaround for parameterizing tests w/ fixtures
    ds = request.getfixturevalue(ds)
    folds = ds.folds

    all_train = set()  # everything that we run cross-val on
    all_valids = []
    for i, fold in enumerate(folds["train"]):
        train, valid = fold["train"], fold["valid"]
        tset, vset = set(train), set(valid)

        assert len(tset) == len(train), f"Fold {i}: non unique idx in train"
        assert len(vset) == len(vset), f"Fold {i}: non unique idx in valid"

        assert tset & vset == set(), f"Fold {i}: intersection between train/valid"
        all_valids.append(vset)

        all_train |= tset
        all_train |= vset

    assert (
        reduce(set.intersection, all_valids) == set()
    ), "Non empty intersection between valids"
    assert (
        set(folds["test"]) & all_train == set()
    ), "Intersection between test and train"


def test_hcp_loaders(hcp_ds_no_thr):
    def get_subj_from_loader(loader):
        ids = []
        for x in loader:
            ids.extend(x.subj_id)
        set_ids = set(ids)
        assert len(set_ids) == len(ids)
        return set_ids

    all_valids = []
    for split in hcp_ds_no_thr.get_cv_loaders():
        # get loaders
        train, valid = split["train"], split["valid"]
        t_ids = get_subj_from_loader(train)
        v_ids = get_subj_from_loader(valid)

        assert t_ids & v_ids == set()
        assert train.dataset != valid.dataset

        all_valids.append(v_ids)

    assert (
        reduce(set.intersection, all_valids) == set()
    ), "Non empty intersection between valids"


def test_dense_loaders(hcp_dense_connprofile):
    for split in hcp_dense_connprofile.get_cv_loaders():
        # get loaders
        train, valid = split["train"], split["valid"]
        batch_1 = next(iter(train))
        batch_2 = next(iter(valid))

    # test loader
    test_loader = hcp_dense_connprofile.get_test_loader(8)
    for b in test_loader:
        assert b


def test_hcp_test_loader(hcp_ds_no_thr):
    loader = hcp_ds_no_thr.get_test_loader(8)
    for b in loader:
        assert b


def test_hcp_dense_ts(hcp_dense_ts):
    x, _ = hcp_dense_ts[0]
    assert x.shape[0] == 268
    assert x.shape[1] == 165


def test_hcp_dense_connprofile(hcp_dense_connprofile):
    assert hcp_dense_connprofile.data[0].shape[0] == 268
    assert hcp_dense_connprofile.data[0].shape[1] == 268
