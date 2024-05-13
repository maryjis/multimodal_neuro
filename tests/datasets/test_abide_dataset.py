from functools import reduce

import numpy as np
import pytest
from neurograph.config import get_config
from neurograph.unidata.datasets.abide import ABIDEDenseDataset, ABIDEGraphDataset

from ..conftest import data_path


@pytest.fixture(scope="session")
def abide_ds_no_thr():
    return ABIDEGraphDataset(root=data_path, no_cache=True)


@pytest.fixture(scope="session")
def abide_ds_abs_thr():
    return ABIDEGraphDataset(root=data_path, abs_thr=0.3, no_cache=False)


@pytest.fixture(scope="session")
def abide_ds_pt_thr():
    return ABIDEGraphDataset(root=data_path, pt_thr=0.5, no_cache=False)


@pytest.fixture(scope="session")
def abide_dense_connprofile():
    return ABIDEDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="conn_profile",
    )


@pytest.fixture(scope="session")
def abide_dense_ts_notscaled():
    return ABIDEDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="timeseries",
        scale=False,
    )


@pytest.fixture(scope="session")
def abide_dense_ts_scaled():
    return ABIDEDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="timeseries",
        scale=True,
    )


@pytest.fixture(scope="session")
def abide_ds_schaefer_no_thr():
    return ABIDEGraphDataset(root=data_path, atlas="schaefer", no_cache=True)


@pytest.fixture(scope="session")
def abide_dense_ts_schaefer():
    return ABIDEDenseDataset(
        root=data_path,
        atlas="schaefer",
        experiment_type="fmri",
        feature_type="timeseries",
        time_series_length=78,
        scale=False,
    )


@pytest.fixture(scope="session")
def abide_dense_cp_schaefer():
    return ABIDEDenseDataset(
        root=data_path,
        atlas="schaefer",
        experiment_type="fmri",
        feature_type="conn_profile",
        scale=False,
    )


def test_abide_no_thr(abide_ds_no_thr):
    g = abide_ds_no_thr
    assert g[0].edge_index.shape[1] == abide_ds_no_thr.num_nodes**2


def test_abide_abs_thr(abide_ds_abs_thr):
    g = abide_ds_abs_thr[0]
    assert 0 < g.edge_index.shape[1] < abide_ds_abs_thr.num_nodes**2
    assert g.edge_attr.abs().min() >= 0.3


def test_abide_pt_thr(abide_ds_pt_thr):
    g = abide_ds_pt_thr[0]
    p = abide_ds_pt_thr.pt_thr * (abide_ds_pt_thr.num_nodes**2)
    assert p // 2 < g.edge_index.shape[1]
    assert g.edge_index.shape[1] <= int(p)


def test_abide_target(abide_ds_no_thr):
    target, label2idx, idx2label = abide_ds_no_thr._load_targets()
    target = target[abide_ds_no_thr.target_col]

    assert len(idx2label) == 2
    assert target.nunique() == 2
    assert target.isnull().sum() == 0

    assert target.index.isnull().sum() == 0


@pytest.mark.parametrize("ds", ["abide_ds_no_thr"])
def test_abide_folds(ds, request):
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


def test_abide_loaders(abide_ds_no_thr):
    def get_subj_from_loader(loader):
        ids = []
        for x in loader:
            ids.extend(x.subj_id)
        set_ids = set(ids)
        assert len(set_ids) == len(ids)
        return set_ids

    all_valids = []
    for split in abide_ds_no_thr.get_cv_loaders():
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


def test_abide_test_loader(abide_ds_no_thr):
    loader = abide_ds_no_thr.get_test_loader(8)
    for b in loader:
        assert b


def test_abide_dense_connprofile(abide_dense_connprofile):
    x, _ = abide_dense_connprofile[0]
    assert x.shape[0] == 200
    assert x.shape[1] == 200


def test_abide_dense_ts_scaled(abide_dense_ts_scaled):
    x, _ = abide_dense_ts_scaled[0]
    assert np.isclose(x.mean(), 0.0, rtol=1e-3)
    assert np.isclose(x.std(), 1.0, rtol=1e-3)


def test_abide_dense_ts_notscaled(abide_dense_ts_notscaled):
    x, _ = abide_dense_ts_notscaled[0]
    assert not np.isclose(x.mean(), 0.0, rtol=1e-3)
    assert not np.isclose(x.std(), 1.0, rtol=1e-3)


# Schaefer
def test_abide_graph_schaefer(abide_ds_schaefer_no_thr):
    assert abide_ds_schaefer_no_thr[0].x.shape == (400, 400)


def test_abide_dense_ts_schaefer(abide_dense_ts_schaefer):
    x, _ = abide_dense_ts_schaefer[0]
    assert x.shape == (400, 78)


def test_abide_dense_cp_schaefer(abide_dense_cp_schaefer):
    x, _ = abide_dense_cp_schaefer[0]
    assert x.shape == (400, 400)
