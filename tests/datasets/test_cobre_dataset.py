from functools import reduce
import itertools

import pytest
import torch

from neurograph.config import get_config
from neurograph.unidata.datasets.cobre import (
    CobreDenseDataset,
    CobreGraphDataset,
    CobreDenseTimeSeriesDataset,
)
from neurograph.multidata.datasets.cobre import (
    CobreMultimodalDense2Dataset,
    CobreMultimodalMorphDense2Dataset,
)

from ..conftest import data_path


@pytest.fixture(scope="session")
def cobre_ds_no_thr():
    return CobreGraphDataset(root=data_path, no_cache=True)


@pytest.fixture(scope="session")
def cobre_ds_abs_thr():
    return CobreGraphDataset(root=data_path, abs_thr=0.3, no_cache=False)


@pytest.fixture(scope="session")
def cobre_ds_pt_thr():
    return CobreGraphDataset(root=data_path, pt_thr=0.5, no_cache=False)


@pytest.fixture(scope="session")
def cobre_dense_ts():
    return CobreDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="timeseries",
    )


@pytest.fixture(scope="session")
def cobre_dense_connprofile():
    return CobreDenseDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="conn_profile",
    )


@pytest.fixture(scope="session")
def cobre_timeseries():
    return CobreDenseTimeSeriesDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        feature_type="timeseries",
    )


def test_cobre_no_thr(cobre_ds_no_thr):
    assert len(cobre_ds_no_thr.subj_ids) == len(cobre_ds_no_thr)
    g = cobre_ds_no_thr
    assert g[0].edge_index.shape[1] == cobre_ds_no_thr.num_nodes**2


def test_cobre_abs_thr(cobre_ds_abs_thr):
    assert len(cobre_ds_abs_thr.subj_ids) == len(cobre_ds_abs_thr)
    g = cobre_ds_abs_thr[0]
    assert 0 < g.edge_index.shape[1] < cobre_ds_abs_thr.num_nodes**2
    assert g.edge_attr.abs().min() >= 0.3


def test_cobre_pt_thr(cobre_ds_pt_thr):
    assert len(cobre_ds_pt_thr.subj_ids) == len(cobre_ds_pt_thr)
    g = cobre_ds_pt_thr[0]
    p = cobre_ds_pt_thr.pt_thr * (cobre_ds_pt_thr.num_nodes**2)
    assert p // 2 < g.edge_index.shape[1]
    assert g.edge_index.shape[1] <= int(p)


def test_cobre_target(cobre_ds_no_thr):
    target, label2idx, idx2label = cobre_ds_no_thr._load_targets()
    target = target[cobre_ds_no_thr.target_col]

    assert len(idx2label) == 2
    assert target.nunique() == 2
    assert target.isnull().sum() == 0

    assert target.index.isnull().sum() == 0


@pytest.mark.parametrize("ds", ["cobre_ds_no_thr", "cobre_dense_ts"])
def test_cobre_folds(ds, request):
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

    assert reduce(set.intersection, all_valids) == set(), "Non empty intersection between valids"
    assert set(folds["test"]) & all_train == set(), "Intersection between test and train"


def test_cobre_loaders(cobre_ds_no_thr):
    def get_subj_from_loader(loader):
        ids = []
        for x in loader:
            ids.extend(x.subj_id)
        set_ids = set(ids)
        assert len(set_ids) == len(ids)
        return set_ids

    all_valids = []
    for split in cobre_ds_no_thr.get_cv_loaders():
        # get loaders
        train, valid = split["train"], split["valid"]
        t_ids = get_subj_from_loader(train)
        v_ids = get_subj_from_loader(valid)

        assert t_ids & v_ids == set()
        assert train.dataset != valid.dataset

        all_valids.append(v_ids)

    assert reduce(set.intersection, all_valids) == set(), "Non empty intersection between valids"


def test_dense_loaders(cobre_dense_connprofile):
    for split in cobre_dense_connprofile.get_cv_loaders():
        # get loaders
        train, valid = split["train"], split["valid"]
        batch_1 = next(iter(train))
        batch_2 = next(iter(valid))

    # test loader
    test_loader = cobre_dense_connprofile.get_test_loader(8)
    for b in test_loader:
        assert b
        break


def test_cobre_test_loader(cobre_ds_no_thr):
    loader = cobre_ds_no_thr.get_test_loader(8)

    for b in loader:
        assert b
        break


# test DenseDataset
def test_cobre_dense_ts(cobre_dense_ts):
    assert len(cobre_dense_ts.subj_ids) == len(cobre_dense_ts)
    x, _ = cobre_dense_ts[0]
    assert x.shape[0] == 116
    assert x.shape[1] == 150


def test_cobre_dense_connprofile(cobre_dense_connprofile):
    assert len(cobre_dense_connprofile.subj_ids) == len(cobre_dense_connprofile)
    x, _ = cobre_dense_connprofile[0]
    assert x.shape[0] == 116
    assert x.shape[1] == 116


# test time_series dataset
@pytest.mark.parametrize(
    "fourier_timeseries",
    [True, False],
)
def test_cobre_timeseries_dataset(fourier_timeseries):
    ds = CobreDenseTimeSeriesDataset(
        root=data_path,
        atlas="aal",
        experiment_type="fmri",
        fourier_timeseries=fourier_timeseries,
    )

    sample = ds[0]
    ts_expected_shape = (116, 150 // 2) if fourier_timeseries else (116, 150)

    assert len(sample) == 3
    assert sample[0].shape == ts_expected_shape
    assert sample[1].shape == (116, 116)
    assert sample[2].shape == torch.Size([])


# test dense multimodal
@pytest.mark.parametrize(
    ["fmri_feature_type", "normalize"],
    itertools.product(["conn_profile", "timeseries"], ["global_max", "log"]),
)
def test_cobre_multimodal(fmri_feature_type, normalize):
    mm_cobre_dense = CobreMultimodalDense2Dataset(
        root=data_path,
        atlas="aal",
        fmri_feature_type=fmri_feature_type,
        normalize=normalize,
    )
    if fmri_feature_type == "conn_profile":
        fmri_feature_dim = 116
    else:  # timeseries
        fmri_feature_dim = 150

    dti_feature_dim = 116

    assert len(mm_cobre_dense) == 122

    assert mm_cobre_dense.data_fmri.shape[-1] == fmri_feature_dim
    assert mm_cobre_dense.data_dti.shape[-1] == dti_feature_dim

    fmri, dti, y = mm_cobre_dense[0]
    assert dti.shape[-1] == dti_feature_dim
    assert fmri.shape[-1] == fmri_feature_dim
    assert y.shape == torch.Size([])


def test_cobre_morph_multimodal():
    mm_cobre_dense = CobreMultimodalMorphDense2Dataset(
        root=data_path,
        atlas="aal",
        fmri_feature_type="timeseries",
        normalize="global_max",
    )

    fmri, vector, target = mm_cobre_dense[0]
    assert fmri is not None
    assert vector is not None
    assert target is not None
