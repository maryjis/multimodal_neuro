import numpy as np
from neurograph.multidata.graph import MultiGraphDataset
import pytest

from ..conftest import data_path
from .test_cobre_dataset import cobre_ds_no_thr, cobre_ds_abs_thr, cobre_ds_pt_thr
from .test_abide_dataset import abide_ds_no_thr, abide_ds_abs_thr, abide_ds_pt_thr
from .test_ppmi_dataset import (
    ppmi_ds_no_thr,
    ppmi_ds_abs_thr,
    ppmi_ds_pt_thr,
)


def check_ys(ds):
    t = ds.target_df.copy()
    y_1 = t[ds.target_col].values.reshape(-1)

    t[ds.subj_id_col] = t[ds.subj_id_col].astype(str)
    y_2 = t.set_index(ds.subj_id_col).loc[ds.subj_ids].values.reshape(-1)

    assert np.all(y_1 == y_2)
    assert np.all(ds.y == y_2)
    assert len(ds.subj_ids) == len(ds.y)
    assert len(ds.subj_ids) == len(ds)


@pytest.mark.parametrize(
    "ds",
    [
        "cobre_ds_no_thr",
        "cobre_ds_abs_thr",
        "cobre_ds_pt_thr",
        "abide_ds_no_thr",
        "abide_ds_abs_thr",
        "abide_ds_pt_thr",
        "ppmi_ds_no_thr",
        "ppmi_ds_abs_thr",
        "ppmi_ds_pt_thr",
    ],
)
def test_unigraph_ds_y(ds, request):
    """test that y attr corresponds to real targets"""
    ds = request.getfixturevalue(ds)
    check_ys(ds)


@pytest.mark.parametrize("fusion", ["concat", "dti_binary_mask"])
def test_multigraph_ds_y(fusion):
    ds = MultiGraphDataset(
        root=data_path,
        name="cobre",
        atlas="aal",
        normalize="global_max",
        fusion=fusion,
        no_cache=True,
    )
    check_ys(ds)
