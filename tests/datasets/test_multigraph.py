import pytest
from neurograph.multidata.graph import MultiGraphDataset

from ..conftest import data_path


@pytest.mark.parametrize("fusion", ["concat", "dti_binary_mask"])
def test_muligraph_dataset_no_thr(fusion: str):
    ds = MultiGraphDataset(
        root=data_path,
        name="cobre",
        atlas="aal",
        normalize="global_max",
        fusion=fusion,
        no_cache=True,
    )

    # subject id tests
    assert len(ds) == len(ds.subj_ids)
    assert ds.subj_ids == [g.subj_id for g in ds]

    # preprocessing tests
    g = ds[0]
    if fusion == "concat":
        assert g.x.shape[1] == 116 * 2
    else:
        assert g.x.shape[1] == 116
        assert g.edge_index.shape[1] < 116**2

    # folds test
    train_idx = set()
    for f in ds.folds["train"]:
        train_idx |= set(f["train"])
        train_idx |= set(f["valid"])

        assert set(f["train"]) & set(f["valid"]) == set()

    assert train_idx & set(ds.folds["test"]) == set()


@pytest.mark.parametrize("fusion", ["concat", "dti_binary_mask"])
def test_muligraph_dataset_pt_thr(fusion: str):
    pt_thr = 0.5
    commom_params = dict(
        root=data_path,
        name="cobre",
        atlas="aal",
        normalize="global_max",
        pt_thr=pt_thr,
        no_cache=True,
    )
    if fusion == "dti_binary_mask":
        with pytest.raises(ValueError):
            MultiGraphDataset(fusion=fusion, **commom_params)
    else:
        ds = MultiGraphDataset(fusion=fusion, **commom_params)
        g = ds[0]
        assert g.edge_index.shape[1] <= int(pt_thr * (116**2))
