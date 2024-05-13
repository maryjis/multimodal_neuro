import torch

from .test_cobre_dataset import cobre_dense_ts


def get_dataptr(data: list[torch.Tensor]):
    return set([x.data_ptr() for x in data])


def test_subset_nocopy(cobre_dense_ts):
    """Check that we don't copy any data when creating subset for nested cv"""
    data_ptr = get_dataptr(cobre_dense_ts.data)

    subsets_prt = set()
    for fold in cobre_dense_ts.get_nested_loaders(batch_size=1, num_folds=4):
        train = fold["train"]
        valid = fold["valid"]
        test = fold["test"]

        subsets_prt |= get_dataptr(train.dataset.dataset.data)
        subsets_prt |= get_dataptr(valid.dataset.dataset.data)
        subsets_prt |= get_dataptr(test.dataset.dataset.data)

    assert data_ptr == subsets_prt
