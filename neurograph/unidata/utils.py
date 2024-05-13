""" Utils functions for converting CM to graphs; generating dataset splits etc. """

from functools import wraps
from itertools import pairwise
from typing import Optional

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.data import Data


def square_check(func):
    """Checks that the first argument is a square 2D numpy array"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        first_arg = args[0]
        assert isinstance(first_arg, np.ndarray) or isinstance(
            first_arg, torch.Tensor
        ), "input matrix must be np.ndarray or torch.Tensor!"
        assert first_arg.ndim == 2, "input matrix must be 2d array!"
        assert first_arg.shape[0] == first_arg.shape[1], "input matrix must be square!"

        return func(*args, **kwargs)

    return wrapper


def normalize_cm(conn_matrix: np.ndarray, normalize_type: Optional[str] = None) -> np.ndarray:
    """Normalize weighted adjacency matrix
    e.g. apply log or divide by global matrix max weight
    """

    conn_matrix = conn_matrix.astype(np.float32)
    if normalize_type:
        if normalize_type == "global_max":
            conn_matrix = conn_matrix / conn_matrix.max()
        elif normalize_type == "log":
            # this line handles zeros in CM (sets log(0) to 0)
            conn_matrix = np.log(conn_matrix, where=0 < conn_matrix, out=0.0 * conn_matrix)
        else:  # pragma: no cover
            raise ValueError(f"Unknown `normalize` arg! Given {normalize_type}")

    return conn_matrix


# pylint: disable=too-many-arguments
def prepare_graph(
    conn_matrix: np.ndarray,
    timeseries: Optional[np.ndarray],
    subj_id: str,
    targets: pd.DataFrame,
    abs_thr: Optional[float] = None,
    pt_thr: Optional[float] = None,
    normalize=None,
) -> Data:

    """
    Args:
        matrix (np.ndarray): connectivity matrix or timeseries for one subject
        subj_id (str): subject_id
        abs_thr (float, optional): Absolute threshold for sparsification
        pt_thr (float, optional): Proportional threshold for sparsification (pt_thr must be (0, 1)
        Combine CM, subj_id and target to a pyg.Data object
        `targets` must be indexed by subj_id

    timeseries expected to have a shape (num timesteps, num rois)
    """

    conn_matrix = normalize_cm(conn_matrix, normalize)

    # convert CM edge_index, edge_attr (and sparsify if thr are given)
    edge_index, edge_attr = conn_matrix_to_edges(conn_matrix, abs_thr=abs_thr, pt_thr=pt_thr)

    # compute initial node embeddings -> just original weights
    x = torch.from_numpy(conn_matrix).float()

    # get labels from DF via subject_id
    y = torch.LongTensor(targets.loc[subj_id].values)

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=x,
        timeseries=torch.from_numpy(timeseries).float() if timeseries is not None else None,
        num_nodes=conn_matrix.shape[0],
        y=y,
        subj_id=subj_id,
    )
    # data.validate()

    return data


@square_check
def conn_matrix_to_edges(
    conn_matrix: np.ndarray | torch.Tensor,
    abs_thr: Optional[float] = None,
    pt_thr: Optional[float] = None,
    remove_zeros: bool = False,
    zero_thr=1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:

    """Convert connectivity matrix (np.ndarray) into
    edge_index and edge_attr

    Args:
        conn_matrix (np.ndarray | torch.Tensor): square connectivity matrix
        abs_thr (Optional[float], optional): _description_. Defaults to None.
        pt_thr (Optional[float], optional): _description_. Defaults to None.
        remove_zeros (bool, optional): Remove edges with zero weights
        (only if abs_thr=None and pt_thr=None). Defaults to False.

    Raises:
        ValueError: if both `abs_thr` and `pt_thr` aren't None

    Returns:
        tuple[torch.Tensor, torch.Tensor]: edge_index and edge_attr
        for constructing a PyG.Data object
    """

    if isinstance(conn_matrix, torch.Tensor):  # convert tensor to float
        conn_matrix = conn_matrix.float()
    if not isinstance(conn_matrix, torch.Tensor):  # convert np.ndarray to tensor
        conn_matrix = torch.from_numpy(conn_matrix)

    assert torch.isnan(conn_matrix).sum() == 0, "NaNs in conn_matrix!"  # pragma: no cover

    if abs_thr is not None and pt_thr is not None:  # pragma: no cover
        raise ValueError("both `abs_thr` and `pt_thr` are not None! Choose one!")

    if abs_thr:
        idx = torch.nonzero(conn_matrix.abs() > abs_thr)
    elif pt_thr:
        assert 0 < pt_thr < 1, "thr must be in range (0, 1)"
        abs_cm = torch.abs(conn_matrix)
        vals = torch.sort(abs_cm.flatten(), descending=True, stable=True).values
        top_k = int(pt_thr * conn_matrix.shape[0] ** 2)  # pt * num_nodes**2
        idx = (abs_cm > vals[top_k]).nonzero()
    else:
        if remove_zeros:
            idx = (conn_matrix > zero_thr).nonzero()
        else:
            idx = (torch.isnan(conn_matrix) == 0).nonzero()

    edge_index = idx.T
    edge_weights = conn_matrix[idx[:, 0], idx[:, 1]].float()

    return edge_index, edge_weights


def generate_splits(
    subj_ids: list | np.ndarray,
    y: np.ndarray,
    seed: int = 1380,
):  # pragma: no cover
    """Generate dict with splits: first split to train/test, then
    split train into 5 folds w/ train and valid.
    """

    # split into train/test
    subj_ids = np.array(subj_ids)
    idx = np.arange(len(subj_ids))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        stratify=y,
        shuffle=True,
        random_state=seed,
    )

    train, y_train = subj_ids[train_idx], y[train_idx]
    test, _ = subj_ids[test_idx], y[test_idx]

    # split train into cv folds
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds: dict[str, list] = {"train": []}
    for _, (train_fold, valid_fold) in enumerate(strat_cv.split(train, y_train)):
        folds["train"].append(
            {
                "train": list(train[train_fold]),
                "valid": list(train[valid_fold]),
            }
        )
    folds["test"] = list(test)

    return folds


def get_subj_ids_from_folds(id_folds) -> list[str]:
    """Given a dict with splits (each subset is a list of subject ids),
    collect all subject ids into one list
    """
    subj_ids = []

    if set(id_folds.keys()) == set(["train", "test"]):
        train_folds = id_folds["train"]
    else:  # pragma: no cover
        # special case of old splits when we don't have 'train' key,
        # but have integer keys for each train fold
        train_idx = []
        for key in id_folds.keys():
            try:
                _ = int(key)
            except ValueError:
                continue
            train_idx.append(key)
        train_folds = [id_folds[i] for i in train_idx]

    for fold in train_folds:
        train_ids, valid_ids = fold["train"], fold["valid"]
        subj_ids.extend(train_ids)
        subj_ids.extend(valid_ids)

    subj_ids.extend(id_folds["test"])

    return sorted(list(set(subj_ids)))


def get_nested_splits(
    idx: np.ndarray,
    y: np.ndarray | list[int],
    num_folds: int = 10,
    random_state: int = 1380,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """This function generates nested cross-validation fold w/ num of outer_fold = `num_folds`,
    and num of inner fold = 1. Basically, it returns triples (train, valid, test),
    where valids and tests are disjoint. The union of all valids covers the whole dataset,
    as well as the union of all tests.
    """
    set_idx = set(idx)

    valid_splits = []
    strat_cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for _, valid_idx in strat_cv.split(idx, y):
        valid_splits.append(valid_idx)

    # now we need to assign some test splits to valid splits
    split_idx = list(range(num_folds))
    # [(0,1), (1,2), ..., (n-1, 0)]
    pairs = list(pairwise(split_idx)) + [(num_folds - 1, 0)]

    splits = []
    for val_i, test_i in pairs:
        valid_idx = valid_splits[val_i]
        test_idx = valid_splits[test_i]
        # substract valid and test from all idx to obtain train_idx
        train_idx = np.array(list(set_idx - set(valid_idx) - set(test_idx)))

        splits.append((train_idx, valid_idx, test_idx))

    return splits


# @square_check
# def find_thr(
#     conn_matrix: np.ndarray,
#     k: int = 5,
# ) -> float:

#     """ For a given CM find a threshold so after sparsification
#         the new CM will have `k * num_nodes` edges
#     """

#     n = conn_matrix.shape[0]
#     abs_cm = np.abs(conn_matrix)

#     # find thr to get the desired k
#     # = average number of edges for a node
#     vals = np.sort(abs_cm.ravel())
#     thr_idx = min(max(0, n**2 - 2*k*n - 1), n**2 - 1)
#     thr = vals[thr_idx]

# return thr
