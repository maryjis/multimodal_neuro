from typing import Optional
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from neurograph.unidata.utils import conn_matrix_to_edges


def random_crop(
    x: torch.Tensor,
    length: int,
    strategy: str = "uniform",
) -> torch.Tensor:
    """x is a 2D torch.Tensor with shape (R, T), R = num of ROIs, T = time series length"""

    if x.shape[-1] < length:  # pragma: no cover
        raise ValueError(
            f"Time series length {x.shape[-1]} is less than predifned lenght={length}"
        )

    if strategy == "uniform":
        # +1 since the upper bound is not included in `randint`
        sampling_init = np.random.randint(0, x.shape[-1] - length + 1)
        return x[:, sampling_init : sampling_init + length]
    elif strategy == "per_roi":
        masks = []
        for _ in range(x.shape[0]):
            # here upper bound is included
            start = random.randint(0, x.shape[1] - length)
            end = start + length
            masks.append((start, end))

        new_x = torch.zeros(x.shape[0], length, dtype=torch.float32)
        for i, (l, r) in enumerate(masks):
            new_x[i, :] = x[i, slice(l, r)]

        return new_x


class AdHocCorrMatrix(BaseTransform):
    def __init__(
        self,
        length: int,
        random_crop_strategy: str = "uniform",
        abs_thr: Optional[float] = None,
        pt_thr: Optional[float] = None,
    ):
        self.length = length
        self.strategy = random_crop_strategy
        self.abs_thr = abs_thr
        self.pt_thr = pt_thr

    def __call__(self, data: Data, is_random_crop: bool) -> Data:
        if not hasattr(data, "timeseries") or data.timeseries is None:  # pragma: no cover
            raise ValueError("Use `random_crop=True` while creating a dataset instance")

        # crop timeseries
        if is_random_crop:
            timeseries = random_crop(data.timeseries.T, self.length, self.strategy)
        else:
            timeseries = data.timeseries.T[:, : self.length]

        # update node features with a newly computed corr matrix
        data.x = torch.corrcoef(timeseries).float()

        # update edge_index and edge_attr
        data.edge_index, data.edge_attr = conn_matrix_to_edges(
            data.x,
            abs_thr=self.abs_thr,
            pt_thr=self.pt_thr,
            remove_zeros=False,
        )

        return data
