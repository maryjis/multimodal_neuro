"""Dense datasets"""

from copy import copy
import os.path as osp
from typing import Optional

import scipy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as thDataLoader
from torch.utils.data import Dataset as thDataset
from torch.utils.data import Subset

from neurograph.unidata.augmentations import random_crop

from .base_dataset import BaseDataset
from .utils import get_subj_ids_from_folds, normalize_cm


class BaseDenseDataset(thDataset, BaseDataset):
    """Base class for all Unimodal dense datasets"""

    loader_klass = thDataLoader

    def __init__(
        self,
        root: str,
        atlas: str = "aal",
        experiment_type: str = "fmri",
    ):
        self.atlas = atlas
        self.experiment_type = experiment_type

        # root: experiment specific files (CMs and time series matrices)
        self.root = osp.join(root, self.name, experiment_type)
        # global_dir: dir with meta info and cv_splits
        self.global_dir = osp.join(root, self.name)
        # path to CM and time series
        self.cm_path = osp.join(self.root, "raw", self.atlas)
        self.train = True

    def get_subset(self, idx, train: bool = True):
        # shallow copy the current dataset (we don't want to copy all the data, only attributes)
        dataset = copy(self)
        dataset.to_train() if train else dataset.to_test()
        return Subset(dataset, idx)


class UniDenseDataset(BaseDenseDataset):
    """Base class for dense datasets

    If feature_type is timeseries and random_crop=True,
    then you must also specify time_series_length.

    If `random_crop=False` and `time_series_length` is not None,
    we crop `time_series_length` steps from the start
    """

    data_type: str = "dense"
    available_feature_types: set[str] = {"timeseries", "conn_profile"}

    def __init__(
        self,
        root: str,
        atlas: str = "aal",
        experiment_type: str = "fmri",
        feature_type: str = "timeseries",  # or 'conn_profile'
        # time series augmentation / processing
        time_series_length: Optional[int] = None,
        random_crop: bool = False,
        random_crop_strategy: str = "uniform",  # or "per_roi",
        scale: bool = False,
        # used for DTI
        normalize: Optional[str] = None,  # global_max
    ):
        super().__init__(root, atlas, experiment_type)

        self.feature_type = feature_type
        self.time_series_length = time_series_length
        self.random_crop = random_crop
        self.random_crop_strategy = random_crop_strategy
        # swiched back and forth while training / testing
        self._random_crop = random_crop
        self.scale = scale
        self.normalize = normalize

        self.scaler = StandardScaler()

        # extract subj_ids from splits
        id_folds, _ = self.load_folds()
        self.subj_ids = get_subj_ids_from_folds(id_folds)

        # load data implicitly using self.subj_ids
        self.data, self.y = self.load_data()
        self.y = self.y.reshape(-1)  # reshape to 1d tensor

        # load and process folds data w/ subj_ids
        self.folds = self.load_and_process_folds(self.subj_ids)

        self.num_features = (
            self.time_series_length
            if self.feature_type == "timeseries" and self.time_series_length is not None
            else self.data[0].shape[-1]
        )
        # used for concat pooling
        self.num_nodes = self.data[0].shape[0]

    def load_data(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Implicitly uses `self.subj_ids` for processing data.
        Loads connectivity matrices and time series by `self.load_cms`,
        then preprocess (either connectivity matrices or time series)
        and returns `data` and `y`

        Return:
            data: tensor
            y: tensor
        """
        conn_matrices, time_series, _ = self.load_cms(self.cm_path)
        targets, *_ = self.load_targets(self.global_dir)

        input_matrices = (
            conn_matrices if self.feature_type == "conn_profile" else time_series
        )
        return self.process_matrices(
            input_matrices,
            targets,
            self.subj_ids,
            self.feature_type,
            self.scale,
            self.normalize,
        )

    def process_matrices(
        self,
        matrix_dict: dict[str, np.ndarray],
        targets: pd.DataFrame,
        subj_ids: list[str],
        feature_type: str,
        scale: bool,
        normalize: Optional[str] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Construct data: list[tensor], labels: tensor and list of subject ids
        from the dict mapping subj_id to matrix (CM or time series).

        Args:
            matrix_dict (dict[str, np.ndarray]): mapping subj_id -> time series/connectivity matrix
            targets (pd.DataFrame): pd.DataFrame indexed by subject_id
            subj_ids (list[str]): list of subject ids
            feature_type (str): time series/connectivity matrix
            scale (bool): use StandartScaler
            normalize (str, optional): normalization method. Defaults to None.

        Raises:
            KeyError: if subject id is not present in `targets`

        Returns:
            tuple[list[torch.Tensor], torch.Tensor]:
                list w/ time series/conn_profile tensor,
                LongTensor w/ targets
        """
        datalist = []
        print(matrix_dict.keys())
        for subj_id in subj_ids:
            matrix = matrix_dict[subj_id]
            try:
                # try to get a label for the subject
                _ = targets.loc[subj_id]
            except KeyError as exc:  # pragma: no cover
                raise KeyError("CM subj_id not present in loaded targets") from exc

            # prepare connectivity_matrix
            if feature_type == "conn_profile" and normalize:
                matrix = normalize_cm(matrix, normalize)

            if feature_type == "timeseries" and scale:
                matrix = self.scaler.fit_transform(matrix)

            # NB: we transpose things here: (T, R) -> (R, T)
            datalist.append(torch.FloatTensor(matrix).t())

        # NB: we use LongTensor here
        y = torch.LongTensor(targets.loc[subj_ids].copy().values)

        return datalist, y

    def to_train(self):
        """See `BaseDataset` code"""
        self._random_crop = self.random_crop

    def to_test(self):
        """See `BaseDataset` code"""
        self._random_crop = False

    def __len__(self):
        return len(self.data)  # pragma: no cover

    def __getitem__(self, idx: int):
        if self.feature_type != "timeseries":
            return self.data[idx], self.y[idx]

        # crop timeseries randomly
        if self._random_crop and self.time_series_length is not None:
            timeseries = random_crop(
                self.data[idx], self.time_series_length, self.random_crop_strategy
            )

        # crop time series by predefined length from the beginning
        # if `self.time_series_length` is not None
        # ("not random crop" so to speak)
        elif self.time_series_length is not None:
            timeseries = self.data[idx][:, : self.time_series_length]
        else:  # no crop
            timeseries = self.data[idx]

        return timeseries, self.y[idx]

    def _validate(self):
        if self.feature_type not in self.available_feature_types:  # pragma: no cover
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        if self.feature_type == "conn_profile":  # pragma: no cover
            self.time_series_length = None

        if self.random_crop and self.time_series_length is None:  # pragma: no cover
            raise ValueError(
                "Set `time_series_length` if you want to random crop your timeseries"
            )


# one time gig, not used anymore
class DenseTimeSeriesDataset(BaseDenseDataset):
    """Dense dataset for time series data
    each item  is a pair (time seris, connectivity matrix)

    Used as input to PriorTransformer
    """

    data_type: str = "dense_ts"

    def __init__(
        self,
        root: str,
        atlas: str = "aal",
        experiment_type: str = "fmri",
        fourier_timeseries: bool = False,
    ):
        super().__init__(root, atlas, experiment_type)
        assert (
            experiment_type == "fmri"
        ), "experiment_type must be `fmri` for DenseTimeSeriesDataset"  # pragma: no cover

        # set extra fields
        self.fourier_timeseries = fourier_timeseries

        # extract subj_ids from splits
        id_folds, _ = self.load_folds()
        self.subj_ids = get_subj_ids_from_folds(id_folds)

        # run `load_data` to load all the tensors
        self.time_series, self.connectivity, self.y = self.load_data()
        # reshape targets to 1d tensor
        self.y = self.y.reshape(-1)

        # load and process folds data w/ subj_ids
        self.folds = self.load_and_process_folds(self.subj_ids)

        # we need these attrs for correct model instantiation
        self.num_features = self.time_series.shape[-1]
        self.num_nodes = self.time_series.shape[1]  # used for concat pooling

    def load_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implicitly uses `self.subj_ids` for processing data"""
        # `load_cms` comes from the corresponding trait
        connectivity_dict, timeseries_dict, _ = self.load_cms(self.cm_path)
        targets, *_ = self._load_targets()

        return self.prepare_data(
            timeseries_dict,
            connectivity_dict,
            targets,
            self.subj_ids,
            self.fourier_timeseries,
        )

    def prepare_data(
        self,
        timeseries_dict: dict[str, np.ndarray],
        connectivity_dict: dict[str, np.ndarray],
        targets: pd.DataFrame,
        subj_ids: list[str],
        fourier_timeseries: bool,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Prepare tensors w/ time series and connectivity matrices"""

        ts_list, conn_list = [], []
        for subj_id in subj_ids:
            try:
                # try to get a label for the subject
                _ = targets.loc[subj_id]
            except KeyError as exc:  # pragma: no cover
                raise KeyError("CM subj_id not present in loaded targets") from exc

            ts_matrix = timeseries_dict[subj_id]
            conn_matrix = connectivity_dict[subj_id]

            # prepare matrices
            ts_matrix = self.process_time_series(ts_matrix, fourier_timeseries)
            conn_matrix = self.process_connectivity(conn_matrix)

            # update datalists and subj_ids
            ts_list.append(torch.FloatTensor(ts_matrix).unsqueeze(0))
            conn_list.append(torch.FloatTensor(conn_matrix).unsqueeze(0))

        # NB: we use LongTensor here
        y = torch.LongTensor(targets.loc[subj_ids].copy().values)
        ts_data = torch.cat(ts_list, dim=0)
        conn_data = torch.cat(conn_list, dim=0)

        return ts_data, conn_data, y

    @staticmethod
    def process_time_series(
        matrix: np.ndarray,
        fourier_timeseries: bool,
    ) -> np.ndarray:
        """Transpose original time series matrix
        and, optionally, apply Discrete Fourier Transform
        (scipy.fft.rfft) to it
        """

        matrix = matrix.T
        if fourier_timeseries:
            matrix = scipy.fft.rfft(matrix, axis=1)
            # take absolute values
            # ignore the first column since it contains only zeros
            # for standartized time series
            matrix = np.abs(matrix[:, 1:])

            # stack real and imaginary parts
            # matrix = np.hstack([matrix[:, 1:].real, matrix[:, 1:].imag])

            # apply standartization
            matrix = (matrix - matrix.mean(axis=1, keepdims=True)) / matrix.std(
                axis=1, keepdims=True
            )

        return matrix

    @staticmethod
    def process_connectivity(matrix: np.ndarray) -> np.ndarray:
        """Process connectivity matrix (identity transform for now)"""
        return matrix

    def __len__(self):
        return self.time_series.shape[0]  # pragma: no cover

    def __getitem__(self, idx: int):
        return self.time_series[idx], self.connectivity[idx], self.y[idx]

