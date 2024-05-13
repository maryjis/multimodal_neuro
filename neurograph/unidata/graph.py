"""Base class for graph datasets (unimodal and modal)
Offers the same inteface for loading and saving processed data.
"""

import os.path as osp
import json
from shutil import rmtree
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader as pygDataLoader

from .augmentations import AdHocCorrMatrix
from .base_dataset import BaseDataset
from .utils import get_subj_ids_from_folds, prepare_graph

IndexType = slice | torch.Tensor | np.ndarray | Sequence


# pylint: disable=abstract-method
class BaseGraphDataset(BaseDataset, InMemoryDataset):
    """Provides common methods for all Graph Datasets
    (subclasses of InMemoryDataset)
    """

    abs_thr: Optional[float]
    pt_thr: Optional[float]
    _random_crop: bool
    # callable that computes conn matrix from timeseries
    compute_corr_matrix: Optional[Callable]

    roi_map: dict[int, str]

    loader_klass = pygDataLoader

    def load_files(self) -> None:
        """load preprocessed data and assign them to corresponding isntance attributes"""
        self.data, self.slices = torch.load(self.processed_paths[0])

        # load dataframes w/ subj_ids and targets
        self.target_df = pd.read_csv(self.processed_paths[3])
        self.y = self.target_df[self.target_col].values.reshape(-1)

        with open(self.processed_paths[1], encoding="utf-8") as f_ids:
            self.subj_ids = [l.rstrip() for l in f_ids.readlines()]

        # load cv splits
        with open(self.processed_paths[2], encoding="utf-8") as f_folds:
            self.folds = json.load(f_folds)

        with open(self.processed_paths[4], encoding="utf-8") as f_roimap:
            self.roi_map = json.load(f_roimap)

    def save_files(
        self,
        datalist,
        sel_targets,
        subj_ids,
        folds,
        roi_map,
    ) -> None:
        """Save processed data to `self.processed_dir`"""

        # collate DataList and save to disk
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])

        # save targets df as csv
        sel_targets.to_csv(self.processed_paths[3])

        # save subj_ids as a txt file
        with open(self.processed_paths[1], "w", encoding="utf-8") as f_subjids:
            f_subjids.write("\n".join([str(x) for x in subj_ids]))

        with open(self.processed_paths[2], "w", encoding="utf-8") as f_folds:
            json.dump(folds, f_folds)

        # save mapping (idx -> ROI name)
        with open(self.processed_paths[4], "w", encoding="utf-8") as f_roimap:
            json.dump(roi_map, f_roimap)

    @property
    def file_prefix(self) -> str:
        """Prefix used for loading processed data"""
        raise NotImplementedError  # pragma: no cover

    @property
    def processed_file_names(self) -> str:
        """A list of filenames of a processed dataset"""
        return [
            f"{self.file_prefix}_data.pt",
            f"{self.file_prefix}_subj_ids.txt",
            f"{self.file_prefix}_folds.json",
            f"{self.file_prefix}_targets.csv",
            f"{self.file_prefix}_roi_map.json",
        ]

    def get_num_nodes(self) -> int:
        """Get number of nodes, check that all graphs have the same number of nodes"""
        num_nodes = self.slices["x"].diff().unique()
        assert len(num_nodes) == 1, "You have different number of nodes in graphs!"

        return num_nodes.item()

    def get_subset(self, idx: list[int], train: bool = True) -> 'BaseGraphDataset':
        # pytorch geometric makes a shallow copy of a dataset
        subset = self[idx]
        subset.to_train() if train else subset.to_test()

        return subset


class UniGraphDataset(BaseGraphDataset):
    """Base class for every InMemoryDataset used in this project

    Expected directory structure (ABIDE, for example)
    with mapped class fields that store paths to those directories

    datasets/               # self.root
    datasets/abide          # self.experiments_dir
    ├── fmri                # self.root
    │   ├── processed
    │   └── raw             # self.raw_dir
    │       ├── aal         # self.cm_path
    │       └── schaefer
    ├── abide_splits.json
    ├── Phenotypic_V1_0b_preprocessed1.csv
    └── subject_ids.txt
    """

    # num_features: int (it's set in InMemoryDataset class)
    num_nodes: int

    init_node_features: str = "conn_profile"
    data_type: str = "graph"

    def __init__(
        self,
        root: str,
        atlas: str = "aal",
        experiment_type: str = "fmri",
        init_node_features: str = "conn_profile",  # not used rn
        abs_thr: Optional[float] = None,
        pt_thr: Optional[float] = None,
        no_cache=False,
        time_series_length: Optional[int] = None,
        random_crop: bool = False,
        random_crop_strategy: str = "uniform",  # or "per_roi",
        normalize: Optional[str] = None,  # 'global_max', 'log'
    ):
        """
        Args:
            root (str, optional): root dir where datasets should be saved
            atlas (str): atlas name
            thr (float, optional): threshold used for pruning edges #TODO
            no_cache (bool): if True, delete processed files and run processing from scratch
        """

        self.atlas = atlas
        self.experiment_type = experiment_type
        self.init_node_features = init_node_features
        self.abs_thr = abs_thr
        self.pt_thr = pt_thr
        self.normalize = normalize

        # copied from dense dataset
        self.time_series_length = time_series_length
        self.random_crop = random_crop
        self.random_crop_strategy = random_crop_strategy
        # swiched back and forth while training / testing
        self._random_crop = random_crop

        self._validate()

        # create transform object for performing augmentations
        self.compute_corr_matrix = (
            AdHocCorrMatrix(
                abs_thr=abs_thr,
                pt_thr=pt_thr,
                random_crop_strategy=random_crop_strategy,
                length=time_series_length,
            )
            if self.random_crop
            else None
        )

        # first, root points to `datasets` folder
        # then we change it a bit (add dataset name and experiment type)
        # root: experiment specific files (CMs and time series matrices)
        self.experiments_dir = osp.join(root, self.name)
        self.root = osp.join(self.experiments_dir, experiment_type)

        # global_dir: dir with meta info and cv_splits
        self.global_dir = osp.join(root, self.name)

        if no_cache:  # used for debug
            rmtree(self.processed_dir, ignore_errors=True)  # pragma: no cover

        super().__init__(self.root)
        # TODO: understand what is going on here weird bug; we need to call `_process` manually
        self._process()  # process raw files and save to disk

        self.load_files()  # lod processed files

        self.num_nodes = self.get_num_nodes()

    def process(self) -> None:
        """Load raw files, process and save processed files into `self.processed_dir`"""

        # extract subj_ids from splits file
        id_folds, _ = self.load_folds()
        self.subj_ids = get_subj_ids_from_folds(id_folds)

        # load and process raw data
        datalist, sel_targets, roi_map = self.load_datalist()

        # process folds (map subj_id to idx in datalist)
        folds = self.load_and_process_folds(self.subj_ids)

        # save everything to `self.processed_dir`
        self.save_files(
            datalist,
            sel_targets,
            self.subj_ids,
            folds,
            roi_map,
        )

    def load_datalist(self) -> tuple[list[Data], list[str], dict[int, str]]:
        """Load datalist, targets for `self.subj_ids`, mapping ROI -> node idx
        Implicitly uses `self.cm_path` and `self.subj_ids` for loading and processing data

        Raises:
            KeyError: _description_

        Returns:
            tuple[list[Data], list[str], dict[int, str]
        """

        # `self.load_cms` is inherited from a corresponding trait
        conn_matrices, time_series, roi_map = self.load_cms(self.cm_path)

        # load mapping subject_id -> label
        # (targets, label2idx, idx2label)
        targets, *_ = self._load_targets()

        # prepare data list from cms and targets
        datalist = []
        for subj_id in self.subj_ids:
            try:
                _ = targets.loc[subj_id]  # try to get a label for the subject
            except KeyError as exc:  # pragma: no cover
                msg = (
                    f"Subj_id {subj_id} not present in loaded targets "
                    "Check your json file w/ dataset splits"
                )
                raise KeyError(msg) from exc

            conn_matrix = conn_matrices[subj_id]
            # We don't have timeseries for PPMI so pass None
            timeseries = time_series.get(subj_id, None)

            datalist.append(
                prepare_graph(
                    conn_matrix,
                    timeseries,
                    subj_id,
                    targets,
                    self.abs_thr,
                    self.pt_thr,
                    self.normalize,
                ),
            )

        # select labels by subject ids
        sel_targets = targets.loc[self.subj_ids].copy()

        return datalist, sel_targets, roi_map

    def __getitem__(
        self,
        idx: int | np.integer | IndexType,
    ) -> Dataset | Data:
        r"""Adapted from `torch_geometric.data.Dataset`

        Method `get` is inherired from `InMemoryDataset`

        In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""

        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):

            data = self.get(self.indices()[idx])
            # we changed this part
            data = (
                data
                if self.compute_corr_matrix is None
                else self.compute_corr_matrix(data, self._random_crop)
            )

            return data
        else:
            return self.index_select(idx)

    def to_train(self):
        """See `BaseDataset` code"""
        self._random_crop = self.random_crop

    def to_test(self):
        """See `BaseDataset` code"""
        self._random_crop = False

    @property
    def cm_path(self):
        """Path to a dir w/ CM files"""
        # raw_dir specific to graph datasets :(
        return osp.join(self.raw_dir, self.atlas)

    @property
    def file_prefix(self) -> str:
        thr = ""
        if self.abs_thr:
            thr = f"abs={self.abs_thr}"
        if self.pt_thr:
            thr = f"pt={self.pt_thr}"

        return "_".join(
            s for s in [self.atlas, self.experiment_type, thr, self.normalize] if s
        )

    def _validate(self):
        if self.atlas not in self.available_atlases:  # pragma: no cover
            raise ValueError("Unknown atlas")
        if self.experiment_type not in self.available_experiments:  # pragma: no cover
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")
        if self.pt_thr is not None and self.abs_thr is not None:  # pragma: no cover
            raise ValueError(
                "Both proportional threshold `pt` and absolute threshold `thr` are not None!"
                " Choose one!"
            )

        if self.random_crop and self.time_series_length is None:  # pragma: no cover
            raise ValueError(
                "Set `time_series_length` if you want to random crop your timeseries"
            )

    def __repr__(self):  # pragma: no cover
        return (
            f"{self.__class__.__name__}:"
            f" atlas={self.atlas},"
            f" experiment_type={self.experiment_type},"
            f" pt_thr={self.pt_thr},"
            f" abs_thr={self.abs_thr},"
            f" size={len(self)}"
        )
