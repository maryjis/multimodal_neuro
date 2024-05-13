""" Module offers base classes for Graph and Dense datasets """

import os.path as osp
import json
import logging
from abc import ABC, abstractmethod

from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader as thDataLoader
from torch_geometric.loader import DataLoader as pygDataLoader

from .utils import get_nested_splits

logger = logging.getLogger(__name__)

LoaderT = pygDataLoader | thDataLoader


class BaseDataset(ABC):
    """Basic abstract class for all datasets in this project (both Unimodal and Multimodal)"""

    loader_klass: LoaderT  # type of the loader used by the dataset

    name: str
    atlas: str
    experiment_type: str
    available_atlases: set[str]
    available_experiments: set[str]

    root: str
    global_dir: str

    # filenames
    splits_file: str
    target_file: str

    # this field just mirrors one in dataset config
    data_type: str

    # metadata attrs
    folds: dict[str, Any]

    # needed for nested cross_validation
    subj_ids: list[str]
    y: np.ndarray

    # other resources
    target_df: pd.DataFrame

    # needed for type checks
    num_nodes: int
    num_features: int

    def load_and_process_folds(self, subj_ids: list[str]) -> dict[str, Any]:
        """After reading matrices and subject_ids, pass subject_ids here"""

        # load fold splits from json file
        id_folds, _ = self.load_folds()
        folds = self.process_folds(id_folds, subj_ids)

        return folds

    def load_folds(
        self,
        global_dir: Optional[str] = None,
        splits_file: Optional[str] = None,
    ) -> tuple[dict, int]:
        """Loads json w/ splits, returns a dict w/ splits and number of folds"""
        if global_dir is None:
            global_dir = self.global_dir
        if splits_file is None:
            splits_file = self.splits_file

        with open(osp.join(global_dir, splits_file), encoding="utf-8") as f_splits:
            _folds = json.load(f_splits)

        # for some splits we have a weird format of splits
        # so we need to do some extra steps
        folds = {}
        num_folds = -1
        for k, v in _folds.items():
            if k.isnumeric():  # pragma: no cover
                new_k = int(k)
                num_folds = max(num_folds, new_k)
                folds[new_k] = v
            else:
                folds[k] = v
        return folds, num_folds + 1

    def process_folds(self, id_folds: dict[str, Any], subj_ids: list[str]):
        """Process loaded raw folds (folds w/ subject_ids) into actual folds w/ data indices"""

        # map `subj_id` to `idx` in data_list
        id2idx = {s: i for i, s in enumerate(subj_ids)}
        logger.debug("id2idx: %s", str(id2idx))

        # map each `subj_id` to idx in `data_list` in folds
        folds: dict[str, Any] = {"train": []}
        if "train" in id_folds:
            for fold in id_folds["train"]:
                train_ids, valid_ids = fold["train"], fold["valid"]
                logger.debug("train_ids: %s", str(train_ids))

                one_fold = {
                    "train": [id2idx[subj_id] for subj_id in train_ids],
                    "valid": [id2idx[subj_id] for subj_id in valid_ids],
                }
                folds["train"].append(one_fold)
        else:  # pragma: no cover
            # special case of old splits for ABIDE
            train_idx = []
            for key in id_folds.keys():
                try:
                    _ = int(key)
                except ValueError:
                    continue
                train_idx.append(key)

            for fold_i in train_idx:
                train_ids, valid_ids = (
                    id_folds[fold_i]["train"],
                    id_folds[fold_i]["valid"],
                )
                logger.debug("train_ids: %s", str(train_ids))

                one_fold = {
                    "train": [id2idx[subj_id] for subj_id in train_ids],
                    "valid": [id2idx[subj_id] for subj_id in valid_ids],
                }
                folds["train"].append(one_fold)

        folds["test"] = [id2idx[subj_id] for subj_id in id_folds["test"]]

        return folds

    @abstractmethod
    def get_cv_loaders(
        self,
        batch_size=8,
        valid_batch_size=None,
    ) -> Generator[dict[str, LoaderT], None, None]:
        """Returns an generator that returns a dict
        {'train': loader, 'valid': loader'}
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_test_loader(self, batch_size: int):
        """Returns dataloader for the test part"""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def load_cms(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:
        """Return a dict of CMs with keys being subject ids,
        a dict with times series (if available),
        a dict with mapping ROI name to idx
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def load_targets(
        self,
        dataset_dir: str,
    ) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """Loads pandas dataframe w/ target, indexed by subj_ids. Also returns
        dict for label encoding and decoding (label to id, id to label)
        """
        raise NotImplementedError  # pragma: no cover

    
    def _load_targets(self):
        return self.load_targets(self.global_dir)

    def get_cv_loaders(
        self,
        batch_size: int = 8,
        valid_batch_size: Optional[int] = None,
    ) -> Generator[dict[str, thDataLoader], None, None]:

        valid_batch_size = valid_batch_size if valid_batch_size else batch_size
        for fold in self.folds["train"]:
            train_idx, valid_idx = fold["train"], fold["valid"]
            yield {
                "train": self.loader_klass(
                    self.get_subset(train_idx, True),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True
                ),
                "valid": self.loader_klass(
                    self.get_subset(valid_idx, False),
                    batch_size=valid_batch_size,
                    shuffle=False,
                ),
            }

    def get_test_loader(self, batch_size: int) -> thDataLoader:
        test_idx = self.folds["test"]
        return self.loader_klass(
            self.get_subset(test_idx, False),
            batch_size=batch_size,
            shuffle=False,
        )

    def get_nested_fold_idx(
        self,
        num_folds: int = 10,
        random_state: int = 1380,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:

        """Splits
        See: function  `get_nested_splits` below
        See: `tests/test_targets_subj_ids` for all the checks of datasets
        """

        assert (
            self.name != "ppmi"
        ), "PPMI is not supported (yet)! For PPMI we need to specify groups"
        # be sure that we deal w/ an original graph dataset, but not a subset
        if hasattr(self, "_indices"):
            assert self._indices is None, "Using subset of original dataset!"

        idx, y = np.arange(len(self)), self.y

        return get_nested_splits(
            idx,
            y,
            num_folds,
            random_state,
        )

    def get_nested_loaders(
        self,
        batch_size: int,
        num_folds: int = 10,
        random_state: int = 1380,
    ) -> Generator[dict[str, LoaderT], None, None]:

        """This method yields dict w/ train, valid, and test dataloaders.
        Nested cv folds are generated by `get_nested_splits` function
        (n outer folds, 1 inner fold).
        """

        for (train_idx, valid_idx, test_idx) in self.get_nested_fold_idx(
            num_folds, random_state
        ):
            yield {
                "train": self.loader_klass(
                    self.get_subset(train_idx, True),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True
                ),
                "valid": self.loader_klass(
                    self.get_subset(valid_idx, False),
                    batch_size=batch_size,
                    shuffle=False,
                ),
                "test": self.loader_klass(
                    self.get_subset(test_idx, False),
                    batch_size=batch_size,
                    shuffle=False,
                ),
            }

    def get_subset(self, idx: list[int], train: bool = True) -> "BaseDataset":
        """Return subset of the dataset, given a list of indices

        Args:
            idx (list[int]): sample indices to include into subset
            train (bool, optional): turn on / off augmentations during training.
                Defaults to True.

        Returns:
            BaseDataset: subset of the dataset given indices `idx`
        """
        raise NotImplementedError  # pragma: no cover

    def to_train(self):
        """Switch on data transformation specific for training
        (e.g. random crop)
        """

    def to_test(self):
        """Switch off data transformation specific for traiing
        (e.g. random crop)
        """
