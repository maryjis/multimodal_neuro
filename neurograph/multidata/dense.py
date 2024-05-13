"""Class for multimodal dense dataset, based on unimodal dense dataset.
MutlimodalDense2Dataset combines two modalities: fMRI and DTI.
"""

import os.path as osp
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..unidata.dense import UniDenseDataset
from ..unidata.utils import get_subj_ids_from_folds, normalize_cm


# pylint: disable=too-many-instance-attributes
# We have two different modalities here, so the number of attribute is roughly doubled
class MutlimodalDense2Dataset(UniDenseDataset):
    """Returns TWO embeddings from fMRI and DTI, hence it's `MultimodalDense2Dataset`"""

    data_type: str = "multimodal_dense_2"  # this is "tag" used in config
    name: str  # comes from corresponding Trait

    # here we don't call super().__init__() since init logic is completely different
    # however, we inherit from the base class to reuse other methods
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        root: str,
        atlas: str = "aal",
        fmri_feature_type: str = "timeseries",  # or 'conn_profile'
        normalize: Optional[str] = None,  # global_max, log
    ):
        self.atlas = atlas
        self.fmri_feature_type = fmri_feature_type
        self.normalize = normalize  # we normalize only DTI data

        # TO DO: add support of cropping to multimodal dense dataset
        self.random_crop = False

        # root: experiment specific files (CMs and time series matrices)
        # NB: for multimodal dataset `root` and `global_dir` are equal,
        # it's required for compatibility w/ `NeuroDenseDataset`
        self.root = osp.join(root, self.name)
        # global_dir: dir with meta info and cv_splits; used to load targets
        self.global_dir = osp.join(root, self.name)

        # path to CM and time series
        self.cm_path_fmri = osp.join(self.root, "fmri", "raw", self.atlas)
        self.cm_path_dti = osp.join(self.root, "dti", "raw", self.atlas)

        # extract subj_ids from splits
        id_folds, _ = self.load_folds()
        self.subj_ids = get_subj_ids_from_folds(id_folds)

        # load and process folds data w/ subj_ids
        self.folds = self.load_and_process_folds(self.subj_ids)

        # load two datalists
        self.data_fmri, y_fmri = self.process(
            self.cm_path_fmri,
            self.subj_ids,
            self.fmri_feature_type,
        )
        # conn_profile is the only option for DTI
        self.data_dti, y_dti = self.process(
            self.cm_path_dti,
            self.subj_ids,
            "conn_profile",
        )
        self.y = y_fmri.reshape(-1)  # reshape to 1d tensor

        # TODO: check ROI MAPS
        assert (
            self.data_fmri.shape[0] == self.data_dti.shape[0]
        ), "Diffrent datalists lenght for modalities"
        assert (
            self.data_fmri.shape[0] == self.y.shape[0]
        ), "Number of targets is not equal to number of samples"
        assert torch.all(
            y_fmri == y_dti
        ), "Unequal targets for fmri and dti. Check splits json file"

        self.num_fmri_features = self.data_fmri.shape[-1]
        self.num_dti_features = self.data_dti.shape[-1]

        # used for concat pooling
        self.num_fmri_nodes = self.data_fmri.shape[1]
        self.num_dti_nodes = self.data_dti.shape[1]

    def process(
        self,
        cm_path,
        subj_ids: list[str],
        feature_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """Load CMs and targets and pack into batch of tensors"""

        # load_cms and load_target come from corresponding Trait
        conn_matrices, time_series, _ = self.load_cms(cm_path)
        targets, *_ = self._load_targets()

        if feature_type == "timeseries":
            return self.prepare_tensors(time_series, targets, subj_ids, self.normalize)
        if feature_type == "conn_profile":
            return self.prepare_tensors(
                conn_matrices, targets, subj_ids, self.normalize
            )
        raise ValueError(
            f"Unknown fMRI feature_type: {self.fmri_feature_type}"
        )  # pragma: no cover

    @staticmethod
    def prepare_tensors(
        matrix_dict: dict[str, np.ndarray],
        targets: pd.DataFrame,
        subj_ids: list[str],
        normalize: Optional[str] = None,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Transform connection matrices / time series dict into tensors

        Args:
            matrix_dict (dict[str, np.ndarray]): dict mapping
                subject_id to connectivity matrix or time series matrix
            targets (pd.DataFrame): One column DataFrame indexed by ``subject_id``
            subj_ids (list[str]): list of subject_ids
            normalize (str, optional): normalization method for DTI data
        """
        datalist = []
        for subj_id in subj_ids:
            try:
                # prepare connectivity_matrix
                matrix = matrix_dict[subj_id]
                matrix = normalize_cm(matrix, normalize)

                # convert matrix to tensor, append to datalist
                datalist.append(torch.FloatTensor(matrix).t().unsqueeze(0))
            except KeyError as exc:  # pragma: no cover
                raise KeyError("CM subj_id not present in loaded targets") from exc

        # NB: we use LongTensor here
        y = torch.LongTensor(targets.loc[subj_ids].copy().values)
        data = torch.cat(datalist, dim=0)
        # subj_ids returned for compatibility (for less refactoring later)
        return data, y

    def __len__(self):
        return self.data_fmri.shape[0]

    def __getitem__(self, idx: int):
        return self.data_fmri[idx], self.data_dti[idx], self.y[idx]



class MutlimodalDenseMorphDataset(UniDenseDataset):
    """Returns TWO embeddings from fMRI and T1 (morphometry), hence it's `MutlimodalDenseMorpDataset`"""

    data_type: str = "morph_multimodal_dense_2"  # this is "tag" used in config
    name: str  # comes from corresponding Trait

    # here we don't call super().__init__() since init logic is completely different
    # however, we inherit from the base class to reuse other methods
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        root: str,
        atlas: str = "aal",
        fmri_feature_type: str = "timeseries",  # or 'conn_profile'
        normalize: Optional[str] = None,  # global_max, log
    ):
        super().__init__(root, atlas, "fmri", fmri_feature_type, normalize=normalize)
        self.morph_data = self.load_morph_data()

        self.morph_num_features =self.morph_data[0].shape[0]


    def load_morph_data(self):
        morph_data =self.load_morphometry(self.root)

        return self.prepare_tensors(morph_data,  self.subj_ids, self.normalize)

    def prepare_tensors(
            self,
            vector_dict: dict[str, np.ndarray],
            subj_ids: list[str],
            normalize: Optional[str] = None,
        ) -> list[torch.Tensor]:
            """Transform moprph vector dict into list of tensors

            Args:
                vector_dict (dict[str, np.ndarray]): dict mapping
                    subject_id to vector
                subj_ids (list[str]): list of subject_ids
                normalize (str, optional): normalization method for DTI data
            """
            datalist = []
            for subj_id in subj_ids:
                try:
                    # prepare vector
                    vector = vector_dict[subj_id]
                    # vector to tensor, append to datalist
                    datalist.append(torch.FloatTensor(vector.astype(float)))
                except KeyError as exc:  # pragma: no cover
                    raise KeyError("Moprh vector subj_id not present in loaded targets") from exc

            return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        timeseries, y =super().__getitem__(idx)
        return timeseries, self.morph_data[idx], y
