""" Dataset classes for ABIDE dataset """

import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from ..graph import UniGraphDataset
from ..dense import UniDenseDataset


class ABIDETrait:
    """Common fields and methods for all ABIDE datasets"""

    atlas: str
    name = "abide"
    available_atlases = {"aal", "schaefer"}
    available_experiments = {"fmri", "dti"}
    splits_file = "abide_splits.json"
    target_file = "Phenotypic_V1_0b_preprocessed1.csv"
    subj_id_col = "SUB_ID"
    target_col = "DX_GROUP"
    con_matrix_suffix = "_cc200_correlation.mat"
    embed_sufix = "*.1D"
    schaefer_filename = "abide_schaefer_filtered.save"

    # global_dir: str  # just for type checks

    def load_cms(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:
        if self.atlas == "aal":
            return self._load_cms_aal(path)
        if self.atlas == "schaefer":
            return self._load_cms_schaefer(path)
        raise ValueError(
            f"Unnown atlas. Options {self.available_atlases}"
        )  # pragma: no cover"msdl",

    def _load_cms_schaefer(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:
        data = torch.load(Path(path) / self.schaefer_filename)
        return data["conn_matrices"], data["timeseries"], {}

    def _load_cms_aal(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:

        """Load connectivity matrices, fMRI time series
        and mapping node idx -> ROI name.

        Return:
            connectivity matrices, timeseries, ROI idx->name mapping
        """

        path = Path(path)

        data = {}
        time_series = {}
        # ROI names, extacted from CMs
        roi_map: dict[int, str] = {}

        for p in path.iterdir():
            if p.is_dir():
                name = p.name

                mat = loadmat(p / f"{name}{self.con_matrix_suffix}")
                values = mat["connectivity"].astype(np.float32)

                embed_name = list(p.glob(self.embed_sufix))[0]
                values_embed = (
                    pd.read_csv(embed_name, delimiter="\t").astype(np.float32).values
                )
                time_series[name] = values_embed
                data[name] = values

        return data, time_series, roi_map

    def load_targets(
        self,
        dataset_dir: str,
    ) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """Load and process *cobre* targets"""

        target = pd.read_csv(osp.join(dataset_dir, self.target_file))
        target = target[[self.subj_id_col, self.target_col]]
        target[self.subj_id_col] = target[self.subj_id_col].astype(str)
        # check that there are no different labels assigned to the same ID
        max_labels_per_id = (
            target.groupby(self.subj_id_col)[self.target_col].nunique().max()
        )
        assert max_labels_per_id == 1, "Diffrent targets assigned to the same id!"

        # remove duplicates by subj_id
        target.drop_duplicates(subset=[self.subj_id_col], inplace=True)
        # set subj_id as index
        target.set_index(self.subj_id_col, inplace=True)

        # label encoding
        label2idx: dict[str, int] = {
            x: i for i, x in enumerate(target[self.target_col].unique())
        }
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
        target[self.target_col] = target[self.target_col].map(label2idx)

        return target, label2idx, idx2label


# NB: trait must go first
# pylint: disable=too-many-ancestors
class ABIDEGraphDataset(ABIDETrait, UniGraphDataset):
    """Graph dataset for ABIDE dataset"""


class ABIDEDenseDataset(ABIDETrait, UniDenseDataset):
    """Dense dataset for ABIDE dataset"""
