""" HCP dataset classes """
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# load base class to modify them w/ trait
from ..graph import UniGraphDataset
from ..dense import UniDenseDataset

logger = logging.getLogger(__name__)


class HCPTrait:
    """Common fields and methods for all HCP datasets"""

    name = "hcp"
    available_atlases = {"shen", "schaefer"}
    available_experiments = {"fmri", "dti"}
    splits_file = "hcp_splits.json"

    subj_id_col = "SUB_ID"
    fmri_id_col = "FILE_ID"
    target_col = "DX_GROUP"

    global_dir: str  # just for type checks

    def load_cms(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:
        """Load connectivity matrices, fMRI time series
        and mapping node idx -> ROI name.

        Maps sibj_id to CM and ts
        """
        logger.info("Load cms from HCP Trait: ")
        path = Path(path)

        data = {}
        time_series = {}
        # ROI names, extacted from CMs
        roi_map: dict[int, str] = {}

        for p in tqdm(path.glob("*.csv")):
            name = p.stem
            x = pd.read_csv(p)
            if "Node" in x.columns:
                x = x.drop("Node", axis=1)

            values = x.values.astype(np.float32)
            if p.stem.endswith("_embed"):
                time_series[name[:-6]] = values[:, 1:]
            else:
                data[name] = values
                if not roi_map:
                    roi_map = dict(enumerate(x.columns))

        return data, time_series, roi_map

    def load_targets(
        self,
        dataset_dir: str,
    ) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        """Load and process *hcp* targets"""
        logger.info("Load targets for HCP: ")
        path = Path(self.cm_path)

        targets = []
        for p in path.glob("*.csv"):
            if not p.stem.endswith("_embed"):
                array_splits = p.stem.split("_")
                targets.append((p.stem, array_splits[2]))

        target = pd.DataFrame.from_records(
            targets, columns=[self.fmri_id_col, self.target_col]
        )
        target.set_index(self.fmri_id_col, inplace=True)

        # label encoding
        label2idx: dict[str, int] = {
            x: i for i, x in enumerate(target[self.target_col].unique())
        }
        idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
        target[self.target_col] = target[self.target_col].map(label2idx)

        return target, label2idx, idx2label


# NB: trait must go first
# pylint: disable=too-many-ancestors
class HCPGraphDataset(HCPTrait, UniGraphDataset):
    """Graph dataset for HCP dataset"""


class HCPDenseDataset(HCPTrait, UniDenseDataset):
    """Dense dataset for HCP dataset"""
