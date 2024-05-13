"""Class for multimodal graph dataset based on base graph dataset"""

import os
from pathlib import Path
from typing import Optional

from torch_geometric.data import Data
from neurograph.unidata.datasets import traits
from neurograph.unidata.graph import BaseGraphDataset
from neurograph.unidata.utils import get_subj_ids_from_folds
from neurograph.multidata.utils import prepare_mulimodal_graph


class MultiGraphDataset(BaseGraphDataset):
    """Graph dataset combining fMRI and DTI data"""

    def __init__(
        self,
        root: str,
        name: str,
        atlas: str = "aal",
        abs_thr: Optional[float] = None,
        pt_thr: Optional[float] = None,
        normalize: Optional[str] = None,
        fusion: str = "concat",  # "dti_binary_mask",
        no_cache: bool = False,  # tests / debug only
    ):
        """Multimodal Graph Dataset
        parameterized by dataset name

        Args:
            root (str): path to `datasets` dir
            name (str): dataset name (e.g. cobre)
        """

        self.name = name
        self.atlas = atlas
        self.abs_thr = abs_thr
        self.pt_thr = pt_thr
        self.fusion = fusion
        self.normalize = normalize

        self.global_dir = Path(root) / name
        self.root = str(self.global_dir / "multimodal")
        self.fmri_path = self.global_dir / "fmri" / "raw" / self.atlas
        self.dti_path = self.global_dir / "dti" / "raw" / self.atlas
        self.trait = traits[name]()  # trait instance

        if no_cache:
            for path in self.processed_paths:
                if os.path.isfile(path):
                    os.remove(path)

        super().__init__(self.root)
        self._process()
        self.load_files()
        self.num_nodes = self.get_num_nodes()

    def process(self):
        # load folds, get subject_ids
        raw_folds, _ = self.load_folds(self.global_dir, self.trait.splits_file)
        subj_ids = get_subj_ids_from_folds(raw_folds)

        # load connectivity matrices (ignore time series)
        fmri_cms, _, fmri_roi_map = self.trait.load_cms(self.fmri_path)
        dti_cms, _, dti_roi_map = self.trait.load_cms(self.dti_path)

        assert (
            fmri_roi_map == dti_roi_map
        ), "MultimodalGraph2Dataset: ROI map are different for fMRI and DTI!"

        # load and filter targets by subject_id list
        targets, *_ = self.trait.load_targets(self.global_dir)
        targets = targets.loc[subj_ids].copy()

        # prepare graphs
        datalist = self.prepare_datalist(fmri_cms, dti_cms, targets, subj_ids)

        # load and process folds
        folds = self.process_folds(raw_folds, subj_ids)

        # save everything
        self.save_files(
            datalist=datalist,
            sel_targets=targets,
            subj_ids=subj_ids,
            folds=folds,
            roi_map=fmri_roi_map,
        )

    def prepare_datalist(
        self,
        fmri_cms,
        dti_cms,
        targets,
        subj_ids: list[str],
    ) -> list[Data]:
        """Load a list of Data objects
        NB: Sparsification is not done here!
        """

        datalist = []
        for subj_id in subj_ids:
            fmri_cm = fmri_cms[subj_id]
            dti_cm = dti_cms[subj_id]

            try:
                target = targets.loc[subj_id].values
            except KeyError as exc:  # pragma: no cover
                raise KeyError("CM subj_id not present in loaded targets") from exc

            datalist.append(
                prepare_mulimodal_graph(
                    fmri_cm,
                    dti_cm,
                    subj_id,
                    target,
                    abs_thr=self.abs_thr,
                    pt_thr=self.pt_thr,
                    normalize=self.normalize,
                    fusion=self.fusion,
                ),
            )

        return datalist

    @property
    def file_prefix(self) -> str:
        thr = ""
        if self.abs_thr:
            thr = f"abs={self.abs_thr}"  # pragma: no cover
        if self.pt_thr:
            thr = f"pt={self.pt_thr}"
        return "__".join(
            s
            for s in [
                self.name,
                self.atlas,
                "mm_fmri_dti",
                thr,
                self.normalize,
                self.fusion,
            ]
            if s
        )

    def load_cms(self):
        """stub method to comply to interface"""

    def load_targets(self):
        """stub method to comply to interface"""

    @property
    def target_col(self):
        """Name of target column in `self.target_df`"""
        return self.trait.target_col

    @property
    def subj_id_col(self):
        """Name of subject_id column in `self.target_df`"""
        return self.trait.subj_id_col

    def __repr__(self):  # pragma: no cover
        return (
            f"{self.__class__.__name__}:"
            f" name={self.name},"
            f" atlas={self.atlas},"
            f" pt_thr={self.pt_thr},"
            f" abs_thr={self.abs_thr},"
            f" normalize={self.normalize},"
            f" fusion={self.fusion},"
            f" num_nodes={self.num_nodes},"
            f" num_features={self.num_features},"
            f" size={len(self)}"
        )
