"""Util functions for creating graphs from multimodal data"""
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from neurograph.unidata.utils import conn_matrix_to_edges, normalize_cm


def prepare_mulimodal_graph(
    fmri_cm: np.ndarray,
    dti_cm: np.ndarray,
    subj_id: str,
    target: int | float,
    abs_thr: Optional[float] = None,
    pt_thr: Optional[float] = None,
    normalize: Optional[str] = None,
    fusion: str = "concat",
) -> Data:
    """NB: We do not apply thresholding to DTI data.
    NB: connectivity matrix is used both for computing edge_index AND
    node features

    Args:
        fmri_cm (np.ndarray): fMRI connectivity matrix
        dti_cm (np.ndarray): DTI connectivity matrix
        subj_id (str): subject_id
        target (np.ndarray): 1d-array: target for this subject_id.
            (`target.loc[subj_id].values`)
        abs_thr (Optional[float], optional):
            Absolute threshold for sparsification. Defaults to None.
        pt_thr (Optional[float], optional):
            Proportional threshold for sparsification (pt_thr must be (0, 1). Defaults to None.
        normalize (Optional[str], optional): How to normalize connectivity matrix.
            Options: global_max, log. Defaults to None.
        fusion (str, optional): How to combine modalities.
            Options: 'dti_binary_mask', 'concat'. Defaults to 'concat'.

    Returns:
        Data: a resulting graph
    """

    assert not (abs_thr and pt_thr), "Specify only abs_thr or pt_thr"
    if fusion == "dti_binary_mask" and (abs_thr or pt_thr):
        raise ValueError(
            "Please set abs_thr and pt_thr when using fusion=dti_binary_mask"
        )

    fmri_cm = fmri_cm.astype(np.float32)
    dti_cm = dti_cm.astype(np.int32)

    # final_cm: node embeddings
    if fusion == "dti_binary_mask":
        bin_mask = (dti_cm != 0).astype(np.float32)
        fmri_cm = fmri_cm * bin_mask
        final_cm = fmri_cm
    elif fusion == "concat":
        # normalize DTI data
        dti_cm = normalize_cm(dti_cm, normalize)
        # concat DTI CM to fMRI CM
        final_cm = np.hstack([fmri_cm, dti_cm])
    else:
        raise ValueError("Unknown `fusion` value")  # pragma: no cover

    # convert fMRI CM to edge_index, edge_attr
    fmri_edge_index, fmri_edge_attr = conn_matrix_to_edges(
        fmri_cm,
        abs_thr=abs_thr,
        pt_thr=pt_thr,
        remove_zeros=True,
    )
    data = Data(
        edge_index=fmri_edge_index,
        edge_attr=fmri_edge_attr,
        x=torch.from_numpy(final_cm).float(),
        num_nodes=final_cm.shape[0],
        y=torch.LongTensor(target),
        subj_id=subj_id,
    )
    # data.validate()
    return data
