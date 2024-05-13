import numpy as np
from neurograph.multidata.utils import prepare_mulimodal_graph


def test_prepare_mulimodal_graph_dti_mask():
    fmri_cm = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.3],
            [0.5, 0.3, 0.0],
        ]
    )

    dti_cm = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 5.0],
            [1.0, 5.0, 7.0],
        ]
    )

    g = prepare_mulimodal_graph(
        fmri_cm,
        dti_cm,
        "id0",
        np.array([0]),
        normalize="global_max",
        fusion="dti_binary_mask",
    )

    assert g.edge_index.shape[1] == 4
    assert (g.x != 0.0).sum().item() == 4


def test_prepare_mulimodal_graph_concat():
    fmri_cm = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.3],
            [0.5, 0.3, 0.0],
        ]
    )

    dti_cm = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 5.0],
            [1.0, 5.0, 7.0],
        ]
    )

    g = prepare_mulimodal_graph(
        fmri_cm,
        dti_cm,
        "id0",
        np.array([0]),
        normalize="global_max",
        fusion="concat",
    )

    assert g.edge_index.shape[1] == 4
    assert (g.x != 0.0).sum().item() == 9
