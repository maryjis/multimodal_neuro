import itertools

import pytest
from neurograph.config import BrainGATConfig
from neurograph.models.gat import build_gat_block, BrainGAT

from .utils import random_pyg_batch


@pytest.mark.parametrize(
    ["use_batchnorm", "use_weighted_edges", "mp_type", "proj_dim"],
    [
        vals
        for vals in itertools.product(
            [True, False],
            [True, False],
            [
                "attention_weighted",
                "attention_edge_weighted",
                "sum_attention_edge",
                "edge_node_concate",
                "node_concate",
            ],
            [8, None],
        )
    ],
)
def test_build_braingat_block(
    use_batchnorm,
    use_weighted_edges,
    mp_type,
    proj_dim,
):

    input_dim = 32
    hidden_dim = 16

    block = build_gat_block(
        input_dim,
        hidden_dim,
        use_batchnorm=use_batchnorm,
        use_abs_weight=use_weighted_edges,
        mp_type=mp_type,
        proj_dim=proj_dim,
        num_heads=2,
        dropout=0.5,
    )

    batch_size = 10
    num_nodes = 17
    data = random_pyg_batch(
        batch_size=batch_size, num_nodes=num_nodes, num_features=input_dim
    )

    x, edge_index, edge_attr, _ = data.x, data.edge_index, data.edge_attr, data.batch
    res = block(x, edge_index, edge_attr)

    out_dim = proj_dim if proj_dim else hidden_dim

    assert res.shape == (batch_size * num_nodes, out_dim)


@pytest.mark.parametrize(
    ["num_layers", "use_batchnorm", "use_abs_weight", "pooling", "mp_type"],
    [
        vals
        for vals in itertools.product(
            [1, 2],
            [True, False],
            [True, False],
            ["concat", "sum", "mean"],
            [
                "attention_weighted",
                "attention_edge_weighted",
                "sum_attention_edge",
                "edge_node_concate",
                "node_concate",
            ],
        )
    ],
)
def test_brain_gat(
    num_layers,
    use_batchnorm,
    use_abs_weight,
    pooling,
    mp_type,
):
    input_dim = 32
    hidden_dim = 16
    n_classes = 2
    model_cfg = BrainGATConfig(
        num_layers=num_layers,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        use_batchnorm=use_batchnorm,
        use_abs_weight=use_abs_weight,
        pooling=pooling,
        num_heads=2,
        mp_type=mp_type,
    )

    batch_size, num_nodes = 10, 17
    data = random_pyg_batch(
        batch_size=batch_size, num_nodes=num_nodes, num_features=input_dim
    )
    model = BrainGAT(input_dim, num_nodes, model_cfg)

    assert model(data).shape == (batch_size, n_classes)
