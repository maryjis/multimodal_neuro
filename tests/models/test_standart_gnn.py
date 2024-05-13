import itertools

import pytest
from neurograph.config import StandartGNNConfig
from neurograph.models.standart_gnn import build_gnn_block, StandartGNN

from .utils import random_pyg_batch


@pytest.mark.parametrize(
    ["layer_module", "use_batchnorm", "use_weighted_edges"],
    [
        vals
        for vals in itertools.product(
            ["GATConv", "GCNConv"],
            [True, False],
            [True, False],
        )
    ],
)
def test_build_gnn_block(
    layer_module,
    use_batchnorm,
    use_weighted_edges,
):

    input_dim = 32
    hidden_dim = 16

    block = build_gnn_block(
        input_dim,
        hidden_dim,
        layer_module,
        2,  # num heads
        use_batchnorm,
        use_weighted_edges,
        0.5,  # dropout
    )

    batch_size = 10
    num_nodes = 17
    data = random_pyg_batch(
        batch_size=batch_size, num_nodes=num_nodes, num_features=input_dim
    )

    x, edge_index, edge_attr, _ = data.x, data.edge_index, data.edge_attr, data.batch
    if use_weighted_edges:
        res = block(x, edge_index, edge_attr)
    else:
        res = block(x, edge_index)

    assert res.shape == (batch_size * num_nodes, hidden_dim)


@pytest.mark.parametrize(
    ["layer_module", "use_batchnorm", "use_weighted_edges", "pooling"],
    [
        vals
        for vals in itertools.product(
            ["GATConv", "GCNConv"],
            [True, False],
            [True, False],
            ["concat", "mean", "sum"],
        )
    ],
)
def test_standart_gnn(
    layer_module,
    use_batchnorm,
    use_weighted_edges,
    pooling,
):
    input_dim = 32
    hidden_dim = 16
    n_classes = 2
    model_cfg = StandartGNNConfig(
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        layer_module=layer_module,
        use_batchnorm=use_batchnorm,
        use_weighted_edges=use_weighted_edges,
        pooling=pooling,
        num_heads=2,
    )

    batch_size, num_nodes = 10, 17
    data = random_pyg_batch(
        batch_size=batch_size, num_nodes=num_nodes, num_features=input_dim
    )
    model = StandartGNN(input_dim, num_nodes, model_cfg)

    assert model(data).shape == (batch_size, n_classes)
