import itertools

import pytest
import torch

from neurograph.config.config import TemporalCNNConfig, TemporalConvConfig
from neurograph.models.temporal_cnn import TemporalCNN


@pytest.mark.parametrize(
    [
        "pooling_readout",
        "groups",
        "dilation",
        "kernel_size",
        "pooling_type",
        "pooling_size",
    ],
    [
        vals
        for vals in itertools.product(
            ["mean", "max", "meanmax"],
            [1, 4],
            [1, 2],
            [2, 3],
            ["mean", "max"],
            [2, 4],
        )
    ],
)
def test_temporal_cnn(
    pooling_readout, groups, dilation, kernel_size, pooling_type, pooling_size
):
    layer = TemporalConvConfig(
        out_channels=4,
        groups=groups,
        dilation=dilation,
        kernel_size=kernel_size,
        pooling_type=pooling_type,
        pooling_size=pooling_size,
    )

    model_cfg = TemporalCNNConfig(
        # no hidden_dim since input is an original timeseries
        n_classes=2,
        pooling_readout=pooling_readout,
        layers=[layer, layer],
    )

    b, r, t = 3, 4, 128
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model = TemporalCNN(input_dim=t, num_nodes=r, model_cfg=model_cfg)
    assert model.forward((x, y)).shape == (b, 2)
