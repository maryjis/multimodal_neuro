import pytest
import torch

from neurograph.config import (
    MLPlayer,
    MLPConfig,
    PriorTransformerConfig,
)
from neurograph.models.prior_transformers import PriorTransformer


@pytest.mark.parametrize(
    "num_classes, pooling, expected",
    [
        (1, "concat", 1),
        (1, "mean", 1),
        (1, "sum", 1),
        (2, "concat", 2),
        (2, "mean", 2),
        (2, "sum", 2),
    ],
)
def test_transformers(num_classes, pooling, expected):
    b, n, d_i = 3, 11, 21
    x = torch.randn(b, n, d_i)
    attn = torch.randn(b, n, n)
    y = torch.randint(0, 3, (b,))

    t_cfg = PriorTransformerConfig(
        n_classes=num_classes,
        num_layers=2,
        hidden_dim=32,
        num_heads=4,
        attn_dropout=0.5,
        mlp_dropout=0.5,
        mlp_hidden_multiplier=2.0,
        pooling=pooling,
        # final MLP layer config
        head_config=MLPConfig(
            layers=[
                MLPlayer(
                    out_size=32,
                    dropout=0.5,
                    act_func="ELU",
                ),
            ]
        ),
    )
    m = PriorTransformer(
        input_dim=d_i,
        num_nodes=n,
        model_cfg=t_cfg,
    )
    o = m((x, attn, y))

    assert o.shape == (b, expected)
