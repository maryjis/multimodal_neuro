import itertools

import pytest
import torch

from neurograph.config import MultiModalTransformerConfig
from neurograph.models.multimodal_transformers import MultiModalTransformer


@pytest.mark.parametrize(
    ["make_projection", "attn_type"],
    [
        vals
        for vals in itertools.product(
            [True, False],
            ["concat", "hierarchical", "cross-attention", "sum", "multiply"],
        )
    ],
)
def test_multimodal_transformers(make_projection, attn_type):
    model_cfg = MultiModalTransformerConfig(
        hidden_dim=32,
        n_classes=2,
        return_attn=False,
        make_projection=make_projection,
        attn_type=attn_type,
    )

    inp_dim_1 = 32
    inp_dim_2 = 32
    x_1 = torch.randn(10, 3, inp_dim_1)
    x_2 = torch.randn(10, 3, inp_dim_2)
    y = torch.randint(0, 2, (10,))

    model = MultiModalTransformer(
        inp_dim_1,
        inp_dim_2,
        3,
        3,
        model_cfg,
    )
    assert model.forward((x_1, x_2, y)).shape == (10, 2)
