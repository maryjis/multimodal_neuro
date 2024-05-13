import itertools
import pytest
import torch

from neurograph.config.config import ConformerConfig
from neurograph.models.conformer import Conformer


def _get_cfg():
    return ConformerConfig(
        ffn_dim=8,
        num_channels_multiplier=4,
        num_layers=2,
        n_classes=2,
    )


@pytest.mark.parametrize(
    ["relative_key", "convolution_first"],
    [
        vals
        for vals in itertools.product(
            [True, False],
            [True, False],
        )
    ],
)
def test_conformer_relative_key(relative_key, convolution_first):

    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg()
    model_cfg.relative_key = relative_key
    model_cfg.convolution_first = convolution_first

    model = Conformer(input_dim=t, num_nodes=r, model_cfg=model_cfg)
    assert all(
        [
            hasattr(model.layers[i].attn, "distance_embedding") == relative_key
            for i in range(2)
        ]
    )
    assert model.forward((x, y)).shape == (b, 2)


@pytest.mark.parametrize(
        "act_func_name, act_func_params",
        [
            ('SiLU', None),
            ('Snake', {'a': 0.1}),
            ('Snake', {'a': 10}),
        ]
)
def test_conformer_act_func(act_func_name, act_func_params):

    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg()
    model_cfg.act_func_name = act_func_name
    model_cfg.act_func_params = act_func_params

    model = Conformer(input_dim=t, num_nodes=r, model_cfg=model_cfg)

    assert model.forward((x, y)).shape == (b, 2)