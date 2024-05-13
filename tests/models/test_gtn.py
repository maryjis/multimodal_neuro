import pytest
import torch

from neurograph.config.config import GTNConfig
from neurograph.models.gtn import GTN


def _get_cfg(hidden_dim):
    return GTNConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        n_classes=2,
        num_heads=2,
    )


@pytest.mark.parametrize(
    "use_pos_embed",
    [True, False],
)
def test_gtn(use_pos_embed):

    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg(32)
    model_cfg.use_pos_embed = use_pos_embed

    model = GTN(input_dim=t, num_nodes=r, model_cfg=model_cfg)

    assert model.forward((x, y)).shape == (b, 2)
