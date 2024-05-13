import itertools
import pytest
import torch

from neurograph.config.config import CustomGTNConfig, ConformerConfig, FEDformerConfig
from neurograph.models.gtn import CustomGTN


def _get_cfg(hidden_dim):
    return CustomGTNConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        n_classes=2,
        num_heads=2,
    )


def test_custom_gtn_conformer():

    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg(32)
    model_cfg.use_pos_embed = True
    model_cfg.timestep_model_config = ConformerConfig()

    model = CustomGTN(input_dim=t, num_nodes=r, model_cfg=model_cfg)

    assert model.forward((x, y)).shape == (b, 2)


@pytest.mark.parametrize(
    "attn_block_type",
    ["FEAf", "FEBf", "FEAw", "FEBw"],
)
def test_custom_gtn_fedformer(attn_block_type):

    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg(32)
    model_cfg.use_pos_embed = True
    model_cfg.timestep_model_config = FEDformerConfig(attn_block_type=attn_block_type)

    model = CustomGTN(input_dim=t, num_nodes=r, model_cfg=model_cfg)

    assert model.forward((x, y)).shape == (b, 2)
