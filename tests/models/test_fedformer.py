import itertools
import pytest
import torch

from neurograph.config.config import FEDformerConfig
from neurograph.models.fedformer.fedformer import FEDformer


def _get_cfg(
    hidden_dim, token_embed_type="conv", token_embed_params={"depthwise": True}
):
    return FEDformerConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        n_classes=2,
        num_heads=2,
        token_embed_type=token_embed_type,
        token_embed_params=token_embed_params,
    )


@pytest.mark.parametrize(
    ["attn_block_type", "attn_act"],
    [
        vals
        for vals in itertools.product(
            ["FEAf", "FEBf", "FEAw", "FEBw"],
            ["tanh", "softmax"],
        )
    ],
)
def test_fedformer_attn_block(attn_block_type, attn_act):

    b, r, t = 3, 34, 17
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg(34)
    model_cfg.attn_block_type = attn_block_type
    model_cfg.attn_act = attn_act

    model = FEDformer(input_dim=t, num_nodes=r, model_cfg=model_cfg)

    assert model.forward((x, y)).shape == (b, 2)


@pytest.mark.parametrize(
    "token_embed_type, token_embed_params",
    [
        ("linear", {}),
        ("conv", {"depthwise": True}),
        ("conv", {"depthwise": False}),
    ],
)
def test_fedformer_token_embed(token_embed_type, token_embed_params):

    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)
    y = torch.randint(0, 2, (10,))

    model_cfg = _get_cfg(32)
    model_cfg.attn_block_type = "FEBw"
    model_cfg.attn_act = "tanh"

    model_cfg.token_embed_type = token_embed_type
    model_cfg.token_embed_params = token_embed_params

    model = FEDformer(input_dim=t, num_nodes=r, model_cfg=model_cfg)

    assert model.forward((x, y)).shape == (b, 2)
