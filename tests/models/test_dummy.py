import itertools

import pytest
import torch

from neurograph.config.config import DummyMultimodalDense2Config
from neurograph.models.dummy import DummyMultimodalDense2Model


@pytest.mark.parametrize("act_func", ["ReLU", None])
def test_dummy_mm2(act_func):
    model_cfg = DummyMultimodalDense2Config(act_func=act_func)

    inp_dim_1 = 32
    inp_dim_2 = 32
    x_1 = torch.randn(10, 3, inp_dim_1)
    x_2 = torch.randn(10, 3, inp_dim_2)
    y = torch.randint(0, 2, (10,))

    model = DummyMultimodalDense2Model(
        inp_dim_1,
        inp_dim_2,
        3,
        3,
        model_cfg,
    )
    assert model.forward((x_1, x_2, y)).shape == (10, 2)
