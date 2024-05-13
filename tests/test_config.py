import pytest

from neurograph.config import get_config
from neurograph.config.config import Config, ModelConfig, TrainConfig, validate_config


def get_dummy_config(loss: str, n_classes: int):
    cfg = get_config()
    cfg.train = TrainConfig()
    cfg.train.loss = loss

    cfg.model = ModelConfig(
        name="model",
        n_classes=n_classes,
        data_type="graph",
    )
    return cfg


def test_validate_config():
    assert validate_config(get_dummy_config("BCEWithLogitsLoss", 1)) == None

    with pytest.raises(ValueError):
        validate_config(get_dummy_config("BCEWithLogitsLoss", 2))
    with pytest.raises(ValueError):
        validate_config(get_dummy_config("CrossEntropyLoss", 1))
    with pytest.raises(ValueError):
        validate_config(get_dummy_config("MSE", 2))
