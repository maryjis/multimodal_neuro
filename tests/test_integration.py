from neurograph.config import (
    TimeSeriesDenseDatasetConfig,
    UnimodalDatasetConfig,
    MultimodalDatasetConfig,
    MultiGraphDatasetConfig,
    get_config,
    MultiModalTransformerConfig,
    PriorTransformerConfig,
    TransformerConfig,
    BolTConfig,
    BrainGATConfig,
    validate_config,
    TrainConfig,
)
from neurograph.train.initialize import dataset_factory, init_wandb
from neurograph.train.train import train
import wandb

from .conftest import data_path


def get_testing_cfg():
    cfg = get_config()
    cfg.log.wandb_mode = "disabled"
    cfg.train = TrainConfig()
    cfg.train.epochs = 1
    cfg.train.device = "cpu"

    return cfg


def run(cfg):
    validate_config(cfg)
    init_wandb(cfg)

    dataset = dataset_factory(cfg.dataset)

    cfg.train.save_model = False
    res = train(dataset, cfg, ".")
    wandb.finish()

    return res


def test_bce():
    """test that everything works w/ BCEWithLogitsLoss"""

    cfg = get_testing_cfg()
    cfg.dataset = UnimodalDatasetConfig(
        data_type="dense",
        name="cobre",
        data_path=data_path,
        random_crop=False,
    )
    cfg.model = TransformerConfig()
    cfg.train.loss = "BCEWithLogitsLoss"
    cfg.model.n_classes = 1

    assert run(cfg)


def test_train_unimodal_dense():
    cfg = get_testing_cfg()
    cfg.dataset = UnimodalDatasetConfig(
        data_type="dense",
        name="cobre",
        data_path=data_path,
    )
    cfg.model = TransformerConfig()

    assert run(cfg)


def test_train_timeseries_dense():
    cfg = get_testing_cfg()
    cfg.dataset = TimeSeriesDenseDatasetConfig(
        experiment_type="fmri",
        data_type="DenseTimeSeriesDataset",
        name="cobre",
        fourier_timeseries=True,
        data_path=data_path,
    )
    cfg.model = PriorTransformerConfig()

    assert run(cfg)


def test_train_unimodal_graph():
    cfg = get_testing_cfg()
    cfg.dataset = UnimodalDatasetConfig(
        data_type="graph",
        name="cobre",
        data_path=data_path,
    )
    cfg.model = BrainGATConfig(n_classes=2)

    assert run(cfg)


def test_train_multimodal_graph():
    cfg = get_testing_cfg()
    cfg.dataset = MultiGraphDatasetConfig()
    cfg.model = BrainGATConfig(n_classes=2)

    assert run(cfg)


def test_train_multimodal_dense():
    cfg = get_testing_cfg()
    cfg.dataset = MultimodalDatasetConfig(
        name="cobre",
    )
    cfg.model = MultiModalTransformerConfig(n_classes=2)

    assert run(cfg)


def test_train_bolt():
    cfg = get_config(overrides=["train=bolt_train"])
    cfg.dataset = UnimodalDatasetConfig(
        data_type="dense",
        feature_type="timeseries",
        atlas="shen",
        name="hcp",
        data_path=data_path,
        time_series_length=150,
        random_crop=True,
    )
    cfg.log.wandb_mode = "disabled"
    cfg.train.device = "cpu"
    cfg.train.batch_size = 32
    cfg.train.epochs = 1
    cfg.train.num_outer_folds = 4
    cfg.model = BolTConfig(
        n_classes=7,
        nOfLayers=1,
        numHeads=1,
        headDim=2,
        windowSize=100,
    )

    assert run(cfg)
