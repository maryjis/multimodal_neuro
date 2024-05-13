from neurograph.run_cwn.run_cwn import prepate_data, train
from neurograph.train.initialize import init_wandb
from neurograph.config import (
    get_config,
    CellularDatasetConfig,
    SparseCINConfig,
)
import wandb


def get_testing_cfg():
    cfg = get_config()
    cfg.log.wandb_mode = "disabled"
    cfg.train.epochs = 1
    cfg.train.device = "cpu"

    # set dataset and model here
    cfg.dataset = CellularDatasetConfig(
        pt_thr=0.05,
        max_ring_size=3,
        n_jobs=1,
        top_pt_rings=0.01,
    )
    cfg.model = SparseCINConfig(hidden=2, num_layers=1)

    return cfg


def test_cwn():
    cfg = get_testing_cfg()
    init_wandb(cfg)
    dataset, complex_list = prepate_data(cfg.dataset)
    res = train(complex_list, dataset, cfg)
    wandb.finish()

    assert res
