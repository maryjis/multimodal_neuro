""" Global config classes and some hydra stuff """

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from .dataset import (
    CellularDatasetConfig,
    MultimodalDatasetConfig,
    MultimodalMorphDatasetConfig,
    MultiGraphDatasetConfig,
    TimeSeriesDenseDatasetConfig,
    UnimodalDatasetConfig,
)


@dataclass
class MLPlayer:
    """Config of one MLP layer"""

    out_size: int = 10
    act_func: Optional[str] = "ReLU"
    act_func_params: Optional[dict] = None
    dropout: Optional[float] = None


@dataclass
class MLPConfig:
    """Config of MLP model"""

    # layers define only hidden dimensions, so final MLP will have n+1 layer.
    # So, if you want to create a 1-layer network, just leave layers empty

    # in and out sizes are optional and usually depend on upstream model and the task
    # for now, they are ignored
    in_size: Optional[int] = None
    out_size: Optional[int] = None

    # act func for the last layer. None -> no activation function
    act_func: Optional[str] = None
    act_func_params: Optional[dict] = None
    layers: list[MLPlayer] = field(
        default_factory=lambda: [
            MLPlayer(
                out_size=32,
                dropout=0.6,
                act_func="LeakyReLU",
                act_func_params=dict(negative_slope=0.2),
            ),
            MLPlayer(
                out_size=32,
                dropout=0.6,
                act_func="LeakyReLU",
                act_func_params=dict(negative_slope=0.2),
            ),
        ]
    )


@dataclass
class ModelConfig:
    """Base class for model config"""

    # model class name; see `neurograph.models.__init__`
    name: str
    # must match with lossss
    n_classes: int

    # required for correct init of models;
    # see `train.train.init_model`
    data_type: str


@dataclass
class DummyMultimodalDense2Config:
    """Dummy MM2 model config"""

    name: str = "DummyMultimodalDense2Model"
    n_classes: int = 2
    hidden: int = 8
    dropout: float = 0.2
    act_func: Optional[str] = "ReLU"
    act_func_params: Optional[dict] = None


# pylint: disable=too-many-instance-attributes
@dataclass
class StandartGNNConfig(ModelConfig):
    """Config for GCN and GAT from PyG"""

    name: str = "StandartGNN"  # see neurograph.models/
    n_classes: int = 2  # must match with loss
    num_layers: int = 2
    layer_module: str = "GCNConv"
    data_type: str = "graph"
    hidden_dim: int = 32
    use_abs_weight: bool = True
    use_weighted_edges: bool = False
    final_node_dim: int = 32
    pooling: str = "concat"
    dropout: float = 0.3
    use_batchnorm: bool = True
    # gat spefic args
    num_heads: Optional[int] = None

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class BrainGCNConfig(ModelConfig):
    """Config for BrainGB GCN"""

    name: str = "BrainGCN"  # see neurograph.models
    checkpoint: str =""
    n_classes: int = 2  # must match with loss
    data_type: str = "graph"

    mp_type: str = "node_concate"
    pooling: str = "concat"
    num_layers: int = 1
    hidden_dim: int = 16
    prepool_dim: int = 64  # input dim for prepool layer
    final_node_dim: int = 8  # final node_dim after prepool
    use_abs_weight: bool = True
    dropout: float = 0.3
    use_batchnorm: bool = True

    # gcn spefic args
    edge_emb_dim: int = 4
    bucket_sz: float = 0.05

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class BrainGATConfig(ModelConfig):
    """Config for BrainGB GAT"""

    name: str = "BrainGAT"  # see neurograph.models
    checkpoint: str =""
    n_classes: int = 2  # must match with loss
    data_type: str = "graph"

    mp_type: str = "node_concate"
    pooling: str = "concat"
    num_layers: int = 1
    hidden_dim: int = 16
    prepool_dim: int = 64  # input dim for prepool layer
    final_node_dim: int = 8  # final node_dim after prepool
    use_abs_weight: bool = True
    dropout: float = 0.0
    use_batchnorm: bool = True

    # gat spefic args
    num_heads: int = 2

    mlp_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class SparseCINConfig:
    """Config for SparseCIN model from cwn"""

    name: str = "SparseCIN"
    num_input_features: int = 116
    num_classes: int = 2
    num_layers: int = 1
    hidden: int = 16
    dropout_rate: float = 0.5
    max_dim: int = 2
    jump_mode: Optional[str] = None
    nonlinearity: str = "relu"
    readout: str = "sum"
    train_eps: bool = False
    final_hidden_multiplier: int = 1
    use_coboundaries: bool = False
    readout_dims: tuple = (0, 1, 2)
    final_readout: str = "sum"
    apply_dropout_before: str = "lin2"
    graph_norm: str = "bn"


@dataclass
class TemporalConvConfig:
    out_channels: int = 128
    kernel_size: int = 3
    groups: int = 1
    dilation: int = 2
    padding: int = 0
    padding_mode: str = "zeros"
    bias: bool = True
    # act function param
    act_func: str = "ReLU"
    act_params: Optional[dict[str, Any]] = None
    # pooling
    pooling_type: str = "max"
    pooling_size: int = 2


@dataclass
class TemporalCNNConfig(ModelConfig):
    name: str = "TemporalCNN"
    n_classes: int = 2
    data_type: str = "dense"

    pooling_readout: str = "mean"  # "max", "meanmax"
    dropout_rate: float = 0.2

    layers: list[TemporalConvConfig] = field(
        default_factory=lambda: [
            TemporalConvConfig(out_channels=1024),
            TemporalConvConfig(out_channels=128),
        ]
    )


@dataclass
class TransformerConfig(ModelConfig):
    """Unimodal vanilla transformer config"""

    # name is a class name; used for initializing a model
    name: str = "Transformer"

    n_classes: int = 2
    num_layers: int = 1
    hidden_dim: int = 116
    num_heads: int = 4
    attn_dropout: float = 0.5
    mlp_dropout: float = 0.5
    # hidden layer in transformer block mlp
    mlp_hidden_multiplier: float = 0.2

    data_type: str = "dense"

    return_attn: bool = False
    # transformer block MLP parameters
    mlp_act_func: Optional[str] = "GELU"
    mlp_act_func_params: Optional[dict] = None

    pooling: str = "concat"

    # final MLP layer config
    head_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            layers=[
                MLPlayer(
                    out_size=4,
                    dropout=0.5,
                    act_func="GELU",
                ),
            ]
        )
    )


@dataclass
class ConformerConfig(ModelConfig):
    """Unimodal vanilla transformer config"""

    # name is a class name; used for initializing a model
    name: str = "Conformer"
    data_type: str = "dense"

    n_classes: int = 2
    num_layers: int = 1
    pooling: str = "concat"

    act_func_name: str = "SiLU"
    act_func_params: Optional[dict[str, Any]] = field(default_factory=lambda: {})

    # ConformerLayer
    ffn_dim: int = 200  # hidden dim for FFN
    num_heads: int = 4
    relative_key: bool = True
    depthwise_conv_kernel_size: int = 3  # must be odd
    # hidden num_channels = int(num_channels_multiplier * input_dim)
    num_channels_multiplier: float = 2.0
    dropout: float = 0.1  # for mlps, conv, ffn
    use_group_norm: bool = False  # use BatchNorm if True
    convolution_first: bool = False
    return_attn: bool = False
    padding_mode: str = "circular"

    # final MLP layer config
    head_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            layers=[
                MLPlayer(out_size=32, dropout=0.5, act_func="GELU"),
            ]
        )
    )


@dataclass
class FEDformerConfig(ModelConfig):
    """FEDformer config"""

    # name is a class name; used for initializing a model
    name: str = "FEDformer"
    data_type: str = "dense"

    n_classes: int = 2
    num_layers: int = 2
    pooling: str = "concat"
    dropout: float = 0.2

    attn_block_type: str = "FEBf"
    attn_act: str = "tanh"
    hidden_dim: int = 116
    num_heads: int = 4
    return_attn: bool = False

    # mwt params
    k: int = 4
    c: int = 32
    L: int = 3
    base: str = "legendre"

    use_token_embed: bool = True
    token_embed_type: str = "conv"
    token_embed_params: Optional[dict] = field(
        default_factory=lambda: {"depthwise": True}
    )
    use_pos_embed: bool = True

    num_modes: int = 32
    mode_selection: str = "random"
    channel_multiplier: int = 8

    # final MLP layer config
    head_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            layers=[
                MLPlayer(out_size=32, dropout=0.2, act_func="GELU"),
            ]
        )
    )


@dataclass
class GTNConfig(TransformerConfig):
    """Gate Transformer Network config"""

    # name is a class name; used for initializing a model
    name: str = "GTN"
    data_type: str = "dense"

    n_classes: int = 2
    num_layers: int = 1
    hidden_dim: int = 128
    num_heads: int = 4

    # hidden layer in transformer block mlp
    mlp_hidden_multiplier: float = 2.0
    mlp_dropout: float = 0.1

    # attn params
    return_attn: bool = False
    attn_dropout: float = 0.1

    # transformer block MLP parameters
    mlp_act_func: Optional[str] = "GELU"
    mlp_act_func_params: Optional[dict] = None

    # positional embedding
    use_pos_embed: bool = True


@dataclass
class CustomGTNConfig(GTNConfig):
    """Custom Gate Transformer Network config"""

    # name is a class name; used for initializing a model
    name: str = "CustomGTN"
    data_type: str = "dense"

    n_classes: int = 2
    num_layers: int = 1
    hidden_dim: int = 128
    num_heads: int = 4

    # timestep model config
    # available types ConformerConfig, FEDformerConfig
    # unfortunately, hydra doesn't support unions of user data types
    # here we use "dummy" model config
    timestep_model_config: ModelConfig = ModelConfig(
        name="", n_classes=2, data_type="dense"
    )

    # hidden layer in transformer block mlp
    mlp_hidden_multiplier: float = 2.0
    mlp_dropout: float = 0.1

    # attn params
    return_attn: bool = False
    attn_dropout: float = 0.1

    # transformer block MLP parameters
    mlp_act_func: Optional[str] = "GELU"
    mlp_act_func_params: Optional[dict] = None

    # positional embedding
    use_pos_embed: bool = True


@dataclass
class PriorTransformerConfig(TransformerConfig):
    """Unimodal vanilla transformer config"""

    # name is a class name; used for initializing a model
    name: str = "PriorTransformer"
    alpha: float = 1.0
    trainable_alpha: bool = True


@dataclass
class MultiModalTransformerConfig(TransformerConfig):
    """Multimodal (2) vanilla transformer config"""

    # name is a class name; used for initializing a model
    name: str = "MultiModalTransformer"
    attn_type: str = "concat"
    make_projection: bool = True


@dataclass
class BolTConfig(ModelConfig):
    """BolT model original config"""

    # pylint: disable=invalid-name
    name: str = "BolT"
    checkpoint: str = ""

    n_classes: int = 7
    data_type: str = "dense"

    nOfLayers: int = 4
    
    tokens: str = "timeseries" # rois
    dim: int = 268

    numHeads: int = 36
    headDim: int = 20

    windowSize: int = 20
    shiftCoeff: float = 2.0 / 5.0
    fringeCoeff: float = 2
    focalRule: str = "expand"

    mlpRatio: float = 1.0
    attentionBias: bool = True
    drop: float = 0.1
    attnDrop: float = 0.1
    lambdaCons: int = 1

    pooling: str = "cls"


@dataclass
class MultiModalBoltConfig(BolTConfig):
    name: str = "MultiModalBolt"
    data_type: str = "multimodal_dense_2"


@dataclass
class MultiModalMorphBoltConfig(BolTConfig):
    name: str = "MultiModalMorphBolt"
    data_type: str = "morph_multimodal_dense_2"
    fusion_type: str = "sum" # concat, late
    fusion_dropout: float =0.2
    fusion_dim: int =32


@dataclass
class MultiModalMorphBoltV2Config(BolTConfig):
    name: str = "MultiModalMorphBoltV2"
    data_type: str = "morph_multimodal_dense_2"
    fusion_type: str = "sum" # concat, late
    fusion_dropout: float =0.2
    fusion_dim: int =32
    model_type: str = "cross_modality" # original, last, cross_modality
    cross_modality_possition: int = 96


@dataclass
class TrainConfig:
    """Training config: device, num epochs, optim params etc."""

    device: str = "cuda:0"
    num_threads: Optional[int] = 1
    epochs: int = 1
    batch_size: int = 16
    valid_batch_size: int = 8
    optim: str = "Adam"
    optim_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            "weight_decay": 1e-3,
        }
    )
    scheduler: Optional[str] = "ReduceLROnPlateau"
    # used in ReduceLROnPlateau
    scheduler_metric: Optional[str] = "loss"
    scheduler_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "factor": 0.1,
            "patience": 5,
            "verbose": True,
        }
    )
    # select best model on valid based on what metric
    select_best_model: bool = True
    select_best_metric: str = "loss"
    # early stopping patience; set to None to turn off early stopping
    patience: Optional[int] = None
    loss: str = "CrossEntropyLoss"  #'BCEWithLogitsLoss'
    loss_args: Optional[dict[str, Any]] = field(
        # reduction sum is necessary here
        default_factory=lambda: {"reduction": "sum"}
    )
    # if BCE is used
    prob_thr: float = 0.5
    use_nested: bool = True
    num_outer_folds: int = 5
    save_model: bool = True


@dataclass
class TrainBoltConfig(TrainConfig):
    """Default training params for Bolt"""

    device: str = "cuda:0"
    num_threads: Optional[int] = None
    epochs: int = 20
    batch_size: int = 32
    valid_batch_size: int = 32
    optim: str = "Adam"
    optim_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "lr": 2e-4,
            "weight_decay": 0,
        }
    )
    scheduler: Optional[str] = "OneCycleLR"
    # used in ReduceLROnPlateau
    scheduler_metric: Optional[str] = "loss"
    scheduler_args: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "div_factor": 4e-4 / 2e-4,
            "final_div_factor": 2e-4 / 2e-5,
            "pct_start": 0.3,
            "max_lr": 4e-4,
            "total_steps": 500,
            "verbose": True,
        }
    )
    # select best model on valid based on what metric
    select_best_model: bool = False
    select_best_metric: str = "f1_macro"
    loss: str = "BolTLoss"  #'BCEWithLogitsLoss'
    loss_args: Optional[dict[str, Any]] = field(
        # reduction sum is necessary here
        default_factory=lambda: {"lambda_cons": 1.0}
    )
    # if BCE is used
    prob_thr: float = 0.5
    use_nested: bool = True
    num_outer_folds: int = 10
    save_model: bool = True


@dataclass
class LogConfig:
    """Basically, WandB config"""

    # how often print training metrics
    test_step: int = 1
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = "gnn-neuro"
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_mode: Optional[str] = None  # 'disabled' for testing


@dataclass
class Config:
    """Config dataclass w/ default values (see dataclasses above)"""

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"dataset": "base_dataset"},
            {"train": "base_train"},
            {"model": "empty"},  # from yaml file
        ]
    )

    seed: int = 1380
    check_commit: bool = True

    model: Any = MISSING
    dataset: Any = MISSING
    train: Any = MISSING
    log: LogConfig = field(default_factory=LogConfig)


def validate_config(cfg: Config):
    """Function for validating Config instances"""

    losses = ("CrossEntropyLoss", "BCEWithLogitsLoss", "BolTLoss")

    if cfg.train.loss == "CrossEntropyLoss" and cfg.model.n_classes < 2:
        raise ValueError(
            f"For loss = {cfg.train.loss} `Config.model.n_classes` must be > 1"
        )
    if cfg.train.loss == "BCEWithLogitsLoss" and cfg.model.n_classes != 1:
        raise ValueError(
            f"For loss = {cfg.train.loss} `Config.model.n_classes` must be = 1"
        )

    if cfg.train.loss not in losses:
        raise ValueError(f"Unknown loss {cfg.train.loss}. Available options: {losses}")


# register default config as `base_config`
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

# base dataset configs
cs.store(group="dataset", name="base_dataset", node=UnimodalDatasetConfig)
cs.store(group="dataset", name="base_multimodal_dataset", node=MultimodalDatasetConfig)
cs.store(group="dataset", name="morph_multimodal_dataset", node=MultimodalMorphDatasetConfig)
cs.store(group="dataset", name="cellular_dataset", node=CellularDatasetConfig)
cs.store(group="dataset", name="ts_dataset", node=TimeSeriesDenseDatasetConfig)
cs.store(group="dataset", name="multigraph_dataset", node=MultiGraphDatasetConfig)

# base model configs
cs.store(group="model", name="bgbGAT", node=BrainGATConfig)
cs.store(group="model", name="bgbGCN", node=BrainGCNConfig)
cs.store(group="model", name="transformer", node=TransformerConfig)
cs.store(group="model", name="prior_transformer", node=PriorTransformerConfig)
cs.store(group="model", name="standart_gnn", node=StandartGNNConfig)
cs.store(group="model", name="dummy_mm2", node=DummyMultimodalDense2Config)
cs.store(group="model", name="mm_transformer", node=MultiModalTransformerConfig)
cs.store(group="model", name="sparse_cin", node=SparseCINConfig)
cs.store(group="model", name="bolt", node=BolTConfig)
cs.store(group="model", name="temp_cnn", node=TemporalCNNConfig)
cs.store(group="model", name="conformer", node=ConformerConfig)
cs.store(group="model", name="multimodal_bolt", node=MultiModalBoltConfig)
cs.store(group="model", name="multimodal_morph_bolt", node=MultiModalMorphBoltConfig)
cs.store(group="model", name="multimodal_morph_bolt_v2", node=MultiModalMorphBoltV2Config)
cs.store(group="model", name="fedformer", node=FEDformerConfig)
cs.store(group="model", name="gtn", node=GTNConfig)
cs.store(group="model", name="custom_gtn", node=CustomGTNConfig)

# sumodel of custom GTN
cs.store(
    group="model.timestep_model_config", name="sub_conformer", node=ConformerConfig
)
cs.store(
    group="model.timestep_model_config", name="sub_fedformer", node=FEDformerConfig
)


# train config
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="train", name="bolt_train", node=TrainBoltConfig)
