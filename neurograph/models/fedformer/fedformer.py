import torch
from torch import nn

from neurograph.config.config import FEDformerConfig
from neurograph.models.embed import (
    ConvTokenEmbedding,
    LinearTokenEmbedding,
    PositionalEmbedding,
)
from neurograph.models.fedformer.fourier_block import FEAf, FEBf
from neurograph.models.fedformer.wavelet_block import FEAw, FEBw
from neurograph.models.transformers import (
    SingleMultiheadProjection,
    ProjectBeforeMSA,
    compute_final_dimension,
    pool,
)
from neurograph.models.mlp import BasicMLP


class FEDformerLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,  # timestep embed dim
        model_cfg: FEDformerConfig,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = model_cfg.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        assert (
            self.hidden_dim % self.num_heads == 0
        ), "hidden_dim must be multiple of num_heads"  # pragma: no cover

        # multihead projection module
        if model_cfg.attn_block_type in ("FEAf", "FEAw"):
            self.multihead_projection = ProjectBeforeMSA(
                self.hidden_dim, self.hidden_dim, model_cfg.num_heads
            )
        elif model_cfg.attn_block_type in ("FEBf", "FEBw"):
            self.multihead_projection = SingleMultiheadProjection(
                self.hidden_dim, self.hidden_dim, model_cfg.num_heads
            )

        self.attn_block = self.build_attn_block(
            seq_len,
            hidden_dim,
            model_cfg,
        )

        # conv blocks
        hidden_conv_dim = int(model_cfg.channel_multiplier * hidden_dim)
        self.conv_1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_conv_dim,
            kernel_size=1,
            bias=False,
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_conv_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=False,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(model_cfg.dropout)

    def build_attn_block(
        self,
        seq_len: int,
        hidden_dim: int,  # timestep embed dim
        model_cfg: FEDformerConfig,
    ) -> FEAf | FEBf | FEAw | FEBw:

        fourier_paramas = dict(
            d_model=hidden_dim,
            seq_len=seq_len,
            num_heads=model_cfg.num_heads,
            num_modes=model_cfg.num_modes,
            mode_select_method=model_cfg.mode_selection,
        )
        mwt_params = dict(
            d_model=hidden_dim,
            num_modes=model_cfg.num_modes,
            k=model_cfg.k,
            c=model_cfg.c,
            L=model_cfg.L,
            base=model_cfg.base,
        )

        if model_cfg.attn_block_type == "FEAf":
            return FEAf(activation=model_cfg.attn_act, **fourier_paramas)

        if model_cfg.attn_block_type == "FEAw":
            return FEAw(activation=model_cfg.attn_act, **mwt_params)

        if model_cfg.attn_block_type == "FEBf":
            return FEBf(**fourier_paramas)

        if model_cfg.attn_block_type == "FEBw":
            return FEBw(**mwt_params)

        names = ("FEAf", "FEBf", "FEAw", "FEBw")  # pragma: no cover
        raise ValueError(
            f"Unknown Fourier Block type. Options: {names}"
        )  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (b, t, D)
        b, t, _ = x.shape

        # multihead projection: (b, t, D) -> (b, t, h, d)
        out = self.multihead_projection(x)

        # FEB-f block: (b, t, h, d) -> (b, t, h, d)
        out = self.attn_block(*out).reshape(b, t, -1)

        # first skip connection
        x = x + self.dropout(out)

        # conv
        y = x
        y = self.dropout(self.act(self.conv_1(y.transpose(-1, 1))))
        y = self.dropout(self.conv_2(y).transpose(-1, 1))

        # second skip connection
        return x + y


class FEDformer(nn.Module):
    def __init__(
        self,
        input_dim: int,  # time series length
        num_nodes: int,  # num ROIs
        model_cfg: FEDformerConfig,
    ):
        super().__init__()
        self.pooling = model_cfg.pooling

        # if we don't use token embeddings, then we use original time series
        self.hidden_dim = (
            model_cfg.hidden_dim if model_cfg.use_token_embed else num_nodes
        )

        # token embedding
        self.use_token_embed = model_cfg.use_token_embed
        if model_cfg.use_token_embed:
            token_embed_params = (
                {}
                if model_cfg.token_embed_params is None
                else model_cfg.token_embed_params
            )
            if model_cfg.token_embed_type == "conv":
                self.token_embedding = ConvTokenEmbedding(
                    input_dim=num_nodes, d_model=self.hidden_dim, **token_embed_params
                )
            elif model_cfg.token_embed_type == "linear":
                self.token_embedding = LinearTokenEmbedding(
                    input_dim=num_nodes, d_model=self.hidden_dim, **token_embed_params
                )
            else:
                raise ValueError("Unknown token embedding type")  # pragma: no cover

        # positional embedding
        self.use_pos_embed = model_cfg.use_pos_embed
        if model_cfg.use_pos_embed:
            self.position_embedding = PositionalEmbedding(d_model=self.hidden_dim)

        # Fedformer layers
        self.layers = nn.ModuleList(
            [
                FEDformerLayer(
                    hidden_dim=self.hidden_dim,  # timestep embed dim
                    seq_len=input_dim,
                    model_cfg=model_cfg,
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        # FCN head; we aggregate timestemps in Fedformer
        fcn_dim = compute_final_dimension(self.hidden_dim, input_dim, model_cfg.pooling)
        self.final_head = BasicMLP(
            in_size=fcn_dim,
            out_size=model_cfg.n_classes,
            config=model_cfg.head_config,
        )

    def forward(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = batch[0]

        b, _, t = x.shape
        # (b, r, t) -> (b, t, r)
        out = x.transpose(1, 2)

        # (b, t, r) -> (b, t, D)
        if self.use_token_embed:
            out = self.token_embedding(out)
        if self.use_pos_embed:
            out += self.position_embedding(out.size(1))

        for layer in self.layers:
            out = layer(out)

        # out = out.transpose(1, 2) -> (b, r, t)

        # pooling
        out = pool(out, self.pooling)

        # fnn head
        out = self.final_head(out)

        return out
