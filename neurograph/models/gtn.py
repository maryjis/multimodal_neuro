import torch
from torch import nn

from neurograph.config.config import GTNConfig, CustomGTNConfig
from neurograph.models.embed import PositionalEmbedding
from neurograph.models.fedformer.fedformer import FEDformerLayer
from neurograph.models.conformer import ConformerLayer
from neurograph.models.transformers import TransformerBlock, Transformer

from neurograph.models.transformers import compute_final_dimension
from neurograph.models.mlp import BasicMLP


class GTN(nn.Module):
    """Gated Transformer Network"""

    def __init__(
        self,
        input_dim: int,  # time series length
        num_nodes: int,  # num ROIs
        model_cfg: GTNConfig,
    ):
        super().__init__()

        self.num_rois = num_nodes
        self.num_timesteps = input_dim

        self.pooling = model_cfg.pooling

        # if we don't use token embeddings, then we use original time series
        self.hidden_dim = model_cfg.hidden_dim

        # token embedding
        self.embed_timestep = nn.Linear(self.num_rois, self.hidden_dim)
        self.embed_channel = nn.Linear(self.num_timesteps, self.hidden_dim)

        # positional embedding
        self.use_pos_embed = model_cfg.use_pos_embed
        if model_cfg.use_pos_embed:
            self.position_embedding = PositionalEmbedding(d_model=self.hidden_dim)

        # Transformer layers of two "towers"
        self.timestep_layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=self.hidden_dim,
                    msa_cfg=Transformer.build_msa_cfg(model_cfg),
                    mlp_cfg=Transformer.build_mlp_cfg(model_cfg),
                )
                for _ in range(model_cfg.num_layers)
            ]
        )
        self.channel_layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=self.hidden_dim,
                    msa_cfg=Transformer.build_msa_cfg(model_cfg),
                    mlp_cfg=Transformer.build_mlp_cfg(model_cfg),
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        final_size = (self.num_rois + self.num_timesteps) * self.hidden_dim
        self.gate = torch.nn.Linear(final_size, 2)
        self.output_linear = torch.nn.Linear(final_size, model_cfg.n_classes)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = batch[0]

        t_x = x.transpose(1, 2)  # (b, r, t) -> (b, t, r)
        c_x = x

        # compute token embeddings
        t_x = self.embed_timestep(t_x)
        c_x = self.embed_channel(c_x)

        # to timestep data add positional encoding
        if self.use_pos_embed:
            t_x += self.position_embedding(t_x.size(1))

        for layer in self.timestep_layers:
            t_x = layer(t_x)
        for layer in self.channel_layers:
            c_x = layer(c_x)

        # "concat pooling" for both outputs
        t_x = t_x.reshape(t_x.size(0), -1)
        c_x = c_x.reshape(c_x.size(0), -1)

        # concat outputs
        cat_out = torch.cat([t_x, c_x], dim=-1)

        # compute gate coefficients
        gate = torch.softmax(self.gate(cat_out), dim=-1)

        # multiply outputs by gate coefficients elementwise and concat
        out = torch.cat([t_x * gate[:, 0:1], c_x * gate[:, 1:2]], dim=-1)

        return self.output_linear(out)  # final linear layer


class CustomGTN(nn.Module):
    """Custom Gated Transformer Network
    We always use token embeddings here, so `self.hidden_dim` is always equal
    to `model_cfg.hidden_dim`.
    """

    def __init__(
        self,
        input_dim: int,  # time series length
        num_nodes: int,  # num ROIs
        model_cfg: CustomGTNConfig,
    ):
        super().__init__()

        self.num_rois = num_nodes
        self.num_timesteps = input_dim

        self.pooling = model_cfg.pooling
        self.hidden_dim = model_cfg.hidden_dim

        # token embedding
        self.embed_timestep = nn.Linear(self.num_rois, self.hidden_dim)
        self.embed_channel = nn.Linear(self.num_timesteps, self.hidden_dim)

        # positional embedding
        self.use_pos_embed = model_cfg.use_pos_embed
        if model_cfg.use_pos_embed:
            self.position_embedding = PositionalEmbedding(d_model=self.hidden_dim)

        # Transformer layers of two "towers"
        self.timestep_layers = nn.ModuleList(
            [
                self.build_timestep_layer(self.num_timesteps, model_cfg)
                for _ in range(model_cfg.num_layers)
            ]
        )
        self.channel_layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=self.hidden_dim,
                    msa_cfg=Transformer.build_msa_cfg(model_cfg),
                    mlp_cfg=Transformer.build_mlp_cfg(model_cfg),
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        final_size = (self.num_rois + self.num_timesteps) * self.hidden_dim
        self.gate = torch.nn.Linear(final_size, 2)
        self.output_linear = torch.nn.Linear(final_size, model_cfg.n_classes)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = batch[0]

        t_x = x.transpose(1, 2)  # (b, r, t) -> (b, t, r)
        c_x = x

        # compute token embeddings
        t_x = self.embed_timestep(t_x)
        c_x = self.embed_channel(c_x)

        # to timestep data add positional encoding
        if self.use_pos_embed:
            t_x += self.position_embedding(t_x.size(1))

        for layer in self.timestep_layers:
            t_x = layer(t_x)
        for layer in self.channel_layers:
            c_x = layer(c_x)

        # "concat pooling" for both outputs
        t_x = t_x.reshape(t_x.size(0), -1)
        c_x = c_x.reshape(c_x.size(0), -1)

        # concat outputs
        cat_out = torch.cat([t_x, c_x], dim=-1)

        # compute gate coefficients
        gate = torch.softmax(self.gate(cat_out), dim=-1)

        # multiply outputs by gate coefficients elementwise and concat
        out = torch.cat([t_x * gate[:, 0:1], c_x * gate[:, 1:2]], dim=-1)

        return self.output_linear(out)  # final linear layer

    def build_timestep_layer(self, num_timesteps: int, cfg: CustomGTNConfig):
        if cfg.timestep_model_config.name == "Conformer":
            return self._build_conformer_layer(cfg)
        if cfg.timestep_model_config.name == "FEDformer":
            return self._build_fedformer_layer(num_timesteps, cfg)
        raise ValueError("Unknown timestep layer name")  # pragma: no cover

    def _build_conformer_layer(self, cfg: CustomGTNConfig) -> ConformerLayer:
        submodel_cfg = cfg.timestep_model_config
        return ConformerLayer(
            input_dim=cfg.hidden_dim,
            ffn_dim=cfg.hidden_dim,
            act_func_name=submodel_cfg.act_func_name,
            act_func_params=submodel_cfg.act_func_params,
            num_heads=submodel_cfg.num_heads,
            relative_key=submodel_cfg.relative_key,
            depthwise_conv_kernel_size=submodel_cfg.depthwise_conv_kernel_size,
            num_channels_multiplier=submodel_cfg.num_channels_multiplier,
            dropout=submodel_cfg.dropout,
            use_group_norm=submodel_cfg.use_group_norm,
            convolution_first=submodel_cfg.convolution_first,
            return_attn=submodel_cfg.return_attn,
        )

    def _build_fedformer_layer(self, num_timesteps: int, cfg: CustomGTNConfig):
        return FEDformerLayer(
            hidden_dim=cfg.hidden_dim,  # timestep embed dim
            seq_len=num_timesteps,  # timeseries length
            model_cfg=cfg.timestep_model_config,
        )
