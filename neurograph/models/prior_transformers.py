""" Module provides implementation of Vanilla Transformer """

from dataclasses import dataclass

import torch
from torch import nn

from neurograph.config import MLPConfig, MLPlayer, PriorTransformerConfig
from neurograph.models.mlp import BasicMLP
from neurograph.models.transformers import (
    MSA,
    MSAConfig,
    MSAOutput,
    compute_final_dimension,
)


@dataclass
class PriorMSAConfig(MSAConfig):
    """Config for initializing PriorMSA instance"""

    alpha: float = 1.0
    trainable_alpha: bool = True


class PriorMSA(MSA):
    """Computes multihead attention for a tensor x of size [b, n, d]
    given prior attention weights [n, n]
    """

    def __init__(
        self,
        input_dim: int,
        cfg: PriorMSAConfig,
    ):
        super().__init__(input_dim, cfg)
        self.alpha = nn.Parameter(
            torch.tensor(cfg.alpha),
            requires_grad=cfg.trainable_alpha,
        )

    def forward(
        self,
        x: torch.Tensor,
        prior_attn: torch.Tensor,
    ) -> MSAOutput:
        """Compute multihead attention w/ priot attention matrix"""

        # (b, n, d)
        b, n, _ = x.shape

        # project X to Q, K, V -> (b, n, h, p)
        q, k, v = self.project_to_qkv(x)

        # compute raw_scores
        raw_scores = torch.einsum("bihp,bjhp->bhij", q, k)

        # add prior attention weights
        prior_attn_reshaped = prior_attn.unsqueeze(1).repeat(
            1, raw_scores.shape[1], 1, 1
        )
        raw_scores += self.alpha * prior_attn_reshaped

        # normalize each head output
        scores = torch.softmax(raw_scores * self.factor, dim=-1)

        # save attention matrices for each head (used for debug)
        saved_scores = None
        if self.return_attn:
            saved_scores = scores.clone()  # pragma: no cover

        # apply dropout to attention matrix
        scores = self.dropout(scores)

        # compute final embeddings by multiplying `v` by `scores`
        out = torch.einsum("bhij,bjhp->bihp", scores, v)

        # 'concat' each head output
        return MSAOutput(x=out.reshape(b, n, -1), attn=saved_scores)


class PriorTransformerBlock(nn.Module):
    """NB: input_dim must be equal to hidden_dim"""

    def __init__(
        self,
        input_dim: int,
        msa_cfg: PriorMSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()
        self.hidden_dim = msa_cfg.hidden_dim
        assert (
            msa_cfg.hidden_dim == input_dim
        ), "First project input to hidden before sending it to TransformerBlock"

        self.msa = PriorMSA(input_dim, msa_cfg)  # only difference from TransformerBlock
        self.mlp = BasicMLP(
            in_size=self.hidden_dim,
            out_size=self.hidden_dim,
            config=mlp_cfg,
        )
        self.ln_1 = nn.LayerNorm([self.hidden_dim])
        self.ln_2 = nn.LayerNorm([self.hidden_dim])

    def forward(self, x: torch.Tensor, prior_attn: torch.Tensor):
        """Apply one transformer layer to `x`"""

        z_1 = self.ln_1(x)
        s_1 = x + self.msa(z_1, prior_attn).x  # sum_1

        z_2 = self.ln_2(s_1)
        s_2 = s_1 + self.mlp(z_2)  # sum_2
        return s_2


class PriorTransformer(nn.Module):
    """Transformer that uses PriorMSA instead of MSA"""

    def __init__(
        self,
        # comes from dataset
        input_dim: int,
        num_nodes: int,  # used for concat pooling
        model_cfg: PriorTransformerConfig,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.pooling = model_cfg.pooling
        num_classes = model_cfg.n_classes

        self.lin_proj = nn.Linear(input_dim, model_cfg.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                PriorTransformerBlock(
                    model_cfg.hidden_dim,
                    self.build_msa_cfg(model_cfg),
                    self.build_mlp_cfg(model_cfg),
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        fcn_dim = compute_final_dimension(
            model_cfg.hidden_dim, self.num_nodes, model_cfg.pooling
        )
        self.fcn = BasicMLP(
            in_size=fcn_dim, out_size=num_classes, config=model_cfg.head_config
        )

    def forward(self, batch):
        """Compute class predictions"""
        # x, y
        x, priot_attn, _ = batch
        # project to hidden_dim
        out = self.lin_proj(x)

        # go thru transformer layers
        for block in self.blocks:
            out = block(out, priot_attn)

        # pool
        if self.pooling == "concat":
            out = out.reshape(out.size(0), -1)
        elif self.pooling == "mean":  # pragma: no cover
            out = out.mean(axis=1)
        else:  # pragma: no cover
            out = out.sum(axis=1)

        # infer mlp head
        out = self.fcn(out)

        return out

    @staticmethod
    def build_msa_cfg(cfg: PriorTransformerConfig) -> PriorMSAConfig:
        """Create PriorMSAConfig instance from TransformerConfig"""
        return PriorMSAConfig(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.attn_dropout,
            return_attn=cfg.return_attn,
            alpha=cfg.alpha,
            trainable_alpha=cfg.trainable_alpha,
        )

    @staticmethod
    def build_mlp_cfg(cfg: PriorTransformerConfig):
        """Create config for 2-layer MLP from TransformerConfig"""
        # 2-layer MLP
        return MLPConfig(
            # no act func on the output of MLP
            layers=[
                MLPlayer(
                    out_size=int(cfg.hidden_dim * cfg.mlp_hidden_multiplier),
                    dropout=cfg.mlp_dropout,
                    act_func=cfg.mlp_act_func,  # put class name here
                    act_func_params=cfg.mlp_act_func_params,
                ),
            ],
        )
