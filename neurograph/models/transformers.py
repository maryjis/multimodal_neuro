""" Module provides implementation of Vanilla Transformer """

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from neurograph.config import MLPConfig, MLPlayer, TransformerConfig
from neurograph.models.mlp import BasicMLP

max_pos_embed_distance = 1024


def compute_final_dimension(hidden_dim: int, num_nodes: int, pooling: str) -> int:
    """Function to compute final dimension of transformer block output based on pooling type"""
    if pooling == "concat":
        return num_nodes * hidden_dim
    if pooling in ("mean", "sum"):  # pragma: no cover
        return hidden_dim
    raise ValueError("Unknown pooling type!")  # pragma: no cover


def pool(out: torch.Tensor, pooling_type: str) -> torch.Tensor:
    if pooling_type == "concat":
        out = out.reshape(out.size(0), -1)
    elif pooling_type == "mean":  # pragma: no cover
        out = out.mean(axis=1)
    else:  # pragma: no cover
        out = out.sum(axis=1)

    return out


@dataclass
class MSAConfig:
    """Multihead attention block config"""

    num_heads: int
    hidden_dim: int  # token embedding dim after MSA
    return_attn: bool
    dropout: float = 0.0
    # use relative pos embeddings for keys
    relative_key: bool = False


@dataclass
class MSAOutput:
    """Dataclass that store Multihead attention output(s)"""

    x: torch.Tensor
    attn: Optional[torch.Tensor] = None


class SingleMultiheadProjection(nn.Module):
    """Module that projects `x` (b, t, D) tensor into multihead version (b, t, h, d)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be multiple of num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """(b, n, D) -> (b, n, h, d)"""
        b, n, _ = x.shape

        return (self.lin(x).reshape(b, n, self.num_heads, self.head_dim),)


class ProjectBeforeMSA(nn.Module):
    """Module that projects `x` vector into `q`, `k`, `v` vectors"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be multiple of num_heads"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.q_lin = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_lin = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_lin = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input `x` into `q`, `k`, `v` tensors"""
        # (b, n, d)
        b, n, _ = x.shape
        h = self.num_heads
        p = self.head_dim

        q = self.q_lin(x).reshape(b, n, h, p)
        k = self.k_lin(x).reshape(b, n, h, p)
        v = self.v_lin(x).reshape(b, n, h, p)

        return q, k, v


class MSA(nn.Module):
    """Computes multihead attention for a tensor x of size [b, n, d]"""

    def __init__(
        self,
        input_dim: int,
        cfg: MSAConfig,
    ):
        super().__init__()
        assert (
            cfg.hidden_dim % cfg.num_heads == 0
        ), "hidden_dim must be multiple of num_heads"

        self.input_dim = input_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.dropout_rate = cfg.dropout  # dropout after self-attention
        self.return_attn = cfg.return_attn
        self.factor = 1 / math.sqrt(self.head_dim)

        self.project_to_qkv = ProjectBeforeMSA(
            self.input_dim, self.hidden_dim, self.num_heads
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        self.relative_key = cfg.relative_key
        if self.relative_key:
            # shared among heads
            self.distance_embedding = nn.Embedding(
                2 * max_pos_embed_distance - 1, self.head_dim
            )

    def forward(self, x: torch.Tensor) -> MSAOutput:
        """Compute multihead attention"""

        # (b, n, d)
        b, n, _ = x.shape

        # project X to Q, K, V -> (b, n, h, p)
        q, k, v = self.project_to_qkv(x)

        # compute raw_scores
        raw_scores = torch.einsum("bihd,bjhd->bhij", q, k)

        if self.relative_key:
            # compute relative differences
            seq_length = n
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=x.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=x.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + max_pos_embed_distance - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=q.dtype
            )  # fp16 compatibility

            # add relative positional embeddings to raw scores
            # relative_position_scores = torch.einsum("bhld,lrd->bhlr", q, positional_embedding)
            relative_position_scores = torch.einsum(
                "bihd,ird->bhir", q, positional_embedding
            )
            raw_scores = raw_scores + relative_position_scores

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


class TransformerBlock(nn.Module):
    """NB: input_dim must be equal to hidden_dim"""

    def __init__(
        self,
        input_dim: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()
        self.hidden_dim = msa_cfg.hidden_dim
        assert (
            msa_cfg.hidden_dim == input_dim
        ), "First project input to hidden before sending it to TransformerBlock"

        self.msa = MSA(input_dim, msa_cfg)
        self.mlp = BasicMLP(
            in_size=self.hidden_dim,
            out_size=self.hidden_dim,
            config=mlp_cfg,
        )
        self.ln_1 = nn.LayerNorm([self.hidden_dim])
        self.ln_2 = nn.LayerNorm([self.hidden_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one transformer layer to `x`"""
        # https://arxiv.org/pdf/2002.04745.pdf
        z_1 = self.ln_1(x)
        s_1 = x + self.msa(z_1).x  # sum_1

        z_2 = self.ln_2(s_1)
        s_2 = s_1 + self.mlp(z_2)  # sum_2

        return s_2


class Transformer(nn.Module):
    """Class that contains several stacked transformer block, pooling and final MLP"""

    def __init__(
        self,
        # comes from dataset
        input_dim: int,
        num_nodes: int,  # used for concat pooling
        model_cfg: TransformerConfig,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.pooling = model_cfg.pooling
        num_classes = model_cfg.n_classes

        self.lin_proj: nn.Module
        if input_dim != model_cfg.hidden_dim:
            self.lin_proj = nn.Linear(input_dim, model_cfg.hidden_dim)
        else:
            self.lin_proj = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
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
        x = batch[0]
        # project to hidden_dim
        out = self.lin_proj(x)

        # go thru transformer layers
        for block in self.blocks:
            out = block(out)

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
    def build_msa_cfg(cfg: TransformerConfig):
        """Create MSAConfig instance from TransformerConfig"""
        return MSAConfig(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.attn_dropout,
            return_attn=cfg.return_attn,
        )

    @staticmethod
    def build_mlp_cfg(cfg: TransformerConfig):
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
