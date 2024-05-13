""" Module provides implementation of Multimodal version of Vanilla Transformer
    with different cross-modality attention mechanisms
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from neurograph.config import (
    MLPConfig,
    MLPlayer,
    MultiModalTransformerConfig,
    TransformerConfig,
)
from neurograph.models.mlp import BasicMLP
from neurograph.models.transformers import MSA, MSAConfig, Transformer, TransformerBlock


def compute_raw_attn_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute raw multihead attention scores (multiply Q and K)

    Args:
        q (torch.Tensor): queries tensor
        k (torch.Tensor): keys tensor

    Returns:
        torch.Tensor: raw attention matrices for each head
    """
    return torch.einsum("bihp,bjhp->bhij", q, k)


@dataclass
class MSACrossAttentionOutput:
    """Dataclass that stores cross-attention
    operation output
    """

    x_1: torch.Tensor
    x_2: torch.Tensor
    attn_1: Optional[torch.Tensor] = None
    attn_2: Optional[torch.Tensor] = None


class MSACrossAttention(MSA):
    """Multihead cross attention block that accepts two modalities x_1 and x_2
    and computes pairwise attention weights
    """

    def __init__(
        self,
        input_dim: int,
        cfg: MSAConfig,
    ):
        super().__init__(input_dim, cfg)

        # add extra linear layers for x_2
        self.q_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_lin_head2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def project_to_qkv_head1(self, x: torch.Tensor):
        """project token embeddings to query, keys and values"""
        return self.project_to_qkv(x)

    def project_to_qkv_head2(self, x: torch.Tensor):
        """project token embeddings to query, keys and values
        (weird and ugly hack for 2 modalities)
        """
        # (b, n, d)
        b, n, _ = x.shape
        h = self.num_heads
        p = self.head_dim

        q = self.q_lin_head2(x).reshape(b, n, h, p)
        k = self.k_lin_head2(x).reshape(b, n, h, p)
        v = self.v_lin_head2(x).reshape(b, n, h, p)

        return q, k, v

    # pylint: disable=too-many-locals
    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> MSACrossAttentionOutput:
        # (b, n, d)
        b, n, _ = x_1.shape
        # h is self.num_heads

        # project X1 to Q1, K1, V1 -> (b, n, h, p)
        q_1, k_1, v_1 = self.project_to_qkv_head1(x_1)

        # project X2 to Q2, K2, V2 -> (b, n, h, p)
        q_2, k_2, v_2 = self.project_to_qkv_head2(x_2)

        # compute Q_2 @ K_1
        raw_scores_1 = compute_raw_attn_scores(q_2, k_1)

        # compute Q_1 @ K_2
        raw_scores_2 = compute_raw_attn_scores(q_1, k_2)

        # normalize each head output
        scores_1 = torch.softmax(raw_scores_1 * self.factor, dim=-1)
        scores_2 = torch.softmax(raw_scores_2 * self.factor, dim=-1)

        # save attention matrices for each x_i
        saved_scores_1, saved_scores_2 = None, None
        if self.return_attn:  # pragma: no cover
            saved_scores_1 = scores_1.clone()
            saved_scores_2 = scores_2.clone()

        # apply dropout to attention matrices
        scores_1 = self.dropout(scores_1)
        scores_2 = self.dropout2(scores_2)

        # compute final embeddings
        out1 = torch.einsum("bhij,bjhp->bihp", scores_1, v_1)
        out2 = torch.einsum("bhij,bjhp->bihp", scores_2, v_2)

        # 'concat' each head output
        return MSACrossAttentionOutput(
            x_1=out1.reshape(b, n, -1),
            x_2=out2.reshape(b, n, -1),
            attn_1=saved_scores_1,
            attn_2=saved_scores_2,
        )


class FFN(nn.Module):
    """FFN from TransformerBlock adapted for CrossAttentionBlock"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()

        self.mlp = BasicMLP(
            in_size=input_dim,
            out_size=output_dim,
            config=mlp_cfg,
        )

        # We need only one LN here, because we get
        # attention weights from the outside already normalized
        self.ln_1 = nn.LayerNorm([input_dim])

    # pylint: disable=missing-function-docstring
    def forward(self, x: torch.Tensor, cross_attn: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/pdf/2002.04745.pdf
        s_1 = x + cross_attn  # sum_1

        z_2 = self.ln_1(s_1)
        s_2 = s_1 + self.mlp(z_2)  # sum_2

        return s_2


class CrossAttentionBlock(nn.Module):
    """Transformer block where self-attention is replaced by CrossAttention"""

    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()

        assert input_dim_1 == input_dim_2
        self.hidden_dim = msa_cfg.hidden_dim

        self.ln1 = nn.LayerNorm([self.hidden_dim])
        self.ln2 = nn.LayerNorm([self.hidden_dim])

        self.msa = MSACrossAttention(input_dim_1, msa_cfg)

        self.head1 = FFN(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            mlp_cfg=mlp_cfg,
        )
        self.head2 = FFN(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            mlp_cfg=mlp_cfg,
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        # first, normalize each modality independently
        z_1 = self.ln1(x_1)
        z_2 = self.ln1(x_2)

        # compute embeddings via cross-attention:
        # out.x_1 = softmax(Q_2 @ K_1) @ V_1
        # out_.x2 = softmax(Q_1 @ K_2) @ V_2
        out = self.msa(z_1, z_2)

        # LN + MLP
        s_1 = self.head1(x_1, out.x_1)
        s_2 = self.head2(x_2, out.x_2)

        return torch.cat((s_1, s_2), dim=2)


class HierarchicalAttentionBlock(nn.Module):
    """Hierarchical attention transformer block for 2 modalities.
    Runs 2 transformer blocks in parallel for each X
    and then concats outputs
    """

    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        msa_cfg: MSAConfig,
        mlp_cfg: MLPConfig,
    ):
        super().__init__()

        self.block_1 = TransformerBlock(input_dim_1, msa_cfg, mlp_cfg)
        self.block_2 = TransformerBlock(input_dim_2, msa_cfg, mlp_cfg)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        """Compute transformer outputs for 2 modalities and concat them"""
        z_1 = self.block_1(x_1)
        z_2 = self.block_2(x_2)

        return torch.cat((z_1, z_2), dim=2)


class MultiModalTransformer(nn.Module):
    """Multimodal Transformer w/ early fusion: first modalities are fused
    and then sent to a standart Transfomer model
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        num_nodes_1: int,  # used for concat pooling
        num_nodes_2: int,
        model_cfg: MultiModalTransformerConfig,
    ):
        super().__init__()
        # assert input_dim_1 == input_dim_2
        assert num_nodes_1 == num_nodes_2

        self.attn_type = model_cfg.attn_type
        self.make_projection = True

        # we fuse modalitities so we obtain embeddings of size
        # model_cfg.hidden_dim
        transformer_hidden_dim = model_cfg.hidden_dim

        if self.attn_type == "concat":
            transformer_hidden_dim = (
                model_cfg.hidden_dim
                if self.make_projection
                else input_dim_1 + input_dim_2
            )
        elif self.attn_type in ["sum", "multiply"]:
            transformer_hidden_dim = (
                model_cfg.hidden_dim if self.make_projection else input_dim_1
            )

        self.transformer = Transformer(transformer_hidden_dim, num_nodes_1, model_cfg)

        # project to half of the final tranformer hidden_dim if we use concat, cross_attm,
        project_dim = (
            model_cfg.hidden_dim
            if self.attn_type in ["sum", "multiply"]
            else model_cfg.hidden_dim // 2
        )
        self.lin_proj1 = nn.Linear(input_dim_1, project_dim)
        self.lin_proj2 = nn.Linear(input_dim_2, project_dim)

        dim_1 = project_dim if self.make_projection else input_dim_1
        dim_2 = project_dim if self.make_projection else input_dim_2

        attn_block_params = [
            dim_1,
            dim_2,
            self.build_msa_attn_cfg(model_cfg),
            self.build_mlp_attn_cfg(model_cfg),
        ]

        if self.attn_type == "hierarchical":
            self.mm_block = HierarchicalAttentionBlock(*attn_block_params)
        elif self.attn_type == "cross-attention":
            self.mm_block = CrossAttentionBlock(*attn_block_params)

    def forward(self, batch):
        """Project inputs (optionally) then combine modalities `x_1` and `x_2`
        into one embedding `x` via different mechanisms
        (concat, sum, hierarchical attention, cross-attention)
        """

        x_fmri, x_dti, y = batch
        # project to the dimension first
        x_fmri = self.lin_proj1(x_fmri)
        x_dti = self.lin_proj2(x_dti)

        if self.attn_type == "concat":
            x = torch.cat((x_fmri, x_dti), dim=2)
        elif self.attn_type == "sum":
            x = x_fmri + x_dti
        elif self.attn_type == "multiply":
            x = x_fmri * x_dti
        elif self.attn_type in ["hierarchical", "cross-attention"]:
            x = self.mm_block(x_fmri, x_dti)
        else:
            raise ValueError("Invalid `attn_type`")  # pragma: no cover

        return self.transformer((x, y))

    @staticmethod
    def build_msa_attn_cfg(cfg: TransformerConfig):
        """Build Config for MSA or MSACrossAttention module"""
        return MSAConfig(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim // 2,
            dropout=cfg.attn_dropout,
            return_attn=cfg.return_attn,
        )

    @staticmethod
    def build_mlp_attn_cfg(cfg: TransformerConfig):
        """Build config for 2-layer MLP"""
        return MLPConfig(
            # no act func on the output of MLP
            layers=[
                MLPlayer(
                    out_size=int((cfg.hidden_dim // 2) * cfg.mlp_hidden_multiplier),
                    dropout=cfg.mlp_dropout,
                    act_func=cfg.mlp_act_func,  # put class name here
                    act_func_params=cfg.mlp_act_func_params,
                ),
            ],
        )
