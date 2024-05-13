"""Adapted from https://github.com/pytorch/audio/blob/main/torchaudio/models/conformer.py

BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from typing import Optional

import torch
from torch import nn

from neurograph.config.config import ConformerConfig
from neurograph.models.available_modules import available_activations
from neurograph.models.mlp import BasicMLP
from neurograph.models.transformers import compute_final_dimension, MSA, MSAConfig


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        act_func_name: str = "SiLU",
        act_func_params: Optional[dict] = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
        padding_mode: str = "circular",
    ) -> None:
        super().__init__()

        if (depthwise_kernel_size - 1) % 2 != 0:  # pragma: no cover
            raise ValueError(
                "depthwise_kernel_size must be odd to achieve 'SAME' padding."
            )
        self.layer_norm = torch.nn.LayerNorm(input_dim)

        act_params = act_func_params if act_func_params else {}
        act_func = available_activations[act_func_name](**act_params)

        # we perform temporal convolution here
        self.sequential = torch.nn.Sequential(
            # k=1 (pointwise) conv that just increases number of channels
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
                groups=input_dim,
            ),  # -> (B, )
            torch.nn.GLU(dim=1),
            # second depthwise convolution along temporal dimension
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=input_dim,
                bias=bias,
                padding_mode=padding_mode,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            act_func,
            # third pointwise conv returns original number of channels
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                groups=input_dim,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.sequential(x)

        return x.transpose(1, 2)  # (B, T, D)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        act_func_name: str = "SiLU",
        act_func_params: Optional[dict] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        act_params = act_func_params if act_func_params else {}
        act_func = available_activations[act_func_name](**act_params)

        # input_dim -> hidden_dim -> input_dim; SilU as act func in the middle
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            act_func,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(x)


class ConformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        act_func_name: str,
        act_func_params: Optional[dict],
        ffn_dim: int,
        num_heads: int,
        relative_key: bool,
        depthwise_conv_kernel_size: int,
        num_channels_multiplier: float,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        return_attn: bool = False,
        padding_mode: str = "circular",
    ):
        super().__init__()

        self.convolution_first = convolution_first

        self.ffn1 = _FeedForwardModule(
            input_dim, ffn_dim, act_func_name, act_func_params, dropout=dropout
        )

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        # init attention module
        msa_cfg = MSAConfig(
            num_heads=num_heads,
            hidden_dim=input_dim,
            dropout=dropout,
            return_attn=return_attn,
            relative_key=relative_key,
        )
        self.attn = MSA(input_dim=input_dim, cfg=msa_cfg)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            act_func_name=act_func_name,
            act_func_params=act_func_params,
            num_channels=int(input_dim * num_channels_multiplier),
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
            padding_mode=padding_mode,
        )
        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def _apply_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block w/ conv module
        x: of shape (B, T, D)
        """
        residual = x
        x = self.conv_module(x)
        x = residual + x

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input must have shape (b, t, r) (timesteps first, ROIs last)"""

        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        # internally proejcts x to q, k, v and compute MSA
        x = self.attn(x).x
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)

        return x


class Conformer(nn.Module):
    """Conformer never decreases the input dim, so we can use residual connections"""

    def __init__(
        self,  # comes from dataset
        input_dim: int,  # num of timesteps
        num_nodes: int,  # num of rois,
        model_cfg: ConformerConfig,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_classes = model_cfg.n_classes
        self.pooling = model_cfg.pooling

        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    num_nodes,  # Conformer expects (b, t, r) so input_dim == num_nodes
                    act_func_name=model_cfg.act_func_name,
                    act_func_params=model_cfg.act_func_params,
                    ffn_dim=model_cfg.ffn_dim,
                    num_heads=model_cfg.num_heads,
                    relative_key=model_cfg.relative_key,
                    depthwise_conv_kernel_size=model_cfg.depthwise_conv_kernel_size,
                    num_channels_multiplier=model_cfg.num_channels_multiplier,
                    dropout=model_cfg.dropout,
                    use_group_norm=model_cfg.use_group_norm,
                    convolution_first=model_cfg.convolution_first,
                    return_attn=model_cfg.return_attn,
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        fcn_dim = compute_final_dimension(input_dim, self.num_nodes, model_cfg.pooling)
        self.final_head = BasicMLP(
            in_size=fcn_dim,
            out_size=model_cfg.n_classes,
            config=model_cfg.head_config,
        )

    def forward(self, batch):
        x = batch[0]  # (b, r, t)
        out = x.transpose(1, 2)  # (b, t, r)

        for layer in self.layers:
            out = layer(out)

        # transpose back to timeseries last
        out = out.transpose(1, 2)  # (b, r, t)

        # pool
        if self.pooling == "concat":
            out = out.reshape(out.size(0), -1)
        elif self.pooling == "mean":  # pragma: no cover
            out = out.mean(axis=1)
        else:  # pragma: no cover
            out = out.sum(axis=1)

        # final mlp head
        out = self.final_head(out)

        return out
