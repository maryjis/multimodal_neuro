from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:

        """Get tensor w/ positional embedding of length `length`

        Returns:
            torch.Tensor: shape (1, length, d_model)
        """
        return self.pe[:, :length]

    def __repr__(self):  # pragma: no cover
        return f"PositionalEmbedding(d_model={self.d_model}, max_len={self.max_len})"


class LinearTokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: shape (b, t, r)"""
        return self.lin(x)


class ConvTokenEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        depthwise: bool,
        kernel_size: int = 3,
    ):
        """input_dim (int): number of ROIs"""
        super().__init__()

        if depthwise and d_model % input_dim != 0:  # pragma: no cover
            m = (
                "ConvTokenEmbedding: "
                "`d_model` must be multiple of `input_dim` if `depthwise=True`"
            )
            raise ValueError(m)

        # 1d conv w/ curcular padding w/o bias
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
            groups=input_dim if depthwise else 1,
        )
        # init conv1d weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """apply 1d conv along temporal dimension
        Args:
            x (torch.Tensor): shape (b, t, r)

        Returns:
            torch.Tensor: shape (b, t, r)
        """
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
