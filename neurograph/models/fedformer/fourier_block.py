"""
Adapted from https://github.com/MAZiqing/FEDformer

MIT License

Copyright (c) 2021 DAMO Academy @ Alibaba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_activation(attn_m: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "tanh":
        return attn_m.tanh()
    if activation == "softmax":
        res = torch.softmax(abs(attn_m), dim=-1)
        return torch.complex(res, torch.zeros_like(res))
    raise Exception(
        "{} actiation function is not implemented".format(activation)
    )  # pragma: no cover


def get_frequency_modes(seq_len, modes=64, mode_select_method="random"):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;

    max num of modes limited to seq_len // 2 since
    the spectrum of real-valued time series has mirror symmetry
    """

    modes = min(modes, seq_len // 2)
    if mode_select_method == "random":
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))  # pragma: no cover
    # sort mode indices
    index.sort()

    return index


def compl_mul1d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication"""
    # x, w -> out
    # (batch, num_heads, in_channel), (num_heads, in_channel, out_channel) ->
    # -> (batch, num_heads, out_channel)
    return torch.einsum("bhi,hio->bho", x, weights)


class FEBf(nn.Module):
    """
    1D Fourier block. It performs representation learning on frequency domain,
    it does FFT, linear transform, and Inverse FFT.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_heads: int,
        num_modes: int = 0,
        mode_select_method: str = "random",
    ):
        super().__init__()
        self.d_head = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.modes = num_modes
        self.mode_select_method = mode_select_method

        # get modes on frequency domain
        # we **FIX** mode indices at the moment when we initialize the block
        self.index = get_frequency_modes(
            seq_len, modes=num_modes, mode_select_method=mode_select_method
        )
        logger.debug("modes={}, index={}".format(num_modes, self.index))

        self.scale = 1 / (d_model * d_model)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                num_heads,  #  num heads
                d_model // num_heads,  # hidden_dim per head
                d_model // num_heads,  # dim per head
                len(self.index),
                dtype=torch.cfloat,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FEB-f block
        Args:
            x (torch.Tensor): input after multihead projection of shape (b, t, h, d)

        Returns:
            torch.Tensor: output of shape (b, t, h, d)
        """
        b, t, h, d = x.shape

        # (b, t, h, d) -> (b, h, d, t)
        x = x.permute(0, 2, 3, 1)

        # Compute Fourier coefficients (apply per L)
        x_ft = torch.fft.rfft(x, dim=-1)

        # Perform Fourier neural operations
        # `L // 2 + 1` is output length after rfft
        # Also this input length is expected by irfft to get an outpur of length `L`

        out_ft = torch.zeros(b, h, d, t // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = compl_mul1d(
                x_ft[:, :, :, i], self.weights1[:, :, :, wi]
            )

        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # (b, h, e, l)

        # reshape back to original shape
        return x.permute(0, 3, 1, 2)

    def __repr__(self):  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"d_head={self.d_head}, "
            f"seq_len={self.seq_len}, "
            f"num_heads={self.num_heads}, "
            f"modes={self.modes}, "
            f"mode_select_methods={self.mode_select_method}"
            ")"
        )


class FEAf(nn.Module):
    """
    1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_heads: int,
        num_modes: int = 0,
        mode_select_method: str = "random",
        activation: str = "tanh",
    ):
        super().__init__()

        self.activation = activation

        # get modes for queries and keys (values) of frequency domain
        self.index_q = get_frequency_modes(
            seq_len, modes=num_modes, mode_select_method=mode_select_method
        )
        self.index_kv = get_frequency_modes(
            seq_len, modes=num_modes, mode_select_method=mode_select_method
        )

        logger.debug("modes_q={}, index_q={}".format(len(self.index_q), self.index_q))
        logger.debug(
            "modes_kv={}, index_kv={}".format(len(self.index_kv), self.index_kv)
        )

        self.scale = 1 / (num_heads * d_model * num_heads * d_model)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                num_heads,  #  num heads
                d_model // num_heads,  # hidden_dim per head
                d_model // num_heads,  # hidden_dim per head
                len(self.index_q),
                dtype=torch.cfloat,
            )
        )

    def fft_select_modes(self, x: torch.Tensor, mode_index: list[int]) -> torch.Tensor:
        """Compute 1d fft then select modes
        Args:
            x: shape (b, h, d, t)

        Returns:
            torch.Tensor of shape (b, h, d, num_modes)
        """

        b, h, d, _ = x.shape
        out = torch.zeros(b, h, d, len(mode_index), device=x.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(x, dim=-1)

        # select and copy modes to the out tensor
        for i, j in enumerate(mode_index):
            out[:, :, :, i] = xq_ft[:, :, :, j]

        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:

        b, t, h, d = q.shape
        # (b, t, h, d) -> (b, h, d, t)
        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients for q, k, v
        xq_ft_ = self.fft_select_modes(xq, self.index_q)
        xk_ft_ = self.fft_select_modes(xk, self.index_kv)
        xv_ft_ = self.fft_select_modes(xv, self.index_kv)

        # perform attention mechanism on frequency domain
        # -> (b, h, modes q, modes k)
        xqk_ft = torch.einsum("bhdx,bhdy->bhxy", xq_ft_, xk_ft_)

        xqk_ft = apply_activation(xqk_ft, self.activation)

        # multiply values by attention scores
        xqkv_ft = torch.einsum("bhxy,bhdy->bhdx", xqk_ft, xv_ft_)

        # multiply values by weight matrix
        xqkvw = torch.einsum("bhdx,hdox->bhox", xqkv_ft, self.weights1)

        # create empty tensor for iFFT
        out_ft = torch.zeros(b, h, d, t // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        # Return to time domain
        out = torch.fft.irfft(out_ft * self.scale, n=xq.size(-1))

        # reshape back to original shape
        return out.permute(0, 3, 1, 2)
