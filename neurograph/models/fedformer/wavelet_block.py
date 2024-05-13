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


from typing import List, Tuple
import math

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

from .fourier_block import FEAf, FEBf, apply_activation
from .utils import get_filter


def pad(x: torch.Tensor) -> torch.Tensor:
    """pad `x` w/ zeroes so its length is some power of 2"""

    N = x.size(1)
    nl = pow(2, math.ceil(np.log2(N)))  # a power of 2 closest to N

    if N == nl:
        return x

    extra_x = x[:, 0 : nl - N, :, :]

    return torch.cat([x, extra_x], 1)


class FEBw(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(
        self,
        d_model: int = 1,  # total number of input features
        num_modes: int = 16,  # num of modes in low-pass filter
        k: int = 8,  # multiwavelet dimensionality
        c: int = 128,  # num of channels per each MW dim
        L: int = 0,
        base: str = "legendre",
    ):
        super().__init__()

        self.k = k
        self.c = c
        self.L = L
        self.d_model = d_model

        # projection before and after MWT
        self.lin_in = nn.Linear(d_model, c * k)
        self.lin_out = nn.Linear(c * k, d_model)

        # multiwavelet transform modules
        self.mwt = MWT1d(k, num_modes, L, c, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shape transformation:
        (b, t, h, d) ->
            (b, t, d_model) -> (b, t, c, k) -> (b, t, d_model) ->
        (b, t, h, d)
        """
        b, t, h, d = x.shape

        # reshape
        x = x.view(b, t, -1)

        # apply linear proj (d_model -> c * k), reshape to (b, t, c, k)
        out = self.lin_in(x).view(b, t, self.c, self.k)

        # wavelet decomposition/reconstruction
        out = self.mwt(out)

        # project back to d_model
        out = self.lin_out(out.view(b, t, -1))

        # reshape back to original multihead shape
        return out.reshape(b, t, h, d)


class FEAw(nn.Module):
    """
    1D Multiwavelet Self Attention layer
    """

    def __init__(
        self,
        d_model: int = 1,  # total number of input features
        num_modes: int = 16,  # num of modes in low-pass filter
        k: int = 8,  # multiwavelet dimensionality
        c: int = 128,  # num of channels per each MW dim
        L: int = 0,
        base: str = "legendre",
        activation: str = "tanh",
    ):
        super().__init__()

        self.c = c
        self.k = k
        self.L = L

        # filter matrices
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        # decomposition matrices
        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        # reconstruction matrices
        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

        self.attn_1 = FEAf_lowpass(
            d_model=d_model,
            num_modes=num_modes,
            activation=activation,
        )
        self.attn_2 = FEAf_lowpass(
            d_model=d_model,
            num_modes=num_modes,
            activation=activation,
        )
        self.attn_3 = FEAf_lowpass(
            d_model=d_model,
            num_modes=num_modes,
            activation=activation,
        )
        self.attn_4 = FEAf_lowpass(
            d_model=d_model,
            num_modes=num_modes,
            activation=activation,
        )

        self.T_0 = nn.Linear(k, k)

        # linear projection for q, k, v
        self.in_k = nn.Linear(d_model, c * k)
        self.in_q = nn.Linear(d_model, c * k)
        self.in_v = nn.Linear(d_model, c * k)

        # out projection: c * k -> d_model
        self.out = nn.Linear(c * k, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        b, t, h, dim = q.shape

        # reshape q, k, v to (B, N, d_model)
        q = q.view(b, t, -1)
        k = k.view(b, t, -1)
        v = v.view(b, t, -1)

        # Project embeddings to c*k and reshape to (b, n, c, k)
        q = self.in_q(q).view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.in_k(k).view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.in_v(v).view(v.shape[0], v.shape[1], self.c, self.k)

        # Pad to a length 2^x
        q, k, v = pad(q), pad(k), pad(v)

        # Coeff list for q, k, v
        Ud_q = torch.jit.annotate(List[Tensor], [])
        Ud_k = torch.jit.annotate(List[Tensor], [])
        Ud_v = torch.jit.annotate(List[Tensor], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        # coarsest scale s coefs
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose q, k, v
        for i in range(self.L):
            d, q = self.decompose_step(q)
            Ud_q += [q]
            Us_q += [d]
        for i in range(self.L):
            d, k = self.decompose_step(k)
            Ud_k += [k]
            Us_k += [d]
        for i in range(self.L):
            d, v = self.decompose_step(v)
            Ud_v += [v]
            Us_v += [d]

        # compute Ud, Us coefs for reconstruction from Ud, Us for q, k, v with FEAf
        for i in range(self.L):
            dq, sq = Ud_q[i], Us_q[i]
            dk, sk = Ud_k[i], Us_k[i]
            dv, sv = Ud_v[i], Us_v[i]

            Ud += [self.attn_1(sq, sk, sv) + self.attn_2(dq, dk, dv)]
            Us += [self.attn_3(sq, sk, sv)]

        # apply FEAf to values
        v = self.attn_4(q, k, v)

        # reconstruct
        for i in range(self.L - 1, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.reconstruct_step(v)

        # out projection, reshape back
        v = self.out(v[:, :t, :, :].reshape(b, t, -1))

        return v.reshape(b, t, h, dim)

    def decompose_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xa = torch.cat(
            [
                x[:, ::2, :, :],
                x[:, 1::2, :, :],
            ],
            -1,
        )
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)

        return d, s

    def reconstruct_step(self, x: torch.Tensor) -> torch.Tensor:
        B, N, c, ich = x.shape  # (B, N, c, k)

        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o

        return x


class MWT1d(nn.Module):
    """This module performs non-standart multiwavelet transform where
    d, s coef modified w/ FEB-f block (low-pass filter and linear transfrom for each mode).

    We deconstruct, then reconstruct input, so the shape is not changed
    """

    def __init__(
        self,
        k: int = 3,  # multiwavelet dim
        num_modes: int = 16,  # num of modes in FEB-f blocks
        L: int = 1,  # num of steps
        c: int = 4,  # num of channels per multiwavelet dim
        base: str = "legendre",
    ):
        super().__init__()

        self.k = k
        self.L = L

        # get filter matrices
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)

        # get reconstruction matrices
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        # clip small values
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        # low-pass filters
        self.A = sparseKernelFT1d(k, num_modes, c)
        self.B = sparseKernelFT1d(k, num_modes, c)
        self.C = sparseKernelFT1d(k, num_modes, c)

        self.T_0 = nn.Linear(k, k)

        # decomposition filters, concatenated
        self.register_buffer("ec_s", torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer("ec_d", torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        # reconstruction filters
        self.register_buffer("rc_e", torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer("rc_o", torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run decomposition and reconstruction
        (b, t, c, k) -> (b, t, c, k)
        """

        N = x.size(1)
        x = pad(x)

        Us, Ud, x = self.decompose(x)

        x = self.reconstruct(Us, Ud, x)

        return x[:, :N, :, :]  # return reconstructed input

    def decompose(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Perforfm `self.L` steps of decomposition, return Ud, Us and s_L coefficient"""

        # lists of low-pass filtered d, s coef for each step
        Ud = torch.jit.annotate(List[torch.Tensor], [])
        Us = torch.jit.annotate(List[torch.Tensor], [])

        # at the first step we just use the original signal.
        # basically we're mixing signals with FEB-f blocks,
        # but at different resolutions
        for _ in range(self.L):
            # compute d, s coefficients
            d, s = self.decompose_step(x)

            # apply low-pass filters to MWT coefficients
            Ud += [self.A(d) + self.B(s)]
            Us += [self.C(d)]

            # update x, set it to s
            x = s

        x = self.T_0(x)  # the coarsest scale transform

        return Ud, Us, x

    def reconstruct(
        self,
        Us: list[torch.Tensor],
        Ud: list[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """reconstruct (from coarsest to finest scale)"""

        for i in range(self.L - 1, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.reconstruct_step(x)

        return x

    def decompose_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One iteration of fast multiwavelet transform

        Returns: d, s coefficients
        """

        # donwsample `x` by taking half of points
        # starting from point 0 and 1
        # concat two downsampled signals per each step
        # so the last dim == 2*k
        xa = torch.cat(
            [
                x[:, ::2, :, :],
                x[:, 1::2, :, :],
            ],
            -1,
        )

        # multuply by filters to get MWT coefficients
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)

        return d, s

    def reconstruct_step(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, c, k) -> (B, 2 * N, c, k)

        Reconstruct s_{i+1} coef from s_i and d_i.
        Here the input length is doubled.
        """

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k

        # muliply by reconstruction matrices
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        # prepare output tensor of the right shape
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)

        # save reconstructed points
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o

        return x


class sparseKernelFT1d(nn.Module):
    """loss-pass filter and reweighting of channels for each mode"""

    def __init__(
        self,
        k: int,
        alpha: int,  # num of low-freq Fourier modes
        c: int = 1,
    ):
        super().__init__()

        self.modes1 = alpha
        self.scale = 1 / (c * k * c * k)

        # trainable weights (linear projection for each channel)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat)
        )
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        """Change num of channels. x - mode dim"""
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        """(B, N, c, k) -> (B, N, c, k)"""

        # B: batch, N: timeseries length, c: channels, k: multiwavelet dim
        B, N, c, k = x.shape  # (B, N, c, k)

        # reshape and reshape to(B, F, N), where F - total number of features per timestep
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)  # FFT

        # Multiply relevant Fourier modes (loss-pass filter)
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        # reverse transform get a timeseries of the original length N
        # in float32
        x = torch.fft.irfft(out_ft, n=N)

        # reshape back to the original shape
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class FEAf_lowpass(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_modes: int = 16,
        activation: str = "tanh",
    ):
        super().__init__()

        self.d_model = d_model
        self.num_modes = num_modes
        self.activation = activation

        self.scale = 1 / d_model / d_model

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

        b, t, c, mw_dim = q.shape

        # (b, t, c, k) -> (b, k, c, t)
        xq = q.permute(0, 3, 2, 1)
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)

        # low-pass filter
        index = list(range(0, min(int(t // 2), self.num_modes)))

        # Compute Fourier coefficients for q, k, v
        # -> (b, k, c, num_modes)
        xq_ft_ = self.fft_select_modes(xq, index)
        xk_ft_ = self.fft_select_modes(xk, index)
        xv_ft_ = self.fft_select_modes(xv, index)

        # perform attention mechanism on frequency domain (between modes)
        # -> (b, h, modes q, modes k)
        xqk_ft = torch.einsum("bhdx,bhdy->bhxy", xq_ft_, xk_ft_)

        # apply activation function to attention scores
        xqk_ft = apply_activation(xqk_ft, self.activation)

        # multiply values by attention scores
        xqkv_ft = torch.einsum("bhxy,bhdy->bhdx", xqk_ft, xv_ft_)

        # create empty tensor for iFFT
        out_ft = torch.zeros(
            b, mw_dim, c, t // 2 + 1, device=xq.device, dtype=torch.cfloat
        )
        # copy values in frequencty domain to `out_ft`
        for i, j in enumerate(index):
            out_ft[:, :, :, j] = xqkv_ft[:, :, :, i]

        # Return to time domain
        out = torch.fft.irfft(out_ft * self.scale, n=xq.size(-1))

        # reshape back to original shape (b, k, c, t) -> (b, t, c, k)
        return out.permute(0, 3, 2, 1)
