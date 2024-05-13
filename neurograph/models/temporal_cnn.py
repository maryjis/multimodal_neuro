from typing import Any, Optional
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from neurograph.config.config import TemporalCNNConfig, TemporalConvConfig
from neurograph.models.available_modules import available_activations


class TemporalConv(nn.Module):
    def __init__(
        self,
        # conv params
        in_channels: int,
        cfg: TemporalConvConfig,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            groups=cfg.groups,
            dilation=cfg.dilation,
            padding=cfg.padding,
            padding_mode=cfg.padding_mode,
            bias=cfg.bias,
        )
        act_params = {} if cfg.act_params is None else cfg.act_params
        self.act = available_activations[cfg.act_func](**act_params)

        self.bn = nn.BatchNorm1d(cfg.out_channels)

        if cfg.pooling_type == "mean":
            self.pool = nn.AvgPool1d(cfg.pooling_size)
        elif cfg.pooling_type == "max":
            self.pool = nn.MaxPool1d(cfg.pooling_size)
        else:
            raise ValueError(
                f"Unknown pooling type: {cfg.pooling_type}. Use mean, max"
            )  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv along T -> BatchNorm -> act func -> pooling along T"""
        out = x
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)

        return out


class TemporalCNN(nn.Module):
    def __init__(
        self,
        # comes from dataset
        input_dim: int,  # for compatibility
        num_nodes: int,  # num ROIs
        model_cfg: TemporalCNNConfig,
    ):
        super().__init__()

        self.conv_1 = TemporalConv(num_nodes, model_cfg.layers[0])
        self.conv_2 = TemporalConv(
            model_cfg.layers[0].out_channels,
            model_cfg.layers[1],
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveAvgPool1d(1)

        self.readout = model_cfg.pooling_readout
        self.dropout = nn.Dropout(model_cfg.dropout_rate)

        dim = 2 if self.readout == "meanmax" else 1
        self.lin = nn.Linear(
            model_cfg.layers[1].out_channels * dim,
            model_cfg.n_classes,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch[0]
        out = x

        out = self.conv_1(out)
        out = self.conv_2(out)

        # (b, c, t) -> (b, c, 1) -> (b, c)
        h_avg = torch.squeeze(self.avg_pool(out))
        h_max = torch.squeeze(self.max_pool(out))

        # combine pooling results
        if self.readout == "meanmax":
            out = torch.cat((h_avg, h_max), 1)
        elif self.readout == "mean":
            out = h_avg
        elif self.readout == "max":
            out = h_max

        out = self.dropout(out)
        out = self.lin(out)

        return out
