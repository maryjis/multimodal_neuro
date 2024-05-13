import math

import torch
from torch import nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series w/ stride=1
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: shape (b, t, r)"""

        # padding on the both ends of time series
        # in total we add (kernel_size - 1) elements to each timeseries
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)

        x = torch.cat([front, x, end], dim=1)

        # apply 1D average pooling (permute since pooling expects (N, C, L)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class series_decomp(nn.Module):
    """
    Series decomposition block into season and trend components.
    Trend is just a moving average.
    Season = inital timeseries - trend
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        season = x - trend  # subtract moving average

        return season, trend


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: list[int]):
        # kernel_size: a list of kernel sizes
        super(series_decomp_multi, self).__init__()

        # create N moving averages modules
        self.moving_avg = [moving_avg(kernel) for kernel in kernel_size]

        # create linear layer to project each feature value to a vector
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        # apply each moving average, append to list
        for func in self.moving_avg:
            moving_avg = func(x)
            # add a new dim
            moving_mean.append(moving_avg.unsqueeze(-1))

        # concat all mv averages per new dimension
        moving_mean = torch.cat(moving_mean, dim=-1)

        # project each feature value to a vector of size = len(kernel_size)
        # (num of all mv averages)
        # Apply Softmax along new dimension
        # find weighted average of different moving averages
        trend = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),
            dim=-1,
        )
        # find residual between x and moving average
        season = x - trend

        return season, trend
