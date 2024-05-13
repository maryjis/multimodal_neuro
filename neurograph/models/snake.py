import torch
from torch import nn


class Snake(nn.Module):
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = a
        assert self.a != 0.0, "Parameter `a` in Snake must be non-zero!"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(self.a * x) * torch.sin(self.a * x) / self.a

    def __repr__(self):  # pragma: no cover
        return f"Snake(a={self.a})"
