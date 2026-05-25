from __future__ import annotations

import torch
from torch import nn


class NumericMLPBranch(nn.Module):
    """4-D numeric input -> 16-D feature."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
