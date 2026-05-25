from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """32-D fused feature -> 1 scalar output."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
