from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """
    64-D image feature + 16-D numeric feature -> 32-D fused feature.
    """

    def __init__(self, image_dim: int = 64, numeric_dim: int = 16, fused_dim: int = 32) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.numeric_proj = nn.Linear(numeric_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, image_feat: torch.Tensor, numeric_feat: torch.Tensor) -> torch.Tensor:
        image_proj = self.image_proj(image_feat)
        numeric_proj = self.numeric_proj(numeric_feat)
        gate = self.gate(torch.cat([image_proj, numeric_proj], dim=1))
        return gate * image_proj + (1.0 - gate) * numeric_proj


class ConcatFusion(nn.Module):
    """
    Ablation fusion:
    concatenate projected features, then compress to fused_dim.
    """

    def __init__(self, image_dim: int = 64, numeric_dim: int = 16, fused_dim: int = 32) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.numeric_proj = nn.Linear(numeric_dim, fused_dim)
        self.out_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, image_feat: torch.Tensor, numeric_feat: torch.Tensor) -> torch.Tensor:
        image_proj = self.image_proj(image_feat)
        numeric_proj = self.numeric_proj(numeric_feat)
        return self.out_proj(torch.cat([image_proj, numeric_proj], dim=1))
