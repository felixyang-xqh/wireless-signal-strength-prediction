from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style block for course-project experiments."""

    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        return x + residual


class ConvNeXtBackbone(nn.Module):
    """
    Backbone that matches the agreed structure:
    64x64x4 -> 16x16x32 -> 16x16x32 -> 8x8x64 -> 8x8x64 -> GAP -> 64-D.
    """

    def __init__(self, in_channels: int = 1, stem_channels: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=4, stride=4),
            LayerNorm2d(stem_channels),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(stem_channels),
            ConvNeXtBlock(stem_channels),
        )
        self.downsample = nn.Sequential(
            LayerNorm2d(stem_channels),
            nn.Conv2d(stem_channels, out_channels, kernel_size=2, stride=2),
        )
        self.stage2 = ConvNeXtBlock(out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return x
