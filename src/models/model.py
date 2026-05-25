from __future__ import annotations

import torch
from torch import nn

from models.backbone import ConvNeXtBackbone
from models.cbam import CBAM
from models.fusion import ConcatFusion, GatedFusion
from models.mlp_branch import NumericMLPBranch
from models.regression_head import RegressionHead


class RadioMapSeerRegressor(nn.Module):
    """
    Agreed current model:
    ConvNeXt-style CNN backbone + CBAM + MLP branch + Gated Fusion + Regression Head
    """

    def __init__(self, use_cbam: bool = True, fusion_type: str = "gated") -> None:
        super().__init__()
        self.backbone = ConvNeXtBackbone(in_channels=4, stem_channels=32, out_channels=64)
        self.cbam = CBAM(channels=64, reduction=8, kernel_size=7) if use_cbam else nn.Identity()
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.numeric_branch = NumericMLPBranch(input_dim=4, hidden_dim=16, output_dim=16)

        if fusion_type == "gated":
            self.fusion = GatedFusion(image_dim=64, numeric_dim=16, fused_dim=32)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(image_dim=64, numeric_dim=16, fused_dim=32)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.head = RegressionHead(input_dim=32, hidden_dim=16, dropout=0.3)
        self.use_cbam = use_cbam
        self.fusion_type = fusion_type

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(image)
        x = self.cbam(x)
        x = self.image_pool(x).flatten(1)
        return x

    def encode_numeric(self, numeric: torch.Tensor) -> torch.Tensor:
        return self.numeric_branch(numeric)

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        image_feat = self.encode_image(image)
        numeric_feat = self.encode_numeric(numeric)
        fused_feat = self.fusion(image_feat, numeric_feat)
        output = self.head(fused_feat)
        return output.squeeze(1)
