from __future__ import annotations

import math
from typing import Dict

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().view(-1)
    target = target.detach().float().view(-1)

    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = math.sqrt(mse)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
