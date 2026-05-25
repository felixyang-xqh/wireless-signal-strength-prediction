from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import regression_metrics


@dataclass
class EpochResult:
    loss: float
    mse: float
    mae: float
    rmse: float


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "numeric": batch["numeric"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochResult:
    model.train()

    total_loss = 0.0
    all_pred = []
    all_target = []
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False, disable=True), start=1):
        batch = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["image"], batch["numeric"])
        loss = criterion(pred, batch["label"])
        loss.backward()
        optimizer.step()

        batch_size = batch["label"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_pred.append(pred.detach().cpu())
        all_target.append(batch["label"].detach().cpu())

        if max_batches is not None and batch_idx >= max_batches:
            break

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    metrics = regression_metrics(all_pred, all_target)
    mean_loss = total_loss / total_samples

    return EpochResult(
        loss=mean_loss,
        mse=metrics["mse"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochResult:
    model.eval()

    total_loss = 0.0
    all_pred = []
    all_target = []
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="eval", leave=False, disable=True), start=1):
        batch = _move_batch(batch, device)

        pred = model(batch["image"], batch["numeric"])
        loss = criterion(pred, batch["label"])

        batch_size = batch["label"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_pred.append(pred.detach().cpu())
        all_target.append(batch["label"].detach().cpu())

        if max_batches is not None and batch_idx >= max_batches:
            break

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    metrics = regression_metrics(all_pred, all_target)
    mean_loss = total_loss / total_samples

    return EpochResult(
        loss=mean_loss,
        mse=metrics["mse"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    save_path: str | Path,
    best_val_rmse: float,
    extra_state: dict[str, Any] | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_rmse": best_val_rmse,
    }
    if extra_state:
        checkpoint.update(extra_state)

    torch.save(checkpoint, save_path)
