from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.radiomapseer_point_dataset import create_dataloaders
from models.model import RadioMapSeerRegressor
from training.train import evaluate, save_checkpoint, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RadioMapSeer point-regression model."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "radiomapseer_point_residual_v6_400maps_multichannel",
        help="Processed dataset root containing images/ and metadata/.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "radiomapseer_regressor_gpu",
        help="Directory for best/last checkpoints.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor for plateau LR reduction.")
    parser.add_argument("--lr-patience", type=int, default=2, help="Validation epochs before LR reduction.")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop after this many validation epochs without improvement. Set 0 to disable.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="smoothl1",
        choices=["mse", "smoothl1"],
        help="Regression loss function.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        choices=["label", "residual_label", "raw_label"],
        help="Target column used for training and evaluation.",
    )
    parser.add_argument(
        "--smoothl1-beta",
        type=float,
        default=0.1,
        help="Beta parameter for SmoothL1Loss.",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="gated",
        choices=["gated", "concat"],
        help="Fusion strategy for ablation experiments.",
    )
    parser.add_argument(
        "--disable-cbam",
        action="store_true",
        help="Disable CBAM for ablation experiments.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for train batches, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap for val/test batches, useful for smoke tests.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_metrics(prefix: str, result) -> str:
    return (
        f"{prefix} loss={result.loss:.6f} "
        f"mse={result.mse:.6f} mae={result.mae:.6f} rmse={result.rmse:.6f}"
    )


def safe_path_text(path: Path) -> str:
    return str(path).encode("ascii", errors="ignore").decode("ascii") or "<non-ascii-path>"


def save_history_csv(history: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_mse",
        "train_mae",
        "train_rmse",
        "val_loss",
        "val_mse",
        "val_mae",
        "val_rmse",
        "learning_rate",
    ]
    with save_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_test_metrics_json(test_metrics: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)


def plot_training_curves(history: list[dict], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_rmse = [row["train_rmse"] for row in history]
    val_rmse = [row["val_rmse"] for row in history]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker="o", label="train")
    plt.plot(epochs, val_loss, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_rmse, marker="o", label="train")
    plt.plot(epochs, val_rmse, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("RMSE Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


def load_checkpoint_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    print(f"Using device: {device}")
    print(f"Dataset root: {safe_path_text(args.dataset_root)}")
    print(f"Checkpoint dir: {safe_path_text(args.checkpoint_dir)}")
    print(f"Fusion type: {args.fusion_type}")
    print(f"CBAM enabled: {not args.disable_cbam}")
    print(f"Loss type: {args.loss_type}")
    print(f"Label column: {args.label_column}")

    dataloaders = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_column=args.label_column,
    )

    model = RadioMapSeerRegressor(
        use_cbam=not args.disable_cbam,
        fusion_type=args.fusion_type,
    ).to(device)
    if args.loss_type == "mse":
        criterion = nn.MSELoss()
    elif args.loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    start_epoch = 1

    if args.resume_checkpoint is not None:
        checkpoint = load_checkpoint_state(args.resume_checkpoint, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_rmse = float(checkpoint.get("best_val_rmse", best_val_rmse))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        best_checkpoint_path = args.checkpoint_dir / "best.pt"
        if best_checkpoint_path.exists():
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            best_val_rmse = float(best_checkpoint.get("best_val_rmse", best_val_rmse))
            best_epoch = int(best_checkpoint.get("best_epoch", best_checkpoint.get("epoch", best_epoch)))

        print(
            f"Resumed from epoch {start_epoch - 1}: "
            f"best_epoch={best_epoch}, best_val_rmse={best_val_rmse:.6f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_result = train_one_epoch(
            model=model,
            loader=dataloaders.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=args.max_train_batches,
        )
        val_result = evaluate(
            model=model,
            loader=dataloaders.val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        print(format_metrics("train", train_result))
        print(format_metrics("val  ", val_result))
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_mse": train_result.mse,
                "train_mae": train_result.mae,
                "train_rmse": train_result.rmse,
                "val_loss": val_result.loss,
                "val_mse": val_result.mse,
                "val_mae": val_result.mae,
                "val_rmse": val_result.rmse,
                "learning_rate": current_lr,
            }
        )

        if val_result.rmse < best_val_rmse:
            best_val_rmse = val_result.rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_path=args.checkpoint_dir / "best.pt",
                best_val_rmse=best_val_rmse,
                extra_state={
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "scheduler_state_dict": scheduler.state_dict(),
                },
            )
            print(f"Saved new best checkpoint with val_rmse={best_val_rmse:.6f}")
        else:
            epochs_without_improvement += 1

        scheduler.step(val_result.rmse)
        next_lr = optimizer.param_groups[0]["lr"]
        if next_lr < current_lr:
            print(f"Reduced learning rate: {current_lr:.6g} -> {next_lr:.6g}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            save_path=args.checkpoint_dir / "last.pt",
            best_val_rmse=best_val_rmse,
            extra_state={
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "scheduler_state_dict": scheduler.state_dict(),
            },
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best_val_rmse={best_val_rmse:.6f}"
            )
            break

    print("\nEvaluating best checkpoint on test split...")
    best_checkpoint = torch.load(args.checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_result = evaluate(
        model=model,
        loader=dataloaders.test_loader,
        criterion=criterion,
        device=device,
        max_batches=args.max_val_batches,
    )
    print(format_metrics("test ", test_result))

    save_history_csv(history, args.checkpoint_dir / "metrics_history.csv")
    save_test_metrics_json(
        {
            "test_loss": test_result.loss,
            "test_mse": test_result.mse,
            "test_mae": test_result.mae,
            "test_rmse": test_result.rmse,
            "best_val_rmse": best_val_rmse,
            "best_epoch": best_epoch,
            "epochs_requested": args.epochs,
            "epochs_trained": len(history),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "min_lr": args.min_lr,
            "early_stop_patience": args.early_stop_patience,
            "loss_type": args.loss_type,
            "label_column": args.label_column,
            "smoothl1_beta": args.smoothl1_beta,
            "fusion_type": args.fusion_type,
            "cbam_enabled": not args.disable_cbam,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        },
        args.checkpoint_dir / "test_metrics.json",
    )
    if history:
        plot_training_curves(history, args.checkpoint_dir)

    print("Training finished.")


if __name__ == "__main__":
    main()
