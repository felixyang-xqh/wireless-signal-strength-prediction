from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.radiomapseer_point_dataset import create_dataloaders
from models.model import RadioMapSeerRegressor
from training.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RadioMapSeer checkpoint.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        choices=["label", "residual_label", "raw_label"],
        help="Target column used for evaluation.",
    )
    parser.add_argument("--fusion-type", type=str, default="gated", choices=["gated", "concat"])
    parser.add_argument("--disable-cbam", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result = evaluate(
        model=model,
        loader=dataloaders.test_loader,
        criterion=nn.MSELoss(),
        device=device,
    )

    metrics = {
        "test_loss": result.loss,
        "test_mse": result.mse,
        "test_mae": result.mae,
        "test_rmse": result.rmse,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_rmse": checkpoint.get("best_val_rmse"),
        "batch_size": args.batch_size,
        "label_column": args.label_column,
        "fusion_type": args.fusion_type,
        "cbam_enabled": not args.disable_cbam,
    }

    print(
        f"test loss={result.loss:.6f} "
        f"mse={result.mse:.6f} mae={result.mae:.6f} rmse={result.rmse:.6f}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
