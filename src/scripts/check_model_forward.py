from __future__ import annotations

import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.radiomapseer_point_dataset import create_dataloaders
from models.model import RadioMapSeerRegressor


def main() -> None:
    dataset_root = CODE_ROOT.parent / "data" / "processed" / "radiomapseer_point_residual_v6_400maps_multichannel"

    bundle = create_dataloaders(dataset_root=dataset_root, batch_size=8, num_workers=0)
    batch = next(iter(bundle.train_loader))

    model = RadioMapSeerRegressor()
    model.eval()

    with torch.no_grad():
        out = model(batch["image"], batch["numeric"])

    print("input image:", tuple(batch["image"].shape))
    print("input numeric:", tuple(batch["numeric"].shape))
    print("output:", tuple(out.shape))
    print("output sample:", out[:3].tolist())


if __name__ == "__main__":
    main()
