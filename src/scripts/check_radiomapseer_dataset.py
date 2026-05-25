from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.radiomapseer_point_dataset import create_dataloaders


def main() -> None:
    dataset_root = CODE_ROOT.parent / "data" / "processed" / "radiomapseer_point_residual_v6_400maps_multichannel"
    bundle = create_dataloaders(dataset_root=dataset_root, batch_size=8, num_workers=0)

    batch = next(iter(bundle.train_loader))
    print("image shape:", tuple(batch["image"].shape))
    print("numeric shape:", tuple(batch["numeric"].shape))
    print("label shape:", tuple(batch["label"].shape))
    print("first image file:", batch["image_file"][0])
    print("first numeric:", batch["numeric"][0].tolist())
    print("first label:", float(batch["label"][0]))


if __name__ == "__main__":
    main()
