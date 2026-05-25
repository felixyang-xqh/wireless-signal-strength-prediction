from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class RadioMapSeerPointDataset(Dataset):
    """
    Point-regression dataset:

        scene patch (64x64x4) + 4 numeric features -> 1 scalar label
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        label_column: str = "label",
        image_transform: Optional[Callable] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.label_column = label_column
        self.image_transform = image_transform

        metadata_path = self.dataset_root / "metadata" / "samples.csv"
        df = pd.read_csv(metadata_path)
        if label_column not in df.columns:
            raise ValueError(f"Missing label column '{label_column}' in {metadata_path}")

        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No samples found for split='{split}' in {metadata_path}")

        self.df = df
        self.images_dir = self.dataset_root / "images"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = self.images_dir / row["image_file"]
        image = Image.open(image_path).convert("RGBA")

        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            # Convert to float tensor in [0,1], shape: [4, H, W].
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        numeric = torch.tensor(
            [
                row["tx_x_norm"],
                row["tx_y_norm"],
                row["rx_x_norm"],
                row["rx_y_norm"],
            ],
            dtype=torch.float32,
        )

        label = torch.tensor(row[self.label_column], dtype=torch.float32)

        return {
            "image": image_tensor,
            "numeric": numeric,
            "label": label,
            "image_file": row["image_file"],
            "map_id": int(row["map_id"]),
            "antenna_id": int(row["antenna_id"]),
        }


def create_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    label_column: str = "label",
    image_transform: Optional[Callable] = None,
) -> DataBundle:
    train_set = RadioMapSeerPointDataset(
        dataset_root,
        split="train",
        label_column=label_column,
        image_transform=image_transform,
    )
    val_set = RadioMapSeerPointDataset(
        dataset_root,
        split="val",
        label_column=label_column,
        image_transform=image_transform,
    )
    test_set = RadioMapSeerPointDataset(
        dataset_root,
        split="test",
        label_column=label_column,
        image_transform=image_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
