"""
Prepare a point-regression dataset from RadioMapSeer for the current model:

    scene patch (64x64x4) + 4-D coordinate features -> 1 scalar residual label

Design choices:
- Scene image is a four-channel semantic patch:
  buildings, roads, cars, and the selected antenna mask.
- Numeric features are normalized [0, 1]:
  [tx_x, tx_y, rx_x, rx_y]
- LOS/building-crossing values are written to metadata for analysis only.
- A simple distance-only baseline is fitted on the training split:
  y_base = a * log(1 + distance_pixels) + b
- Final label is the residual:
  y_residual = y_true - y_base

The script intentionally creates a manageable training subset for course-project use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class SampleRecord:
    map_id: int
    antenna_id: int
    tx_x: int
    tx_y: int
    rx_x: int
    rx_y: int
    image_file: str
    raw_label: float
    split: str
    los_blocked: float
    building_cross_count_norm: float
    building_cross_ratio: float


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def save_rgba(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGBA").save(path)


def compose_scene(buildings: np.ndarray, roads: np.ndarray, cars: np.ndarray, antenna: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            buildings,
            roads,
            cars,
            antenna,
        ],
        axis=-1,
    ).astype(np.uint8)


def crop_patch(scene: np.ndarray, center_x: int, center_y: int, patch_size: int = 64) -> np.ndarray:
    half = patch_size // 2
    if scene.ndim == 2:
        pad_width = ((half, half), (half, half))
    else:
        pad_width = ((half, half), (half, half), (0, 0))
    padded = np.pad(scene, pad_width, mode="constant", constant_values=0)
    px = center_x + half
    py = center_y + half
    patch = padded[py - half : py + half, px - half : px + half]
    return patch.astype(np.uint8)


def normalized_features(tx_x: int, tx_y: int, rx_x: int, rx_y: int, size: int = 256) -> np.ndarray:
    denom = float(size - 1)
    return np.array([tx_x / denom, tx_y / denom, rx_x / denom, rx_y / denom], dtype=np.float32)


def receiver_distance(tx_x: int, tx_y: int, rx_x: int, rx_y: int) -> float:
    return math.sqrt((tx_x - rx_x) ** 2 + (tx_y - ry_y) ** 2)


def line_pixels(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points


def building_los_features(buildings: np.ndarray, tx_x: int, tx_y: int, rx_x: int, rx_y: int) -> tuple[float, float, float]:
    points = line_pixels(tx_x, tx_y, rx_x, rx_y)
    h, w = buildings.shape
    valid_points = [(x, y) for x, y in points if 0 <= x < w and 0 <= y < h]
    if not valid_points:
        return 0.0, 0.0, 0.0

    building_count = sum(1 for x, y in valid_points if buildings[y, x] > 0)
    line_length = len(valid_points)
    los_blocked = 1.0 if building_count > 0 else 0.0
    building_cross_count_norm = building_count / 256.0
    building_cross_ratio = building_count / line_length
    return los_blocked, building_cross_count_norm, building_cross_ratio


def choose_receiver_points(
    gain_map: np.ndarray,
    count: int,
    min_gain: int = 5,
    seed: int | None = None,
) -> List[Tuple[int, int]]:
    ys, xs = np.where(gain_map >= min_gain)
    coords = list(zip(xs.tolist(), ys.tolist()))
    if not coords:
        ys, xs = np.where(gain_map > 0)
        coords = list(zip(xs.tolist(), ys.tolist()))
    if not coords:
        return []
    rng = random.Random(seed)
    if len(coords) <= count:
        rng.shuffle(coords)
        return coords
    return rng.sample(coords, count)


def dataset_rows(dataset_csv: Path) -> List[dict]:
    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dirs(base: Path) -> None:
    for name in ["images", "metadata"]:
        (base / name).mkdir(parents=True, exist_ok=True)


def split_name(index: int, total: int) -> str:
    r = index / max(total, 1)
    if r < 0.7:
        return "train"
    if r < 0.85:
        return "val"
    return "test"


def fit_distance_baseline(records: List[SampleRecord]) -> tuple[float, float]:
    train_records = [rec for rec in records if rec.split == "train"]
    if not train_records:
        raise ValueError("Cannot fit baseline without training samples.")

    distances = np.array(
        [math.log1p(receiver_distance(rec.tx_x, rec.tx_y, rec.rx_x, rec.rx_y)) for rec in train_records],
        dtype=np.float64,
    )
    labels = np.array([rec.raw_label for rec in train_records], dtype=np.float64)

    X = np.stack([distances, np.ones_like(distances)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X, labels, rcond=None)
    a, b = coeffs.tolist()
    return float(a), float(b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomapseer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-maps", type=int, default=200)
    parser.add_argument("--antennas-per-map", type=int, default=8)
    parser.add_argument("--points-per-antenna", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = args.radiomapseer_root
    out = args.output_root
    ensure_dirs(out)

    rows = dataset_rows(root / "dataset.csv")
    rows = rows[: min(args.max_maps, len(rows))]
    rng.shuffle(rows)
    total_maps = len(rows)

    all_records: List[SampleRecord] = []
    image_counter = 0

    for map_idx, row in enumerate(rows):
        map_file = row["maps"]
        json_file = row["json"]

        buildings = load_gray(root / "png" / "buildings_complete" / map_file)
        roads = load_gray(root / "png" / "roads" / map_file)
        cars = load_gray(root / "png" / "cars" / map_file)

        with (root / "antenna" / json_file).open("r", encoding="utf-8") as f:
            antenna_positions = json.load(f)

        antenna_ids = list(range(min(args.antennas_per_map, len(antenna_positions))))
        rng.shuffle(antenna_ids)

        for antenna_id in antenna_ids:
            tx_x, tx_y = antenna_positions[antenna_id]
            antenna_png = f"{Path(map_file).stem}_{antenna_id}.png"
            gain_png = row[f"Gain{antenna_id + 1}"]

            antenna_mask = load_gray(root / "png" / "antennas" / antenna_png)
            gain_map = load_gray(root / "gain" / "DPM" / gain_png)
            scene = compose_scene(buildings, roads, cars, antenna_mask)

            points = choose_receiver_points(
                gain_map=gain_map,
                count=args.points_per_antenna,
                min_gain=5,
                seed=args.seed + map_idx * 1000 + antenna_id,
            )

            map_split = split_name(map_idx, total_maps)
            for rx_x, rx_y in points:
                patch = crop_patch(scene, rx_x, rx_y, patch_size=args.patch_size)
                img_name = f"sample_{image_counter:07d}.png"
                img_path = out / "images" / img_name
                save_rgba(img_path, patch)

                raw_label = float(gain_map[rx_y, rx_x]) / 255.0
                los_blocked, building_cross_count_norm, building_cross_ratio = building_los_features(
                    buildings,
                    tx_x,
                    tx_y,
                    rx_x,
                    rx_y,
                )

                rec = SampleRecord(
                    map_id=map_idx,
                    antenna_id=antenna_id,
                    tx_x=tx_x,
                    tx_y=tx_y,
                    rx_x=rx_x,
                    rx_y=rx_y,
                    image_file=img_name,
                    raw_label=raw_label,
                    split=map_split,
                    los_blocked=los_blocked,
                    building_cross_count_norm=building_cross_count_norm,
                    building_cross_ratio=building_cross_ratio,
                )
                all_records.append(rec)
                image_counter += 1

    baseline_a, baseline_b = fit_distance_baseline(all_records)

    metadata_path = out / "metadata" / "samples.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "map_id",
                "antenna_id",
                "image_file",
                "tx_x_norm",
                "tx_y_norm",
                "rx_x_norm",
                "rx_y_norm",
                "distance_pixels",
                "distance_norm",
                "los_blocked",
                "building_cross_count_norm",
                "building_cross_ratio",
                "raw_label",
                "baseline_label",
                "residual_label",
                "label",
            ]
        )
        for rec in all_records:
            feats = normalized_features(rec.tx_x, rec.tx_y, rec.rx_x, rec.rx_y)
            distance_pixels = receiver_distance(rec.tx_x, rec.tx_y, rec.rx_x, rec.rx_y)
            distance_norm = distance_pixels / math.sqrt(2 * (255 ** 2))
            baseline_label = baseline_a * math.log1p(distance_pixels) + baseline_b
            residual_label = rec.raw_label - baseline_label
            writer.writerow(
                [
                    rec.split,
                    rec.map_id,
                    rec.antenna_id,
                    rec.image_file,
                    f"{feats[0]:.6f}",
                    f"{feats[1]:.6f}",
                    f"{feats[2]:.6f}",
                    f"{feats[3]:.6f}",
                    f"{distance_pixels:.6f}",
                    f"{distance_norm:.6f}",
                    f"{rec.los_blocked:.6f}",
                    f"{rec.building_cross_count_norm:.6f}",
                    f"{rec.building_cross_ratio:.6f}",
                    f"{rec.raw_label:.6f}",
                    f"{baseline_label:.6f}",
                    f"{residual_label:.6f}",
                    f"{residual_label:.6f}",
                ]
            )

    summary_path = out / "metadata" / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        train_count = sum(1 for r in all_records if r.split == "train")
        val_count = sum(1 for r in all_records if r.split == "val")
        test_count = sum(1 for r in all_records if r.split == "test")
        f.write("RadioMapSeer point-regression residual subset\n")
        f.write(f"total_samples={len(all_records)}\n")
        f.write(f"maps_used={total_maps}\n")
        f.write(f"train={train_count}\n")
        f.write(f"val={val_count}\n")
        f.write(f"test={test_count}\n")
        f.write(f"patch_size={args.patch_size}\n")
        f.write(f"max_maps={args.max_maps}\n")
        f.write(f"antennas_per_map={args.antennas_per_map}\n")
        f.write(f"points_per_antenna={args.points_per_antenna}\n")
        f.write("split_strategy=by_map\n")
        f.write("image_channels=4\n")
        f.write("image_channel_order=buildings,roads,cars,antenna\n")
        f.write("numeric_features=tx_x_norm,tx_y_norm,rx_x_norm,rx_y_norm\n")
        f.write("metadata_only_features=distance_norm,los_blocked,building_cross_count_norm,building_cross_ratio\n")
        f.write(f"baseline_a={baseline_a:.8f}\n")
        f.write(f"baseline_b={baseline_b:.8f}\n")
        f.write("target=residual_label\n")

    print(f"Done. Wrote {len(all_records)} samples.")
    print("Metadata file created.")


if __name__ == "__main__":
    main()
