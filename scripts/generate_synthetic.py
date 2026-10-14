"""Create a tiny deterministic print-monitoring dataset for smoke tests and demos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="data/synthetic"); p.add_argument("--n", type=int, default=40); p.add_argument("--size", type=int, default=128); args = p.parse_args()
    root = Path(args.out); image_dir = root / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        defect = i % 2
        split = "test" if i % 5 == 0 else ("val" if i % 5 == 1 else "train")
        rng = np.random.default_rng(i); arr = np.full((args.size, args.size, 3), 35, dtype=np.float32)
        arr = np.clip(arr + rng.normal(0, 8, arr.shape).clip(-35, 35), 0, 255).astype(np.uint8)
        image = Image.fromarray(arr); draw = ImageDraw.Draw(image)
        for y in range(8, args.size, 16): draw.line((0, y, args.size, y), fill=(180, 180, 180), width=1)
        if defect: draw.line((15, args.size // 2, args.size - 15, args.size // 2 + 15), fill=(255, 30, 30), width=3)
        name = f"frame_{i:05d}.png"; image.save(image_dir / name)
        rows.append({"image": name, "label": "defect" if defect else "normal", "split": split, "sequence_id": i // 4, "nozzle_temperature": 205 + defect * 12, "bed_temperature": 60, "extrusion_rate": 0.8 + defect * 0.2, "layer_height": 0.2})
    with (root / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} frames to {root}")


if __name__ == "__main__": main()
