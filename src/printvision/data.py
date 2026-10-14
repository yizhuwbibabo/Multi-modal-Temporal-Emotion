"""Datasets and deterministic transforms for print-process image experiments.

The CSV format intentionally keeps the experimental unit explicit. Required columns are
``image`` and ``label``; optional ``split``, ``sequence_id`` and telemetry columns permit
sequence-aware splits and image + machine-state fusion without changing the loader.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

try:  # Keep import errors local so data inspection works before installing torch.
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and torch (when installed) for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _to_tensor(image: Image.Image, size: int):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    array = np.transpose(array, (2, 0, 1))
    if torch is None:
        return array
    return torch.from_numpy(array).float()


class PrintFrameDataset(Dataset):
    """Load labelled print frames and optional synchronized telemetry.

    Labels can be strings or integer IDs. For a sequence-safe evaluation, generate the CSV
    with a ``split`` column and set ``split`` to ``train``, ``val`` or ``test``. If absent,
    callers should create a split by sequence before training.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path = ".",
        image_size: int = 224,
        split: str | None = None,
        telemetry_columns: Sequence[str] = (),
        label_to_id: dict[str, int] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No rows found in {self.csv_path}")
        if "image" not in rows[0] or "label" not in rows[0]:
            raise ValueError("CSV must contain image and label columns")
        if split is not None and "split" in rows[0]:
            rows = [row for row in rows if row.get("split") == split]
        if not rows:
            raise ValueError(f"No rows for split={split!r} in {self.csv_path}")
        self.rows = rows
        self.image_size = image_size
        self.telemetry_columns = list(telemetry_columns)
        labels = sorted({row["label"] for row in rows})
        self.label_to_id = label_to_id or {name: i for i, name in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.image_root / row["image"]
        image = _to_tensor(Image.open(path), self.image_size)
        label = self.label_to_id[row["label"]]
        telemetry = np.asarray(
            [float(row.get(name, 0.0) or 0.0) for name in self.telemetry_columns], dtype=np.float32
        )
        if torch is None:
            return {"image": image, "telemetry": telemetry, "label": label, "meta": row}
        return {
            "image": image,
            "telemetry": torch.from_numpy(telemetry),
            "label": torch.tensor(label, dtype=torch.long),
            "meta": row,
        }


def write_manifest(rows: Iterable[dict], path: str | Path) -> None:
    """Write rows with stable column ordering, useful after sequence-level splitting."""

    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
