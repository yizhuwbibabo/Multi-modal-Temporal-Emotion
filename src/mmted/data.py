"""NPZ data contract and calibration/standardisation for MMTED.

An NPZ contains ``eda_gsr`` (N,T,2), ``eye`` (N,T,F), ``labels`` (N,), and optional
``subject`` (N,). Samples are one-second windows; when raw traces are longer, use
``window_array`` to create non-overlapping windows before constructing the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    class Dataset:  # type: ignore[no-redef]
        pass


def window_array(array: np.ndarray, window_size: int, stride: int | None = None) -> np.ndarray:
    """Convert ``(trials, samples, channels)`` traces into fixed windows."""

    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError("expected an array with shape (trials, samples, channels)")
    stride = stride or window_size
    chunks = [array[:, start:start + window_size] for start in range(0, array.shape[1] - window_size + 1, stride)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, window_size, array.shape[2]), array.dtype)


@dataclass
class CalibratedStandardizer:
    """Baseline calibration followed by train-only per-channel z-score scaling."""

    calibration_samples: int = 0
    baseline_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, calibration: np.ndarray | None = None) -> CalibratedStandardizer:
        x = np.asarray(x, dtype=np.float32)
        if calibration is None and self.calibration_samples:
            calibration = x[:, :self.calibration_samples]
        self.baseline_ = np.median(calibration, axis=(0, 1)) if calibration is not None else np.zeros(x.shape[-1], dtype=np.float32)
        calibrated = x - self.baseline_
        self.mean_ = calibrated.mean(axis=(0, 1)); self.scale_ = calibrated.std(axis=(0, 1)).clip(min=1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.baseline_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the standardizer on training data first")
        return ((np.asarray(x, dtype=np.float32) - self.baseline_) - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray, calibration: np.ndarray | None = None) -> np.ndarray:
        return self.fit(x, calibration).transform(x)

    def state_dict(self) -> dict[str, list[float]]:
        if self.baseline_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the standardizer before serializing it")
        return {"baseline": self.baseline_.tolist(), "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> CalibratedStandardizer:
        return cls(
            baseline_=np.asarray(state["baseline"], dtype=np.float32),
            mean_=np.asarray(state["mean"], dtype=np.float32),
            scale_=np.asarray(state["scale"], dtype=np.float32),
        )

    @classmethod
    def identity(cls, channels: int) -> CalibratedStandardizer:
        """Return a no-op transform for an NPZ already produced by ``preprocess``."""
        zeros = np.zeros(channels, dtype=np.float32)
        return cls(baseline_=zeros, mean_=zeros.copy(), scale_=np.ones(channels, dtype=np.float32))


def subject_split(subject: np.ndarray, train_fraction: float = 0.8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint sample indices split by participant.

    Stratified sample splitting is available separately; this grouped split is the safer
    default for a new participant and helps detect identity leakage in a project.
    """
    subject = np.asarray(subject)
    unique = np.unique(subject)
    if len(unique) < 2:
        cut = max(1, int(len(subject) * train_fraction))
        return np.arange(cut), np.arange(cut, len(subject))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    cut = min(len(unique) - 1, max(1, round(len(unique) * train_fraction)))
    train_subjects, val_subjects = set(shuffled[:cut]), set(shuffled[cut:])
    train_idx = np.flatnonzero(np.isin(subject, list(train_subjects)))
    val_idx = np.flatnonzero(np.isin(subject, list(val_subjects)))
    return train_idx, val_idx


class MMTEDataset(Dataset):
    """Torch dataset returning physiological, eye and class tensors."""

    def __init__(self, eda_gsr, eye, labels, indices=None, eda_standardizer=None, eye_standardizer=None):
        self.eda_gsr = np.asarray(eda_gsr, dtype=np.float32)
        self.eye = np.asarray(eye, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        if len(self.eda_gsr) != len(self.eye) or len(self.eye) != len(self.labels): raise ValueError("modalities must have the same number of samples")
        self.eda_standardizer, self.eye_standardizer = eda_standardizer, eye_standardizer

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]; eda = self.eda_gsr[i]; eye = self.eye[i]
        if self.eda_standardizer: eda = self.eda_standardizer.transform(eda[None])[0]
        if self.eye_standardizer: eye = self.eye_standardizer.transform(eye[None])[0]
        if torch is None: return {"eda_gsr": eda, "eye": eye, "label": int(self.labels[i])}
        return {"eda_gsr": torch.from_numpy(eda.T), "eye": torch.from_numpy(eye), "label": torch.tensor(self.labels[i], dtype=torch.long)}


def load_npz(path: str | Path):
    data = np.load(path, allow_pickle=False)
    required = {"eda_gsr", "eye", "labels"}
    missing = required.difference(data.files)
    if missing: raise ValueError(f"missing NPZ keys: {sorted(missing)}")
    return {key: data[key] for key in data.files}
