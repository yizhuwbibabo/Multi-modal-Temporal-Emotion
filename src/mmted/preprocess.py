"""Preprocessing helpers and a CLI for calibration + standardisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CalibratedStandardizer, load_npz


def preprocess_npz(input_path, output_path, calibration_samples=0):
    data = load_npz(input_path)
    labels = np.asarray(data["labels"])
    if labels.dtype.kind not in "iu":
        names, labels = np.unique(labels.astype(str), return_inverse=True)
    else:
        names = np.unique(labels).astype(str)
    # Fit only on training subjects (the caller may mark them in the NPZ).
    train_mask = data.get("train_mask", np.ones(len(labels), dtype=bool)).astype(bool)
    eda = CalibratedStandardizer(calibration_samples).fit(data["eda_gsr"][train_mask])
    eye = CalibratedStandardizer(calibration_samples).fit(data["eye"][train_mask])
    out = {**data, "eda_gsr": eda.transform(data["eda_gsr"]), "eye": eye.transform(data["eye"]), "labels": labels.astype(np.int64)}
    out.pop("train_mask", None)
    np.savez_compressed(output_path, **out)
    state = {"labels": names.tolist(), "eda_baseline": eda.baseline_.tolist(), "eda_mean": eda.mean_.tolist(), "eda_scale": eda.scale_.tolist(), "eye_baseline": eye.baseline_.tolist(), "eye_mean": eye.mean_.tolist(), "eye_scale": eye.scale_.tolist()}
    Path(str(output_path) + ".json").write_text(json.dumps(state, indent=2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--calibration-samples", type=int, default=0); args = p.parse_args(argv)
    out = preprocess_npz(args.input, args.output, args.calibration_samples); print(f"wrote {len(out['labels'])} windows to {args.output}")


if __name__ == "__main__": main()
