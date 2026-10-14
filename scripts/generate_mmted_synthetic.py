"""Generate deterministic EDA/GSR + eye windows for a complete MMTED smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="data/mmted_synthetic.npz"); p.add_argument("--n", type=int, default=96); p.add_argument("--classes", type=int, default=4); p.add_argument("--samples", type=int, default=128); p.add_argument("--eye-features", type=int, default=6); args = p.parse_args()
    rng = np.random.default_rng(42); labels = np.arange(args.n) % args.classes; t = np.linspace(0, 1, args.samples, endpoint=False)
    eda = rng.normal(0, .12, (args.n, args.samples, 2)).astype(np.float32); eye = rng.normal(0, .2, (args.n, args.samples, args.eye_features)).astype(np.float32)
    for i, label in enumerate(labels):
        eda[i, :, 0] += .35 * label * np.sin(2 * np.pi * (label + 1) * t); eda[i, :, 1] += .18 * label
        eye[i, :, 0] += .25 * label * np.cos(2 * np.pi * t); eye[i, :, 1] += .12 * label
    subject = np.arange(args.n) // 8; train_mask = (subject % 5) < 3
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.out, eda_gsr=eda, eye=eye, labels=labels, subject=subject, train_mask=train_mask)
    print(f"wrote {args.n} windows ({args.classes} classes) to {args.out}")


if __name__ == "__main__": main()
