"""Five-fold cross-validation for sample-level and participant-level analysis.

The default uses stratified folds. ``--grouped`` provides participant-held-out analysis
for a deployment-oriented evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

from .data import CalibratedStandardizer, MMTEDataset, load_npz
from .model import MMTED
from .train import run_epoch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/mmted_cv.json")
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    data = load_npz(args.data)
    labels = np.asarray(data["labels"], dtype=np.int64)
    subject = data.get("subject", np.arange(len(labels)))
    if args.grouped:
        splitter = GroupKFold(n_splits=args.folds)
        folds = splitter.split(np.zeros(len(labels)), labels, groups=subject)
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = splitter.split(np.zeros(len(labels)), labels)
    results = []
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        torch.manual_seed(args.seed + fold)
        eda_scaler = CalibratedStandardizer().fit(data["eda_gsr"][train_idx])
        eye_scaler = CalibratedStandardizer().fit(data["eye"][train_idx])
        train_set = MMTEDataset(data["eda_gsr"], data["eye"], labels, train_idx, eda_scaler, eye_scaler)
        val_set = MMTEDataset(data["eda_gsr"], data["eye"], labels, val_idx, eda_scaler, eye_scaler)
        model = MMTED(input_channels=data["eda_gsr"].shape[-1], eye_features=data["eye"].shape[-1],
                      num_classes=args.classes, embed_dim=args.embed_dim).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        for _ in range(args.epochs):
            run_epoch(model, train_loader, optimizer, args.device)
        metrics = run_epoch(model, val_loader, None, args.device)
        results.append({"fold": fold, **metrics})
        print(json.dumps(results[-1], allow_nan=True))
    summary = {"folds": results, "mean_accuracy": float(np.mean([x["accuracy"] for x in results])),
               "mean_macro_f1": float(np.mean([x["macro_f1"] for x in results]))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
