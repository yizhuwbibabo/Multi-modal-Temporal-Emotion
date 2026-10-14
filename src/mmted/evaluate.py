"""Evaluate an MMTED checkpoint on its held-out participant windows."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .data import CalibratedStandardizer, MMTEDataset, load_npz
from .model import MMTED


def _restore(state):
    return CalibratedStandardizer.from_state_dict(state)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    data = load_npz(args.data)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    subject = data.get("subject", np.arange(len(data["labels"])))
    val_subjects = set(checkpoint.get("val_subjects", []))
    if not val_subjects:
        unique = np.unique(subject)
        val_subjects = set(unique[max(1, int(0.8 * len(unique))):])
    indices = np.flatnonzero(np.isin(subject, list(val_subjects)))
    if len(indices) == 0:
        raise ValueError("checkpoint validation subjects are absent from this data file")
    eda_scaler = _restore(checkpoint["eda_standardizer"])
    eye_scaler = _restore(checkpoint["eye_standardizer"])
    dataset = MMTEDataset(data["eda_gsr"], data["eye"], data["labels"], indices, eda_scaler, eye_scaler)
    model = MMTED(
        input_channels=checkpoint.get("input_channels", data["eda_gsr"].shape[-1]),
        eye_features=checkpoint["eye_features"], num_classes=checkpoint["num_classes"],
        embed_dim=checkpoint.get("embed_dim", 128), use_attention=checkpoint.get("use_attention", True),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    y, scores = [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=64):
            logits = model(batch["eda_gsr"].to(args.device), batch["eye"].to(args.device))["logits"]
            y.extend(batch["label"].tolist())
            scores.append(logits.cpu().numpy())
    y = np.asarray(y)
    scores = np.concatenate(scores)
    pred = scores.argmax(axis=1)
    result = {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    try:
        result["auc"] = float(roc_auc_score(y, scores, multi_class="ovr"))
    except ValueError:
        result["auc"] = None
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
