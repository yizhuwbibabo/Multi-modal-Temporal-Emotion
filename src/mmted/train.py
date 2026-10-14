"""Train the MMTED physiological-signal baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import CalibratedStandardizer, MMTEDataset, load_npz, subject_split
from .model import MMTED


def _metrics(y: list[int], logits: list[np.ndarray], loss: float) -> dict[str, float]:
    y_arr = np.asarray(y)
    scores = np.concatenate(logits, axis=0) if logits else np.empty((0, 2))
    pred = scores.argmax(axis=1)
    result = {"loss": float(loss), "accuracy": float(accuracy_score(y_arr, pred)),
              "macro_f1": float(f1_score(y_arr, pred, average="macro", zero_division=0))}
    try:
        result["auc"] = float(roc_auc_score(y_arr, scores, multi_class="ovr"))
    except ValueError:
        result["auc"] = float("nan")
    return result


def run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    labels: list[int] = []
    logits: list[np.ndarray] = []
    total = 0.0
    for batch in loader:
        x = batch["eda_gsr"].to(device)
        eye = batch["eye"].to(device)
        target = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            output = model(x, eye)
            loss = criterion(output["logits"], target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * len(target)
        labels.extend(target.detach().cpu().tolist())
        logits.append(output["logits"].detach().cpu().numpy())
    return _metrics(labels, logits, total / max(1, len(loader.dataset)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="NPZ with eda_gsr, eye, labels, and optional subject")
    parser.add_argument("--output", default="runs/mmted")
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4, help="Default cross-entropy learning rate")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--already-preprocessed", action="store_true",
                        help="Treat inputs as output from mmted.preprocess and skip a second transform")
    parser.add_argument("--no-attention", action="store_true", help="Ablation: uniform modality pooling")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_npz(args.data)
    subject = data.get("subject", np.arange(len(data["labels"])))
    train_idx, val_idx = subject_split(subject, args.train_fraction, args.seed)
    if len(val_idx) == 0:
        raise ValueError("validation split is empty; provide at least two subjects")

    if args.already_preprocessed:
        eda_scaler = CalibratedStandardizer.identity(data["eda_gsr"].shape[-1])
        eye_scaler = CalibratedStandardizer.identity(data["eye"].shape[-1])
    else:
        eda_scaler = CalibratedStandardizer().fit(data["eda_gsr"][train_idx])
        eye_scaler = CalibratedStandardizer().fit(data["eye"][train_idx])
    train_set = MMTEDataset(data["eda_gsr"], data["eye"], data["labels"], train_idx, eda_scaler, eye_scaler)
    val_set = MMTEDataset(data["eda_gsr"], data["eye"], data["labels"], val_idx, eda_scaler, eye_scaler)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = MMTED(input_channels=data["eda_gsr"].shape[-1], eye_features=data["eye"].shape[-1],
                  num_classes=args.classes, embed_dim=args.embed_dim,
                  use_attention=not args.no_attention).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device)
        val_metrics = run_epoch(model, val_loader, None, args.device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, allow_nan=True))
        checkpoint = {
            "model": model.state_dict(), "input_channels": data["eda_gsr"].shape[-1],
            "eye_features": data["eye"].shape[-1], "num_classes": args.classes,
            "embed_dim": args.embed_dim, "use_attention": not args.no_attention,
            "train_subjects": np.unique(subject[train_idx]).tolist(),
            "val_subjects": np.unique(subject[val_idx]).tolist(),
            "eda_standardizer": eda_scaler.state_dict(), "eye_standardizer": eye_scaler.state_dict(),
        }
        torch.save(checkpoint, output / "last.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
