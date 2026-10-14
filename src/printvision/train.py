"""Command-line training for the print-frame project protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PrintFrameDataset, seed_everything
from .metrics import classification_metrics
from .model import PrintStateModel


def _epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    all_y, all_pred, all_prob, total = [], [], [], 0.0
    for batch in tqdm(loader, leave=False):
        image, telemetry, label = batch["image"].to(device), batch["telemetry"].to(device), batch["label"].to(device)
        with torch.set_grad_enabled(training):
            output = model(image, telemetry)
            loss = criterion(output["logits"], label)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * label.shape[0]
        all_y.extend(label.detach().cpu().numpy())
        all_pred.extend(output["logits"].argmax(1).detach().cpu().numpy())
        all_prob.extend(output["logits"].softmax(1).detach().cpu().numpy())
    return {"loss": total / len(loader.dataset), **classification_metrics(all_y, all_pred, all_prob)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    seed_everything(config.get("seed", 42))
    d, m, t = config["data"], config["model"], config["train"]
    train = PrintFrameDataset(d["csv"], d["image_root"], d["image_size"], "train", d.get("telemetry_columns", []))
    val = PrintFrameDataset(d["csv"], d["image_root"], d["image_size"], "val", d.get("telemetry_columns", []), train.label_to_id)
    loader = DataLoader(train, batch_size=t["batch_size"], shuffle=True, num_workers=t.get("num_workers", 0))
    val_loader = DataLoader(val, batch_size=t["batch_size"], num_workers=t.get("num_workers", 0))
    model = PrintStateModel(num_classes=m["num_classes"], telemetry_dim=m["telemetry_dim"], image_size=d["image_size"], patch_size=m["patch_size"], embed_dim=m["embed_dim"], depth=m["depth"], heads=m["num_heads"], dropout=m["dropout"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    out = Path(t["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, t["epochs"] + 1):
        train_metrics = _epoch(model, loader, optimizer, args.device)
        val_metrics = _epoch(model, val_loader, None, args.device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}; history.append(record)
        print(json.dumps(record))
        torch.save({"model": model.state_dict(), "config": config, "label_to_id": train.label_to_id}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
