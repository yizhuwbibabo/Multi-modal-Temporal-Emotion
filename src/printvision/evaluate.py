"""Evaluate a saved checkpoint on a held-out sequence split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import PrintFrameDataset
from .metrics import classification_metrics
from .model import PrintStateModel


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--split", default="test"); p.add_argument("--config", help="override config path"); p.add_argument("--device", default="cpu"); p.add_argument("--batch-size", type=int, default=32); args = p.parse_args(argv)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else ckpt["config"]
    d, m = cfg["data"], cfg["model"]
    dataset = PrintFrameDataset(d["csv"], d["image_root"], d["image_size"], args.split, d.get("telemetry_columns", []), ckpt.get("label_to_id"))
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = PrintStateModel(num_classes=m["num_classes"], telemetry_dim=m["telemetry_dim"], image_size=d["image_size"], patch_size=m["patch_size"], embed_dim=m["embed_dim"], depth=m["depth"], heads=m["num_heads"], dropout=m["dropout"]).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval(); y, pred, prob = [], [], []
    with torch.inference_mode():
        for batch in loader:
            out = model(batch["image"].to(args.device), batch["telemetry"].to(args.device)); y.extend(batch["label"].numpy()); pred.extend(out["logits"].argmax(1).cpu().numpy()); prob.extend(out["logits"].softmax(1).cpu().numpy())
    result = {"split": args.split, "n": len(y), **classification_metrics(y, pred, prob)}
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__": main()
