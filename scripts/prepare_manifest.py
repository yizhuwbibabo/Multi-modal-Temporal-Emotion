"""Create sequence-safe train/val/test splits from an image manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def split_for(sequence: str, seed: int = 42) -> str:
    value = int(hashlib.sha1(f"{seed}:{sequence}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=42); args = p.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    if not rows or "image" not in rows[0] or "label" not in rows[0]: raise ValueError("input must contain image,label")
    for i, row in enumerate(rows): row["split"] = split_for(row.get("sequence_id", str(i)), args.seed)
    fields = list(rows[0]); Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__": main()
