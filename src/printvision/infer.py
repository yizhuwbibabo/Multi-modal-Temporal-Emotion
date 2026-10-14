"""Single-frame inference and TorchScript export for edge deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .data import _to_tensor
from .model import PrintStateModel


def load_checkpoint(path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    d, m = config["data"], config["model"]
    model = PrintStateModel(
        num_classes=m["num_classes"], telemetry_dim=m["telemetry_dim"], image_size=d["image_size"],
        patch_size=m["patch_size"], embed_dim=m["embed_dim"], depth=m["depth"],
        heads=m["num_heads"], dropout=m["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model, checkpoint.get("label_to_id", {}), d


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--image", required=True); p.add_argument("--telemetry", nargs="*", type=float, default=[]); p.add_argument("--device", default="cpu"); p.add_argument("--export", help="optional TorchScript output path"); args = p.parse_args(argv)
    model, labels, data_config = load_checkpoint(args.checkpoint, args.device)
    image = _to_tensor(Image.open(args.image), data_config["image_size"]).unsqueeze(0).to(args.device)
    telemetry = torch.tensor([args.telemetry], dtype=torch.float32, device=args.device)
    if telemetry.shape[1] != len(data_config.get("telemetry_columns", [])):
        raise ValueError(f"expected {len(data_config.get('telemetry_columns', []))} telemetry values")
    with torch.inference_mode():
        output = model(image, telemetry); probabilities = output["logits"].softmax(-1)[0]
    index = int(probabilities.argmax()); inverse = {v: k for k, v in labels.items()}
    print({"label": inverse.get(index, str(index)), "confidence": float(probabilities[index]), "state": float(output["state"][0])})
    if args.export:
        class EdgeOutputs(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network

            def forward(self, image_tensor, telemetry_tensor):
                output = self.network(image_tensor, telemetry_tensor)
                return output["logits"], output["state"]

        traced = torch.jit.trace(EdgeOutputs(model), (image, telemetry), check_trace=False)
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        traced.save(args.export)
        print(f"saved TorchScript model to {args.export}")


if __name__ == "__main__": main()
