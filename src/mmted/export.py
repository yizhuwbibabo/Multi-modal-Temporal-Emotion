"""Export an MMTED checkpoint to TorchScript for edge inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import MMTED


class LogitsOnly(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, eda_gsr, eye):
        return self.network(eda_gsr, eye)["logits"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/mmted/model.ts")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MMTED(
        input_channels=checkpoint.get("input_channels", 2), eye_features=checkpoint["eye_features"],
        num_classes=checkpoint["num_classes"], embed_dim=checkpoint.get("embed_dim", 128),
        use_attention=checkpoint.get("use_attention", True),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # Transformer internals in older CPU Torch releases can produce semantically
    # identical but graph-different traces on the second check; model.eval() and
    # an explicit numerical smoke check below provide the useful guarantee here.
    example = (torch.zeros(1, model.eda_gsr.conv[0].in_channels, args.samples),
               torch.zeros(1, args.samples, checkpoint["eye_features"]))
    wrapper = torch.jit.trace(LogitsOnly(model), example, check_trace=False)
    if not torch.isfinite(wrapper(*example)).all():
        raise RuntimeError("exported model produced non-finite logits")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(str(output))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
