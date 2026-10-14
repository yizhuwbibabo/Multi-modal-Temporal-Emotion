"""Small ViT and image + telemetry fusion models suitable for edge deployment."""

from __future__ import annotations

import torch
from torch import nn


class TinyViT(nn.Module):
    """A compact Vision Transformer with no external pretrained-weight download.

    It exposes an embedding before the classifier so the same representation can be used
    for defect classification, retrieval, or a downstream VLM adapter.
    """

    def __init__(self, image_size=128, patch_size=16, embed_dim=192, depth=4, heads=6, dropout=0.1):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=4 * embed_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image):
        x = self.patch_embed(image).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.encoder(torch.cat((cls, x), dim=1) + self.pos_embed)
        return self.norm(x[:, 0])


class PrintStateModel(nn.Module):
    """ViT defect head plus synchronized machine-state (telemetry) fusion.

    ``state`` is an optional scalar target (e.g. process quality or remaining risk), making
    the architecture useful for closed-loop control while preserving a simple classifier API.
    """

    def __init__(self, num_classes=2, telemetry_dim=4, **vit_kwargs):
        super().__init__()
        self.encoder = TinyViT(**vit_kwargs)
        dim = vit_kwargs.get("embed_dim", 192)
        self.telemetry = nn.Sequential(nn.LayerNorm(telemetry_dim), nn.Linear(telemetry_dim, dim // 2), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(dim + dim // 2, dim), nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(dim, num_classes)
        self.state_head = nn.Linear(dim, 1)

    def forward(self, image, telemetry=None):
        visual = self.encoder(image)
        if telemetry is None:
            telemetry = visual.new_zeros((visual.shape[0], self.telemetry[0].normalized_shape[0]))
        fused = self.fusion(torch.cat((visual, self.telemetry(telemetry)), dim=1))
        return {"logits": self.classifier(fused), "state": self.state_head(fused).squeeze(-1), "embedding": fused}
