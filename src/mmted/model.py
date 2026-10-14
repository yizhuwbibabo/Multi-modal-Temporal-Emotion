"""MMTED network: EDA/GSR Conv1D + Transformer, eye Transformer, late fusion."""

from __future__ import annotations

import torch
from torch import nn


class EDAGSRBranch(nn.Module):
    """EDA/GSR encoder with two temporal convolutions and a Transformer."""

    def __init__(self, input_channels=2, embed_dim=128, heads=4, layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(embed_dim),
        )
        block = nn.TransformerEncoderLayer(embed_dim, heads, ff_dim, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        return self.norm(self.transformer(x).mean(1))


class EyeBranch(nn.Module):
    def __init__(self, input_features=6, embed_dim=128, heads=4, layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_features, embed_dim)
        block = nn.TransformerEncoderLayer(embed_dim, heads, ff_dim, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x): return self.norm(self.transformer(self.proj(x)).mean(1))


class MMTED(nn.Module):
    """Multimodal temporal classifier supporting 4-class and 8-class tasks."""

    def __init__(self, eye_features=6, num_classes=4, input_channels=2, embed_dim=128, heads=4, layers=2, ff_dim=256, dropout=0.1, use_attention=True):
        super().__init__()
        self.eda_gsr = EDAGSRBranch(input_channels, embed_dim, heads, layers, ff_dim, dropout)
        self.eye = EyeBranch(eye_features, embed_dim, heads, layers, ff_dim, dropout)
        self.use_attention = use_attention
        self.modality_score = nn.Linear(embed_dim, 1) if use_attention else None
        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, eda_gsr, eye):
        e = self.eda_gsr(eda_gsr); v = self.eye(eye)
        modalities = torch.stack((e, v), dim=1)
        if self.use_attention:
            weights = torch.softmax(self.modality_score(modalities).squeeze(-1), dim=1)
            fused = (modalities * weights.unsqueeze(-1)).sum(dim=1)
        else:
            fused = modalities.mean(dim=1)
        return {"logits": self.classifier(fused), "eda_embedding": e, "eye_embedding": v, "fusion_weights": weights if self.use_attention else None, "embedding": fused}
