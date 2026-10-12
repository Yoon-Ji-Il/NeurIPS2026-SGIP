"""Residual reranker on top of a frozen student.

Bounded residual: residual = max_residual * tanh(raw)
Inputs:
  - geometry features (z-scored using student's train stats)
  - baseline self-signals (score, posterior, score-minus-max/mean, rank)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


BASELINE_SELF_SIGNAL_COLUMNS: tuple[str, ...] = (
    "baseline_student_score",
    "baseline_student_posterior",
    "baseline_score_minus_max",
    "baseline_score_minus_mean",
    "baseline_rank_normalized",
)


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~valid_mask, -1.0e9), dim=-1)


@dataclass
class ResidualConfig:
    input_dim: int
    hidden_dim: int = 64
    num_heads: int = 4
    max_residual: float = 0.5
    dropout: float = 0.1


class ResidualReranker(nn.Module):
    def __init__(self, config: ResidualConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_dim)
        p = float(config.dropout)
        self.max_residual = float(config.max_residual)

        self.candidate_encoder = nn.Sequential(
            nn.Linear(config.input_dim, h), nn.GELU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(h, h), nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=h, num_heads=int(config.num_heads),
            batch_first=True, dropout=p,
        )
        self.attn_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, 2 * h), nn.GELU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(2 * h, h),
        )
        self.ffn_norm = nn.LayerNorm(h)
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(h, 1),
        )

    def forward(
        self, features: torch.Tensor, base_logits: torch.Tensor, valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.candidate_encoder(features)
        key_pad = ~valid_mask
        attended, _ = self.attn(encoded, encoded, encoded, key_padding_mask=key_pad)
        x = self.attn_norm(encoded + attended)
        x = self.ffn_norm(x + self.ffn(x))
        raw = self.head(x).squeeze(-1)
        residual = self.max_residual * torch.tanh(raw)
        logits = base_logits + residual
        return {
            "logits": logits,
            "distribution": masked_softmax(logits, valid_mask),
            "residual": residual,
        }


__all__ = ["BASELINE_SELF_SIGNAL_COLUMNS", "ResidualConfig", "ResidualReranker", "masked_softmax"]
