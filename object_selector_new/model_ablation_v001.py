"""Configurable student selector used for capacity ablations.

This file intentionally does not modify ``object_selector_new/model.py``.
It keeps the same DeepSets-style student structure while exposing MLP width
and depth for paper ablations.
"""

from __future__ import annotations

import torch
from torch import nn


def _dropout(dropout: float) -> nn.Module:
    return nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()


def _hidden_mlp(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> nn.Sequential:
    if int(depth) < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    layers: list[nn.Module] = [
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        _dropout(dropout),
    ]
    for _ in range(int(depth) - 1):
        layers.extend(
            [
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                _dropout(dropout),
            ]
        )
    return nn.Sequential(*layers)


def _head_mlp(hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    if int(depth) < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    if int(depth) == 1:
        return nn.Sequential(nn.Linear(2 * int(hidden_dim), 1))

    layers: list[nn.Module] = [
        nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
        nn.GELU(),
        _dropout(dropout),
    ]
    for _ in range(int(depth) - 2):
        layers.extend(
            [
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                _dropout(dropout),
            ]
        )
    layers.append(nn.Linear(int(hidden_dim), 1))
    return nn.Sequential(*layers)


class StudentSelectorAblation(nn.Module):
    """Student selector with configurable hidden dimension and MLP depth.

    The architecture is permutation-equivariant over candidate order:
    per-candidate shared encoder, permutation-invariant masked mean/max
    context, then a shared context-conditioned scoring head.
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h = int(hidden_dim)
        d = int(depth)
        p = float(dropout)

        self.input_dim = int(input_dim)
        self.hidden_dim = h
        self.depth = d
        self.dropout = p

        self.candidate_encoder = _hidden_mlp(self.input_dim, h, d, p)
        self.context_encoder = _hidden_mlp(2 * h, h, d, p)
        self.head = _head_mlp(h, d, p)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.candidate_encoder(features)
        valid_bool = valid_mask.to(dtype=torch.bool)
        valid_float = valid_bool.to(dtype=encoded.dtype)

        denom = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_mean = (encoded * valid_float[..., None]).sum(dim=1) / denom

        masked_encoded = encoded.masked_fill(~valid_bool[..., None], float("-inf"))
        pooled_max = masked_encoded.amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        context = self.context_encoder(torch.cat([pooled_mean, pooled_max], dim=-1))
        expanded_context = context[:, None, :].expand(-1, features.shape[1], -1)
        pair_features = torch.cat([encoded, expanded_context], dim=-1)

        logits = self.head(pair_features).squeeze(-1)
        masked_logits = logits.masked_fill(~valid_bool, -1.0e9)
        probs = torch.softmax(masked_logits, dim=-1)
        return {
            "logits": logits,
            "masked_logits": masked_logits,
            "probs": probs,
            "distribution": probs,
        }


__all__ = ["StudentSelectorAblation"]
