from __future__ import annotations

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    masked_logits = logits.masked_fill(~valid, -torch.inf)
    has_valid = valid.any(dim=1, keepdim=True)
    safe_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
    probs = torch.softmax(safe_logits, dim=1)
    return torch.where(has_valid, probs, torch.zeros_like(probs))


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(hidden_dim, output_dim), nn.ReLU()])
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _make_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class SameSpaceSelector(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.candidate_encoder = _make_mlp(input_dim, hidden_dim, hidden_dim, dropout)
        self.context_encoder = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.score_head = _make_head(hidden_dim * 2, hidden_dim, dropout)

    def forward(self, candidate_features: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError(
                f"candidate_features must be [B, N, D], got {tuple(candidate_features.shape)}"
            )
        if valid_mask.shape != candidate_features.shape[:2]:
            raise ValueError(f"valid_mask must be [B, N], got {tuple(valid_mask.shape)}")

        encoded = self.candidate_encoder(candidate_features)
        valid = valid_mask.to(device=candidate_features.device, dtype=torch.float32)
        valid_bool = valid_mask.to(device=candidate_features.device, dtype=torch.bool)

        pooled_mean = (encoded * valid[..., None]).sum(dim=1)
        pooled_mean = pooled_mean / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        masked_for_max = encoded.masked_fill(~valid_bool[..., None], -torch.inf)
        pooled_max = masked_for_max.amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        context = self.context_encoder(torch.cat([pooled_mean, pooled_max], dim=-1))
        expanded_context = context[:, None, :].expand(-1, candidate_features.shape[1], -1)
        pair = torch.cat([encoded, expanded_context], dim=-1)
        logits = self.score_head(pair).squeeze(-1)
        distribution = masked_softmax(logits, valid_mask)
        return {
            "logits": logits,
            "distribution": distribution,
            "candidate_embedding": encoded,
            "sample_context": context,
        }
