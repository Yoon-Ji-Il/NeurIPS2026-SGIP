"""Same-space student model + data utilities.

The student receives only geometry features (no want, no name/category, no
future-frame info). It models a per-candidate score over a variable number of
candidates and is trained to mimic the teacher's final fusion posterior.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from object_selector_new.geometry_features import DEFAULT_GEOMETRY_COLUMNS


# Default geometry-only feature columns (no want, no future-frame info).
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = DEFAULT_GEOMETRY_COLUMNS


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------
@dataclass
class StudentSample:
    sample_id: str
    split: str
    candidate_ids: list[str]
    candidate_names: list[str]
    candidate_categories: list[str]
    features: torch.Tensor               # (N_i, F)
    gt_idx: int
    gt_object_id: str
    teacher_posterior: torch.Tensor | None  # (N_i,)
    teacher_scores: torch.Tensor | None     # (N_i,) raw fusion_total
    teacher_selected_idx: int


class StudentDataset(Dataset[StudentSample]):
    def __init__(self, samples: list[StudentSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> StudentSample:
        return self.samples[index]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def parse_feature_columns(raw: str) -> tuple[str, ...]:
    cols = tuple(c.strip() for c in raw.split(",") if c.strip())
    if not cols:
        raise ValueError("feature columns string is empty")
    return cols


def _clean_row(raw: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in raw.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if clean_key:
            out[clean_key] = value.strip() if isinstance(value, str) else str(value)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [_clean_row(row) for row in csv.DictReader(file)]


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_samples_from_csv(
    csv_path: Path,
    *,
    feature_columns: tuple[str, ...],
    teacher_score_field: str | None = "fusion_total",
    teacher_posterior_field: str | None = None,
    teacher_selected_field: str | None = "is_teacher_selected",
) -> list[StudentSample]:
    rows = _read_csv(csv_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("split", ""), row.get("sample_id", ""))
        if key[0] and key[1]:
            grouped.setdefault(key, []).append(row)

    samples: list[StudentSample] = []
    for (split, sample_id), group in grouped.items():
        ordered = sorted(group, key=lambda r: (r.get("object_id", ""), r.get("name", "")))
        candidate_ids = [r.get("object_id", "") for r in ordered]
        candidate_names = [r.get("name", "") for r in ordered]
        candidate_categories = [r.get("category", "") for r in ordered]

        feat = torch.tensor(
            [[_parse_float(r.get(col)) for col in feature_columns] for r in ordered],
            dtype=torch.float32,
        )

        teacher_scores = None
        if teacher_score_field is not None:
            teacher_scores = torch.tensor(
                [_parse_float(r.get(teacher_score_field)) for r in ordered],
                dtype=torch.float32,
            )

        teacher_posterior = None
        if teacher_posterior_field is not None:
            teacher_posterior = torch.tensor(
                [_parse_float(r.get(teacher_posterior_field)) for r in ordered],
                dtype=torch.float32,
            )

        gt_idx = next((i for i, r in enumerate(ordered) if _parse_bool(r.get("is_gt"))), -1)
        gt_count = sum(1 for r in ordered if _parse_bool(r.get("is_gt")))
        if gt_count != 1:
            raise ValueError(f"sample {sample_id} has {gt_count} GT rows; expected 1")

        teacher_selected_idx = -1
        if teacher_selected_field:
            teacher_selected_idx = next(
                (i for i, r in enumerate(ordered) if _parse_bool(r.get(teacher_selected_field))),
                -1,
            )

        samples.append(
            StudentSample(
                sample_id=sample_id,
                split=split,
                candidate_ids=candidate_ids,
                candidate_names=candidate_names,
                candidate_categories=candidate_categories,
                features=feat,
                gt_idx=gt_idx,
                gt_object_id=candidate_ids[gt_idx],
                teacher_posterior=teacher_posterior,
                teacher_scores=teacher_scores,
                teacher_selected_idx=teacher_selected_idx,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------
def compute_feature_stats(samples: list[StudentSample]) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot compute stats on empty samples")
    stacked = torch.cat([s.features for s in samples], dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False).clamp_min(1.0e-6)


def save_feature_stats(path: Path, columns: tuple[str, ...], mean: torch.Tensor, std: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(columns), "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)


def load_feature_stats(path: Path) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return (
        tuple(payload["feature_columns"]),
        torch.tensor(payload["mean"], dtype=torch.float32),
        torch.tensor(payload["std"], dtype=torch.float32).clamp_min(1.0e-6),
    )


def normalize_samples(samples: list[StudentSample], mean: torch.Tensor, std: torch.Tensor) -> list[StudentSample]:
    return [
        StudentSample(
            sample_id=s.sample_id,
            split=s.split,
            candidate_ids=list(s.candidate_ids),
            candidate_names=list(s.candidate_names),
            candidate_categories=list(s.candidate_categories),
            features=(s.features - mean) / std,
            gt_idx=s.gt_idx,
            gt_object_id=s.gt_object_id,
            teacher_posterior=s.teacher_posterior.clone() if s.teacher_posterior is not None else None,
            teacher_scores=s.teacher_scores.clone() if s.teacher_scores is not None else None,
            teacher_selected_idx=s.teacher_selected_idx,
        )
        for s in samples
    ]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_student_samples(samples: list[StudentSample]) -> dict[str, object]:
    batch = len(samples)
    n_max = max(s.features.shape[0] for s in samples)
    f_dim = samples[0].features.shape[1]
    features = torch.zeros((batch, n_max, f_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch, n_max), dtype=torch.bool)
    teacher_posterior = torch.zeros((batch, n_max), dtype=torch.float32)
    has_teacher = any(s.teacher_posterior is not None for s in samples)
    gt_idx = torch.zeros((batch,), dtype=torch.long)
    teacher_selected_idx = torch.zeros((batch,), dtype=torch.long)
    for i, s in enumerate(samples):
        n = s.features.shape[0]
        features[i, :n] = s.features
        valid_mask[i, :n] = True
        if s.teacher_posterior is not None:
            teacher_posterior[i, :n] = s.teacher_posterior
        gt_idx[i] = s.gt_idx
        teacher_selected_idx[i] = max(s.teacher_selected_idx, 0)
    return {
        "features": features,
        "valid_mask": valid_mask,
        "teacher_posterior": teacher_posterior if has_teacher else None,
        "gt_idx": gt_idx,
        "teacher_selected_idx": teacher_selected_idx,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Masked softmax
# ---------------------------------------------------------------------------
def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~valid_mask, -1.0e9), dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StudentSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        h = int(hidden_dim)
        p = float(dropout)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.GELU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(h, h), nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(h, 1),
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.candidate_encoder(features)
        valid_f = valid_mask.to(dtype=encoded.dtype)
        valid_b = valid_mask.to(dtype=torch.bool)

        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_mean = (encoded * valid_f[..., None]).sum(dim=1) / denom

        masked = encoded.masked_fill(~valid_b[..., None], float("-inf"))
        pooled_max = masked.amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        context = self.context_encoder(torch.cat([pooled_mean, pooled_max], dim=-1))
        ctx_b = context[:, None, :].expand(-1, features.shape[1], -1)
        pair = torch.cat([encoded, ctx_b], dim=-1)
        logits = self.head(pair).squeeze(-1)
        return {"logits": logits, "distribution": masked_softmax(logits, valid_mask)}


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "StudentDataset",
    "StudentSample",
    "StudentSelector",
    "build_samples_from_csv",
    "collate_student_samples",
    "compute_feature_stats",
    "load_feature_stats",
    "masked_softmax",
    "normalize_samples",
    "parse_feature_columns",
    "save_feature_stats",
]
