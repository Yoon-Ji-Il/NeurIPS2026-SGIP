"""Same-space student model + data utilities.

The student receives only geometry features (no want, no name/category, no
future-frame info). It models a per-candidate score over a variable number of
candidates and is trained to mimic the teacher's final fusion posterior.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
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
    teacher_motion_score: torch.Tensor | None  # (N_i,)


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
    teacher_score_field: str | None = None,
    teacher_posterior_field: str | None = "rubric_posterior",
    teacher_selected_field: str | None = "is_teacher_selected",
    motion_score_field: str | None = "motion_score",
) -> list[StudentSample]:
    rows = _read_csv(csv_path)
    grouped: list[tuple[tuple[str, str], list[dict[str, str]]]] = []
    current_key: tuple[str, str] | None = None
    current_group: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("split", ""), row.get("sample_id", ""))
        if key[0] and key[1]:
            if current_key is None or key == current_key:
                current_group.append(row)
            else:
                grouped.append((current_key, current_group))
                current_group = [row]
            current_key = key
    if current_key is not None:
        grouped.append((current_key, current_group))

    samples: list[StudentSample] = []
    for (split, sample_id), group in grouped:
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

        teacher_motion_score = None
        if motion_score_field is not None and any(motion_score_field in r for r in ordered):
            teacher_motion_score = torch.tensor(
                [_parse_float(r.get(motion_score_field)) for r in ordered],
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
                teacher_motion_score=teacher_motion_score,
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
            teacher_motion_score=s.teacher_motion_score.clone() if s.teacher_motion_score is not None else None,
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
    teacher_motion_score = torch.zeros((batch, n_max), dtype=torch.float32)
    has_motion = any(s.teacher_motion_score is not None for s in samples)
    gt_idx = torch.zeros((batch,), dtype=torch.long)
    teacher_selected_idx = torch.zeros((batch,), dtype=torch.long)
    for i, s in enumerate(samples):
        n = s.features.shape[0]
        features[i, :n] = s.features
        valid_mask[i, :n] = True
        if s.teacher_posterior is not None:
            teacher_posterior[i, :n] = s.teacher_posterior
        if s.teacher_motion_score is not None:
            teacher_motion_score[i, :n] = s.teacher_motion_score
        gt_idx[i] = s.gt_idx
        teacher_selected_idx[i] = max(s.teacher_selected_idx, 0)
    return {
        "features": features,
        "valid_mask": valid_mask,
        "teacher_posterior": teacher_posterior if has_teacher else None,
        "teacher_motion_score": teacher_motion_score if has_motion else None,
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
# Stage-S2 tie-break
# ---------------------------------------------------------------------------
def _normalize_name_batch(
    names: list[str] | list[list[str]] | None,
    batch_size: int,
) -> list[list[str]] | None:
    if names is None:
        return None
    if batch_size == 1 and names and isinstance(names[0], str):
        return [list(names)]  # type: ignore[list-item]
    return [list(row) for row in names]  # type: ignore[arg-type]


def _stage_s2_active_subset(
    p_row: torch.Tensor,
    names: list[str] | None,
    top_idx: list[int],
    *,
    trigger_mode: str,
    margin_threshold: float,
) -> list[int]:
    if trigger_mode == "name_collision":
        if names is None:
            raise ValueError(f"trigger_mode={trigger_mode} requires names")
        name_counts = Counter(names[i] for i in top_idx)
        return [i for i in top_idx if name_counts[names[i]] >= 2]

    if trigger_mode == "score_margin":
        top1_p = max(float(p_row[i].item()) for i in top_idx)
        active_subset = [
            i for i in top_idx
            if (top1_p - float(p_row[i].item())) <= float(margin_threshold)
        ]
        return active_subset if len(active_subset) >= 2 else []

    if trigger_mode == "either":
        if names is None:
            raise ValueError(f"trigger_mode={trigger_mode} requires names")
        name_counts = Counter(names[i] for i in top_idx)
        name_subset = {i for i in top_idx if name_counts[names[i]] >= 2}
        top1_p = max(float(p_row[i].item()) for i in top_idx)
        margin_subset = {
            i for i in top_idx
            if (top1_p - float(p_row[i].item())) <= float(margin_threshold)
        }
        if len(margin_subset) < 2:
            margin_subset = set()
        return [i for i in top_idx if i in (name_subset | margin_subset)]

    raise ValueError(f"Unknown trigger_mode: {trigger_mode}")


def stage_s2_apply(
    p: torch.Tensor,
    *,
    names: list[str] | list[list[str]] | None = None,
    motion_score: torch.Tensor | None = None,
    top_k: int = 3,
    motion_weight: float = 1.0,
    trigger_mode: str = "name_collision",
    margin_threshold: float = 0.05,
    return_triggered: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[bool], list[list[int]]]:
    """Apply Stage-S2 logit-space motion bonus on the active top-k subset."""
    if trigger_mode in ("name_collision", "either") and names is None:
        raise ValueError(f"trigger_mode={trigger_mode} requires names")

    squeezed = p.dim() == 1
    p_batch = p.unsqueeze(0) if squeezed else p
    if p_batch.dim() != 2:
        raise ValueError(f"Expected p with shape [N] or [B,N], got {tuple(p.shape)}")

    batch_size, num_candidates = p_batch.shape
    names_batch = _normalize_name_batch(names, batch_size)
    if names_batch is not None and len(names_batch) != batch_size:
        raise ValueError(f"names batch size {len(names_batch)} != p batch size {batch_size}")

    motion_batch = None
    if motion_score is not None:
        motion_batch = motion_score.unsqueeze(0) if motion_score.dim() == 1 else motion_score
        if motion_batch.shape != p_batch.shape:
            raise ValueError(
                f"motion_score shape {tuple(motion_score.shape)} does not match p shape {tuple(p.shape)}"
            )

    out = p_batch.clone()
    triggered: list[bool] = []
    top_indices: list[list[int]] = []

    for batch_idx in range(batch_size):
        actual_k = min(max(int(top_k), 1), num_candidates)
        top_idx = torch.topk(p_batch[batch_idx], actual_k).indices.tolist()
        top_indices.append(top_idx)
        names_i = names_batch[batch_idx] if names_batch is not None else None
        if names_i is not None and len(names_i) < num_candidates:
            raise ValueError(
                f"names for batch {batch_idx} has length {len(names_i)}; expected at least {num_candidates}"
            )

        active_subset = _stage_s2_active_subset(
            p_batch[batch_idx],
            names_i,
            top_idx,
            trigger_mode=trigger_mode,
            margin_threshold=margin_threshold,
        )
        triggered.append(bool(active_subset))
        if not active_subset or motion_batch is None:
            continue

        active_idx = torch.tensor(active_subset, dtype=torch.long, device=p_batch.device)
        active_motion = motion_batch[batch_idx, active_idx]
        lo = active_motion.min()
        hi = active_motion.max()
        span = hi - lo
        if float(span.item()) <= 1.0e-12:
            continue

        motion_norm = (active_motion - lo) / span
        logits = p_batch[batch_idx].clamp_min(1.0e-8).log()
        boosted = logits.clone()
        boosted[active_idx] = boosted[active_idx] + float(motion_weight) * motion_norm
        out[batch_idx] = torch.softmax(boosted, dim=-1)

    out = out.squeeze(0) if squeezed else out
    if return_triggered:
        return out, triggered, top_indices
    return out


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
    "stage_s2_apply",
]
