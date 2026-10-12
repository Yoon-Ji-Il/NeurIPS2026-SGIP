"""Residual reranker data utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ResidualSample:
    sample_id: str
    split: str
    candidate_ids: list[str]
    candidate_names: list[str]
    candidate_categories: list[str]
    features: torch.Tensor               # (N_i, F) — geometry (z-scored) + self-signals
    base_logits: torch.Tensor            # (N_i,) — frozen student logit
    base_distribution: torch.Tensor      # (N_i,) — softmax(base_logits)
    teacher_distribution: torch.Tensor   # (N_i,)
    gt_idx: int
    teacher_idx: int
    baseline_idx: int


def _clean_row(raw: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k is None:
            continue
        ck = k.lstrip("\ufeff").strip()
        if ck:
            out[ck] = v.strip() if isinstance(v, str) else str(v)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [_clean_row(r) for r in csv.DictReader(f)]


def _parse_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    t = str(v).strip()
    if not t or t.lower() in {"nan", "none"}:
        return default
    try:
        return float(t)
    except ValueError:
        return default


def _parse_bool(v) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_feature_columns(raw: str) -> tuple[str, ...]:
    cols = tuple(c.strip() for c in raw.split(",") if c.strip())
    if not cols:
        raise ValueError("feature columns string is empty")
    return cols


def load_feature_stats(path: Path) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return (
        tuple(payload["feature_columns"]),
        torch.tensor(payload["mean"], dtype=torch.float32),
        torch.tensor(payload["std"], dtype=torch.float32).clamp_min(1.0e-6),
    )


def compute_baseline_self_signals(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    n = scores.numel()
    if n == 0:
        z = torch.zeros(0, dtype=torch.float32)
        return {
            "baseline_student_posterior": z,
            "baseline_score_minus_max": z,
            "baseline_score_minus_mean": z,
            "baseline_rank_normalized": z,
        }
    posterior = torch.softmax(scores, dim=0)
    score_minus_max = scores - scores.max()
    score_minus_mean = scores - scores.mean()
    sorted_desc = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(sorted_desc, dtype=torch.float32)
    ranks[sorted_desc] = torch.arange(1, n + 1, dtype=torch.float32)
    rank_norm = (n + 1 - ranks) / n  # top → 1, worst → 1/N
    return {
        "baseline_student_posterior": posterior,
        "baseline_score_minus_max": score_minus_max,
        "baseline_score_minus_mean": score_minus_mean,
        "baseline_rank_normalized": rank_norm,
    }


def build_residual_samples_from_csv(
    csv_path: Path,
    *,
    geometry_columns: tuple[str, ...],
    self_signal_columns: tuple[str, ...],
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    baseline_score_field: str = "student_score",
    teacher_posterior_field: str = "teacher_posterior",
) -> list[ResidualSample]:
    rows = _read_csv(csv_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        key = (r.get("split", ""), r.get("sample_id", ""))
        if key[0] and key[1]:
            grouped.setdefault(key, []).append(r)

    if geometry_mean.numel() != len(geometry_columns):
        raise ValueError("geometry_mean length mismatch")
    if geometry_std.numel() != len(geometry_columns):
        raise ValueError("geometry_std length mismatch")

    out: list[ResidualSample] = []
    for (split, sample_id), group in grouped.items():
        ordered = sorted(group, key=lambda r: (r.get("object_id", ""), r.get("name", "")))
        cids = [r.get("object_id", "") for r in ordered]
        cnames = [r.get("name", "") for r in ordered]
        ccats = [r.get("category", "") for r in ordered]

        geom_raw = torch.tensor(
            [[_parse_float(r.get(c)) for c in geometry_columns] for r in ordered],
            dtype=torch.float32,
        )
        geom = (geom_raw - geometry_mean) / geometry_std

        base_logits = torch.tensor(
            [_parse_float(r.get(baseline_score_field)) for r in ordered], dtype=torch.float32,
        )

        derived = compute_baseline_self_signals(base_logits)
        derived["baseline_student_score"] = base_logits.clone()

        signals = []
        for c in self_signal_columns:
            if c not in derived:
                raise ValueError(f"unknown self-signal {c!r}")
            signals.append(derived[c].unsqueeze(-1))
        signal_tensor = torch.cat(signals, dim=-1) if signals else torch.zeros((geom.shape[0], 0), dtype=torch.float32)
        features = torch.cat([geom, signal_tensor], dim=-1)

        teacher = torch.tensor(
            [_parse_float(r.get(teacher_posterior_field)) for r in ordered], dtype=torch.float32,
        ).clamp_min(0.0)
        total = teacher.sum()
        if float(total.item()) <= 0.0:
            teacher = torch.full_like(teacher, 1.0 / max(teacher.numel(), 1))
        else:
            teacher = teacher / total

        base_dist = torch.softmax(base_logits, dim=0)

        gt_idx = next((i for i, r in enumerate(ordered) if _parse_bool(r.get("is_gt"))), -1)
        gt_count = sum(1 for r in ordered if _parse_bool(r.get("is_gt")))
        if gt_count != 1:
            raise ValueError(f"sample {sample_id} has {gt_count} GT rows; expected 1")

        teacher_idx = next(
            (i for i, r in enumerate(ordered) if _parse_bool(r.get("is_teacher_selected"))), -1,
        )
        if teacher_idx < 0:
            teacher_idx = int(torch.argmax(teacher).item())
        baseline_idx = int(torch.argmax(base_logits).item())

        out.append(ResidualSample(
            sample_id=sample_id, split=split,
            candidate_ids=cids, candidate_names=cnames, candidate_categories=ccats,
            features=features,
            base_logits=base_logits, base_distribution=base_dist,
            teacher_distribution=teacher,
            gt_idx=gt_idx, teacher_idx=teacher_idx, baseline_idx=baseline_idx,
        ))
    return out


def collate_residual_samples(samples: list[ResidualSample]) -> dict[str, object]:
    batch = len(samples)
    n_max = max(s.features.shape[0] for s in samples)
    f_dim = samples[0].features.shape[1]
    features = torch.zeros((batch, n_max, f_dim), dtype=torch.float32)
    base_logits = torch.full((batch, n_max), -1.0e9, dtype=torch.float32)
    base_dist = torch.zeros((batch, n_max), dtype=torch.float32)
    teacher_dist = torch.zeros((batch, n_max), dtype=torch.float32)
    valid_mask = torch.zeros((batch, n_max), dtype=torch.bool)
    gt_idx = torch.zeros((batch,), dtype=torch.long)
    teacher_idx = torch.zeros((batch,), dtype=torch.long)
    baseline_idx = torch.zeros((batch,), dtype=torch.long)
    for i, s in enumerate(samples):
        n = s.features.shape[0]
        features[i, :n] = s.features
        base_logits[i, :n] = s.base_logits
        base_dist[i, :n] = s.base_distribution
        teacher_dist[i, :n] = s.teacher_distribution
        valid_mask[i, :n] = True
        gt_idx[i] = s.gt_idx
        teacher_idx[i] = s.teacher_idx
        baseline_idx[i] = s.baseline_idx
    return {
        "features": features, "base_logits": base_logits, "base_distribution": base_dist,
        "teacher_distribution": teacher_dist, "valid_mask": valid_mask,
        "gt_idx": gt_idx, "teacher_idx": teacher_idx, "baseline_idx": baseline_idx,
        "samples": samples,
    }


__all__ = [
    "ResidualSample",
    "build_residual_samples_from_csv",
    "collate_residual_samples",
    "compute_baseline_self_signals",
    "load_feature_stats",
    "parse_feature_columns",
]
