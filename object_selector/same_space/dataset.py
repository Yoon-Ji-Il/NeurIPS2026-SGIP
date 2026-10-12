from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "teacher_candidate_scores.csv"
)
DEFAULT_SAME_SPACE_FEATURE_COLUMNS = (
    "face_alignment_norm",
    "body_alignment_norm",
    "face_line_score",
    "future_hand_contact_score",
    "future_root_endpoint_score",
    "root_alignment",
    "torso_alignment",
    "root_torso_alignment",
    "face_normal_alignment",
)


@dataclass(frozen=True)
class SameSpaceSample:
    sample_id: str
    split: str
    split_index: int
    candidate_ids: list[str]
    candidate_names: list[str]
    candidate_categories: list[str]
    features: torch.Tensor
    gt_idx: int
    gt_object_id: str
    gt_indices: list[int]
    gt_object_ids: list[str]
    teacher_posterior: torch.Tensor | None = None
    teacher_scores: torch.Tensor | None = None
    teacher_selected_idx: int = -1


class SameSpaceDataset(Dataset[SameSpaceSample]):
    def __init__(self, samples: list[SameSpaceSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SameSpaceSample:
        return self.samples[index]


def parse_feature_columns(raw: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        columns = [str(item).strip() for item in raw]
    else:
        columns = [part.strip() for part in str(raw).split(",")]
    output = tuple(column for column in columns if column)
    if not output:
        raise ValueError("At least one feature column is required")
    return output


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        output[clean_key] = str(value).strip() if value is not None else ""
    return output


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [_clean_row(row) for row in csv.DictReader(file)]


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_string_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _load_scene_graph_metrics(
    rows: list[dict[str, str]],
    needed_columns: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    missing = {column for column in needed_columns if column not in rows[0]}
    if not missing:
        return {}

    metrics_by_split: dict[str, dict[str, dict[str, float]]] = {}
    splits = sorted({row.get("split", "") for row in rows if row.get("split", "")})
    for split in splits:
        target = REPO_ROOT / "object_selector" / "scene_graphs" / split / "samples.jsonl"
        if not target.exists():
            continue
        sample_lookup: dict[str, dict[str, float]] = {}
        with target.open("r", encoding="utf-8") as file:
            for line in file:
                payload = json.loads(line)
                sample_id = str(payload.get("sample_id", ""))
                object_lookup: dict[str, float] = {}
                for obj in payload.get("objects", []):
                    object_id = str(obj.get("object_id", ""))
                    metrics = obj.get("metrics", {})
                    for column in missing:
                        if column in metrics:
                            object_lookup[(object_id, column)] = _parse_float(metrics.get(column))
                sample_lookup[sample_id] = object_lookup
        metrics_by_split[split] = sample_lookup
    return metrics_by_split


def _feature_value(
    row: dict[str, str],
    *,
    column: str,
    scene_graph_metrics: dict[str, dict[str, dict[str, float]]],
) -> float:
    if column in row:
        return _parse_float(row.get(column))
    split = row.get("split", "")
    sample_id = row.get("sample_id", "")
    object_id = row.get("object_id", "")
    return _parse_float(
        scene_graph_metrics.get(split, {}).get(sample_id, {}).get((object_id, column)),
        default=0.0,
    )


def build_samples_from_csv(
    path: str | Path,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_SAME_SPACE_FEATURE_COLUMNS,
    teacher_posterior_field: str | None = None,
    teacher_score_field: str | None = None,
    teacher_selected_field: str | None = None,
) -> list[SameSpaceSample]:
    rows = _read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    if "sample_id" not in rows[0] or "object_id" not in rows[0]:
        raise KeyError(f"{path} must contain sample_id and object_id columns")

    scene_graph_metrics = _load_scene_graph_metrics(rows, feature_columns)
    unresolved = [
        column
        for column in feature_columns
        if column not in rows[0]
        and not any(
            (row.get("object_id", ""), column)
            in scene_graph_metrics.get(row.get("split", ""), {}).get(row.get("sample_id", ""), {})
            for row in rows
        )
    ]
    if unresolved:
        raise KeyError(f"Missing feature columns in {path}: {unresolved}")

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id:
            group_key = (
                row.get("split", ""),
                row.get("split_index", "-1"),
                sample_id,
            )
            grouped.setdefault(group_key, []).append(row)

    samples: list[SameSpaceSample] = []
    for (_, _, sample_id), sample_rows in grouped.items():
        sample_rows = sorted(sample_rows, key=lambda row: (row.get("object_id", ""), row.get("name", "")))
        split = sample_rows[0].get("split", "")
        candidate_ids = [row.get("object_id", "") for row in sample_rows]
        candidate_names = [row.get("name", "") for row in sample_rows]
        candidate_categories = [row.get("category", "") for row in sample_rows]
        features = torch.tensor(
            [
                [
                    _feature_value(
                        row,
                        column=column,
                        scene_graph_metrics=scene_graph_metrics,
                    )
                    for column in feature_columns
                ]
                for row in sample_rows
            ],
            dtype=torch.float32,
        )
        gt_flags = [_to_bool(row.get("is_gt")) for row in sample_rows]
        gt_indices = [idx for idx, flag in enumerate(gt_flags) if flag]
        gt_idx = gt_indices[0] if gt_indices else -1
        accepted_gt_ids = _parse_string_list(
            sample_rows[0].get("accepted_gt_object_ids", "")
        )
        if not accepted_gt_ids:
            accepted_gt_ids = [candidate_ids[idx] for idx in gt_indices]
        gt_object_id = accepted_gt_ids[0] if accepted_gt_ids else ""
        split_index = int(_parse_float(sample_rows[0].get("split_index"), -1.0))

        teacher_posterior = None
        if teacher_posterior_field and teacher_posterior_field in sample_rows[0]:
            values = torch.tensor(
                [_parse_float(row.get(teacher_posterior_field)) for row in sample_rows],
                dtype=torch.float32,
            ).clamp_min(0.0)
            total = values.sum()
            if float(total.item()) <= 0.0:
                teacher_posterior = torch.full_like(values, 1.0 / max(values.numel(), 1))
            else:
                teacher_posterior = values / total

        teacher_scores = None
        if teacher_score_field and teacher_score_field in sample_rows[0]:
            teacher_scores = torch.tensor(
                [_parse_float(row.get(teacher_score_field)) for row in sample_rows],
                dtype=torch.float32,
            )

        teacher_selected_idx = -1
        if teacher_selected_field and teacher_selected_field in sample_rows[0]:
            teacher_flags = [_to_bool(row.get(teacher_selected_field)) for row in sample_rows]
            teacher_selected_idx = next((idx for idx, flag in enumerate(teacher_flags) if flag), -1)
        if teacher_selected_idx < 0 and teacher_posterior is not None and teacher_posterior.numel() > 0:
            teacher_selected_idx = int(torch.argmax(teacher_posterior).item())

        samples.append(
            SameSpaceSample(
                sample_id=sample_id,
                split=split,
                split_index=split_index,
                candidate_ids=candidate_ids,
                candidate_names=candidate_names,
                candidate_categories=candidate_categories,
                features=features,
                gt_idx=gt_idx,
                gt_object_id=gt_object_id,
                gt_indices=gt_indices,
                gt_object_ids=accepted_gt_ids,
                teacher_posterior=teacher_posterior,
                teacher_scores=teacher_scores,
                teacher_selected_idx=teacher_selected_idx,
            )
        )
    return samples


def compute_feature_stats(samples: list[SameSpaceSample]) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot compute feature stats for an empty sample list")
    stacked = torch.cat([sample.features for sample in samples], dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return mean, std


def normalize_samples(
    samples: list[SameSpaceSample],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[SameSpaceSample]:
    normalized: list[SameSpaceSample] = []
    for sample in samples:
        normalized.append(
            SameSpaceSample(
                sample_id=sample.sample_id,
                split=sample.split,
                split_index=sample.split_index,
                candidate_ids=list(sample.candidate_ids),
                candidate_names=list(sample.candidate_names),
                candidate_categories=list(sample.candidate_categories),
                features=(sample.features - mean) / std,
                gt_idx=sample.gt_idx,
                gt_object_id=sample.gt_object_id,
                gt_indices=list(sample.gt_indices),
                gt_object_ids=list(sample.gt_object_ids),
                teacher_posterior=sample.teacher_posterior.clone() if sample.teacher_posterior is not None else None,
                teacher_scores=sample.teacher_scores.clone() if sample.teacher_scores is not None else None,
                teacher_selected_idx=sample.teacher_selected_idx,
            )
        )
    return normalized


def save_feature_stats(
    path: str | Path,
    *,
    feature_columns: tuple[str, ...],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    payload = {
        "feature_columns": list(feature_columns),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_feature_stats(path: str | Path) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    feature_columns = parse_feature_columns(payload["feature_columns"])
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    return feature_columns, mean, std


def collate_same_space_samples(samples: list[SameSpaceSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    max_candidates = max(sample.features.shape[0] for sample in samples)
    feature_dim = samples[0].features.shape[1]
    features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    gt_idx = torch.full((batch_size,), -1, dtype=torch.long)
    gt_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)

    teacher_posterior = None
    if all(sample.teacher_posterior is not None for sample in samples):
        teacher_posterior = torch.zeros((batch_size, max_candidates), dtype=torch.float32)

    teacher_selected_idx = torch.full((batch_size,), -1, dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        count = sample.features.shape[0]
        features[batch_index, :count] = sample.features
        valid_mask[batch_index, :count] = True
        gt_idx[batch_index] = sample.gt_idx
        for candidate_index in sample.gt_indices:
            if 0 <= candidate_index < count:
                gt_mask[batch_index, candidate_index] = True
        teacher_selected_idx[batch_index] = sample.teacher_selected_idx
        if teacher_posterior is not None and sample.teacher_posterior is not None:
            teacher_posterior[batch_index, :count] = sample.teacher_posterior

    return {
        "sample_id": [sample.sample_id for sample in samples],
        "split": [sample.split for sample in samples],
        "split_index": [sample.split_index for sample in samples],
        "candidate_ids": [sample.candidate_ids for sample in samples],
        "candidate_names": [sample.candidate_names for sample in samples],
        "candidate_categories": [sample.candidate_categories for sample in samples],
        "features": features,
        "valid_mask": valid_mask,
        "gt_idx": gt_idx,
        "gt_mask": gt_mask,
        "gt_object_id": [sample.gt_object_id for sample in samples],
        "gt_object_ids": [sample.gt_object_ids for sample in samples],
        "teacher_posterior": teacher_posterior,
        "teacher_selected_idx": teacher_selected_idx,
    }
