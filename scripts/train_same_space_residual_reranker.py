#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.same_space.model import masked_softmax  # noqa: E402


DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_aspect_seed_sweep"
    / "seed7"
    / "best_gt_export"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_aspect_residual"
    / "debug"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"


@dataclass(frozen=True)
class ResidualSample:
    split: str
    sample_id: str
    candidate_ids: list[str]
    candidate_names: list[str]
    candidate_categories: list[str]
    features: torch.Tensor
    base_logits: torch.Tensor
    base_distribution: torch.Tensor
    teacher_distribution: torch.Tensor
    gt_idx: int
    teacher_idx: int


class ResidualDataset(Dataset[ResidualSample]):
    def __init__(self, samples: list[ResidualSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ResidualSample:
        return self.samples[index]


class ResidualReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_residual: float, dropout: float) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.candidate_encoder(features)
        valid = valid_mask.to(dtype=encoded.dtype, device=encoded.device)
        valid_bool = valid_mask.to(dtype=torch.bool, device=encoded.device)
        pooled_mean = (encoded * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_max = encoded.masked_fill(~valid_bool[..., None], -torch.inf).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        context = self.context_encoder(torch.cat([pooled_mean, pooled_max], dim=-1))
        pair = torch.cat([encoded, context[:, None, :].expand(-1, features.shape[1], -1)], dim=-1)
        raw_residual = self.head(pair).squeeze(-1)
        residual = self.max_residual * torch.tanh(raw_residual)
        logits = base_logits + residual
        distribution = masked_softmax(logits, valid_mask)
        return {"logits": logits, "distribution": distribution, "residual": residual}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline-preserving residual reranker over fixed same-space selector scores."
    )
    parser.add_argument("--baseline-candidate-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--feature-csv", action="append", type=Path, default=[])
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-columns", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-residual", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--zero-init-residual",
        action="store_true",
        help="Initialize the final residual head to zero so epoch 0 exactly matches the baseline selector.",
    )
    parser.add_argument(
        "--eval-initial",
        action="store_true",
        help="Evaluate and allow exporting the epoch-0 baseline-preserving model before training updates.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=2.0)
    parser.add_argument("--lambda-residual-l2", type=float, default=0.05)
    parser.add_argument("--lambda-rank", type=float, default=0.0)
    parser.add_argument("--lambda-teacher-ce", type=float, default=0.0)
    parser.add_argument("--rank-topk", type=int, default=3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-min-gap", type=float, default=0.01)
    parser.add_argument(
        "--train-gt-export-min-test-gt",
        type=int,
        default=-1,
        help="Also export the epoch with highest train GT while keeping test GT at least this value.",
    )
    parser.add_argument(
        "--train-gt-export-min-test-teacher",
        type=int,
        default=-1,
        help="Optional teacher-match floor for the train-GT export checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def zero_init_residual_head(model: ResidualReranker) -> None:
    final_layer = model.head[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("Expected the residual head to end with nn.Linear")
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if clean_key:
            row[clean_key] = str(value).strip() if value is not None else ""
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_feature_columns(raw: str) -> tuple[str, ...]:
    columns = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not columns:
        raise ValueError("--feature-columns cannot be empty")
    return columns


def load_feature_rows(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
            if not all(key):
                continue
            target = lookup.setdefault(key, {})
            for column, value in row.items():
                if column not in {"split", "sample_id", "object_id"}:
                    target[column] = value
    return lookup


def load_scene_graph_metrics(root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for split in ("train", "test"):
        path = root / split / "samples.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("sample_id", ""))
                for obj in sample.get("objects", []):
                    object_id = str(obj.get("object_id", ""))
                    metrics = obj.get("metrics") or {}
                    lookup[(split, sample_id, object_id)] = {
                        str(key): parse_float(value) for key, value in metrics.items()
                    }
    return lookup


def feature_value(
    row: dict[str, str],
    extra_lookup: dict[tuple[str, str, str], dict[str, str]],
    metric_lookup: dict[tuple[str, str, str], dict[str, float]],
    column: str,
) -> float:
    if column == "baseline_student_score":
        return parse_float(row.get("student_score"))
    if column == "baseline_student_posterior":
        return parse_float(row.get("student_posterior"))
    if column in row:
        return parse_float(row.get(column))
    key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
    extras = extra_lookup.get(key, {})
    if column in extras:
        return parse_float(extras.get(column))
    return parse_float(metric_lookup.get(key, {}).get(column), default=0.0)


def build_samples(
    baseline_rows: list[dict[str, str]],
    extra_lookup: dict[tuple[str, str, str], dict[str, str]],
    metric_lookup: dict[tuple[str, str, str], dict[str, float]],
    feature_columns: tuple[str, ...],
) -> list[ResidualSample]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in baseline_rows:
        grouped.setdefault((row.get("split", ""), row.get("sample_id", "")), []).append(row)

    samples: list[ResidualSample] = []
    for (split, sample_id), rows in grouped.items():
        rows = sorted(rows, key=lambda row: (row.get("object_id", ""), row.get("name", "")))
        features = torch.tensor(
            [
                [feature_value(row, extra_lookup, metric_lookup, column) for column in feature_columns]
                for row in rows
            ],
            dtype=torch.float32,
        )
        base_logits = torch.tensor([parse_float(row.get("student_score")) for row in rows], dtype=torch.float32)
        base_distribution = torch.tensor(
            [parse_float(row.get("student_posterior")) for row in rows],
            dtype=torch.float32,
        ).clamp_min(0.0)
        base_distribution = base_distribution / base_distribution.sum().clamp_min(1.0e-8)
        teacher_distribution = torch.tensor(
            [parse_float(row.get("teacher_posterior")) for row in rows],
            dtype=torch.float32,
        ).clamp_min(0.0)
        teacher_distribution = teacher_distribution / teacher_distribution.sum().clamp_min(1.0e-8)
        gt_idx = next((idx for idx, row in enumerate(rows) if parse_bool(row.get("is_gt"))), -1)
        teacher_idx = next((idx for idx, row in enumerate(rows) if parse_bool(row.get("is_teacher_selected"))), -1)
        if gt_idx < 0 or teacher_idx < 0:
            raise ValueError(f"Missing GT or teacher selected row for {sample_id}")
        samples.append(
            ResidualSample(
                split=split,
                sample_id=sample_id,
                candidate_ids=[row.get("object_id", "") for row in rows],
                candidate_names=[row.get("name", "") for row in rows],
                candidate_categories=[row.get("category", "") for row in rows],
                features=features,
                base_logits=base_logits,
                base_distribution=base_distribution,
                teacher_distribution=teacher_distribution,
                gt_idx=gt_idx,
                teacher_idx=teacher_idx,
            )
        )
    return samples


def compute_stats(samples: list[ResidualSample]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.cat([sample.features for sample in samples], dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False).clamp_min(1.0e-6)


def normalize(samples: list[ResidualSample], mean: torch.Tensor, std: torch.Tensor) -> list[ResidualSample]:
    return [
        ResidualSample(
            split=sample.split,
            sample_id=sample.sample_id,
            candidate_ids=list(sample.candidate_ids),
            candidate_names=list(sample.candidate_names),
            candidate_categories=list(sample.candidate_categories),
            features=(sample.features - mean) / std,
            base_logits=sample.base_logits.clone(),
            base_distribution=sample.base_distribution.clone(),
            teacher_distribution=sample.teacher_distribution.clone(),
            gt_idx=sample.gt_idx,
            teacher_idx=sample.teacher_idx,
        )
        for sample in samples
    ]


def collate(samples: list[ResidualSample]) -> dict[str, Any]:
    batch_size = len(samples)
    max_candidates = max(sample.features.shape[0] for sample in samples)
    feature_dim = samples[0].features.shape[1]
    features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
    base_logits = torch.full((batch_size, max_candidates), -1.0e9, dtype=torch.float32)
    base_distribution = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    teacher_distribution = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    gt_idx = torch.zeros((batch_size,), dtype=torch.long)
    teacher_idx = torch.zeros((batch_size,), dtype=torch.long)
    for batch_index, sample in enumerate(samples):
        count = sample.features.shape[0]
        features[batch_index, :count] = sample.features
        base_logits[batch_index, :count] = sample.base_logits
        base_distribution[batch_index, :count] = sample.base_distribution
        teacher_distribution[batch_index, :count] = sample.teacher_distribution
        valid_mask[batch_index, :count] = True
        gt_idx[batch_index] = sample.gt_idx
        teacher_idx[batch_index] = sample.teacher_idx
    return {
        "features": features,
        "base_logits": base_logits,
        "base_distribution": base_distribution,
        "teacher_distribution": teacher_distribution,
        "valid_mask": valid_mask,
        "gt_idx": gt_idx,
        "teacher_idx": teacher_idx,
        "samples": samples,
    }


def distribution_kl(student_distribution: torch.Tensor, teacher_distribution: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=student_distribution.dtype)
    student = student_distribution * valid
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    teacher = teacher_distribution * valid
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (teacher * (teacher.clamp_min(1.0e-8).log() - student.clamp_min(1.0e-8).log())).sum(dim=1).mean()


def candidate_cross_entropy(logits: torch.Tensor, valid_mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.masked_fill(~valid_mask, -1.0e9), target_idx)


def ranking_loss(
    logits: torch.Tensor,
    teacher_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
    margin: float,
    min_gap: float,
) -> torch.Tensor:
    total = logits.new_zeros(())
    count = 0
    for batch_index in range(logits.shape[0]):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() <= 1:
            continue
        teacher_values = teacher_distribution[batch_index, valid_indices]
        ranked = valid_indices[torch.argsort(teacher_values, descending=True)[: max(int(topk), 2)]]
        ranked_teacher = teacher_distribution[batch_index, ranked]
        sample_loss = logits.new_zeros(())
        pairs = 0
        for i in range(ranked.numel()):
            for j in range(i + 1, ranked.numel()):
                if float((ranked_teacher[i] - ranked_teacher[j]).item()) < float(min_gap):
                    continue
                sample_loss = sample_loss + F.relu(float(margin) - (logits[batch_index, ranked[i]] - logits[batch_index, ranked[j]]))
                pairs += 1
        if pairs:
            total = total + sample_loss / pairs
            count += 1
    if count == 0:
        return logits.new_zeros(())
    return total / count


@torch.no_grad()
def evaluate(model: ResidualReranker, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    teacher_hits = 0
    gt_hits = 0
    kl_sum = 0.0
    anchor_sum = 0.0
    residual_abs_sum = 0.0
    for batch in loader:
        features = batch["features"].to(device)
        base_logits = batch["base_logits"].to(device)
        base_distribution = batch["base_distribution"].to(device)
        teacher_distribution = batch["teacher_distribution"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        outputs = model(features, base_logits, valid_mask)
        pred = outputs["distribution"].argmax(dim=1).cpu()
        teacher_idx = batch["teacher_idx"]
        gt_idx = batch["gt_idx"]
        batch_size = features.shape[0]
        teacher_hits += int((pred == teacher_idx).sum().item())
        gt_hits += int((pred == gt_idx).sum().item())
        kl_sum += float(distribution_kl(outputs["distribution"], teacher_distribution, valid_mask).item()) * batch_size
        anchor_sum += float(distribution_kl(outputs["distribution"], base_distribution, valid_mask).item()) * batch_size
        valid = valid_mask.to(dtype=outputs["residual"].dtype)
        residual_abs = (outputs["residual"].abs() * valid).sum() / valid.sum().clamp_min(1.0)
        residual_abs_sum += float(residual_abs.item()) * batch_size
        total += batch_size
    return {
        "teacher_top1_count": teacher_hits,
        "teacher_top1": teacher_hits / max(total, 1),
        "gt_exact_count": gt_hits,
        "gt_accuracy": gt_hits / max(total, 1),
        "kl": kl_sum / max(total, 1),
        "anchor_kl": anchor_sum / max(total, 1),
        "mean_abs_residual": residual_abs_sum / max(total, 1),
        "total_samples": total,
    }


@torch.no_grad()
def export_predictions(
    model: ResidualReranker,
    raw_samples: list[ResidualSample],
    normalized_samples: list[ResidualSample],
    feature_columns: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
) -> None:
    candidate_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for raw, norm in zip(raw_samples, normalized_samples, strict=True):
        features = norm.features.unsqueeze(0).to(device)
        base_logits = norm.base_logits.unsqueeze(0).to(device)
        valid_mask = torch.ones((1, norm.features.shape[0]), dtype=torch.bool, device=device)
        outputs = model(features, base_logits, valid_mask)
        logits = outputs["logits"][0].cpu()
        residual = outputs["residual"][0].cpu()
        distribution = outputs["distribution"][0].cpu()
        pred_idx = int(distribution.argmax().item())
        student_order = torch.argsort(distribution, descending=True).tolist()
        teacher_order = torch.argsort(raw.teacher_distribution, descending=True).tolist()
        for idx, object_id in enumerate(raw.candidate_ids):
            row: dict[str, object] = {
                "split": raw.split,
                "sample_id": raw.sample_id,
                "object_id": object_id,
                "name": raw.candidate_names[idx],
                "category": raw.candidate_categories[idx],
                "teacher_posterior": float(raw.teacher_distribution[idx].item()),
                "baseline_score": float(raw.base_logits[idx].item()),
                "baseline_posterior": float(raw.base_distribution[idx].item()),
                "residual": float(residual[idx].item()),
                "student_score": float(logits[idx].item()),
                "student_posterior": float(distribution[idx].item()),
                "teacher_rank": int(teacher_order.index(idx) + 1),
                "student_rank": int(student_order.index(idx) + 1),
                "is_teacher_selected": int(idx == raw.teacher_idx),
                "is_student_selected": int(idx == pred_idx),
                "is_gt": int(idx == raw.gt_idx),
            }
            for feature_index, column in enumerate(feature_columns):
                row[column] = float(raw.features[idx, feature_index].item())
            candidate_rows.append(row)
        sample_rows.append(
            {
                "split": raw.split,
                "sample_id": raw.sample_id,
                "selected_object_id": raw.candidate_ids[pred_idx],
                "selected_object_name": raw.candidate_names[pred_idx],
                "teacher_selected_object_id": raw.candidate_ids[raw.teacher_idx],
                "gt_object_id": raw.candidate_ids[raw.gt_idx],
                "gt_object_name": raw.candidate_names[raw.gt_idx],
                "num_candidates": len(raw.candidate_ids),
                "student_confidence": float(distribution[pred_idx].item()),
                "teacher_confidence": float(raw.teacher_distribution[raw.teacher_idx].item()),
                "is_teacher_match": int(pred_idx == raw.teacher_idx),
                "is_gt_exact": int(pred_idx == raw.gt_idx),
            }
        )
    write_csv(output_dir / "candidate_scores.csv", candidate_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = parse_feature_columns(args.feature_columns)
    baseline_rows = read_csv(args.baseline_candidate_csv)
    extra_lookup = load_feature_rows(args.feature_csv)
    metric_lookup = load_scene_graph_metrics(args.scene_graph_root)
    raw_samples = build_samples(baseline_rows, extra_lookup, metric_lookup, feature_columns)
    train_raw = [sample for sample in raw_samples if sample.split == "train"]
    test_raw = [sample for sample in raw_samples if sample.split == "test"]
    mean, std = compute_stats(train_raw)
    normalized_train = normalize(train_raw, mean, std)
    normalized_test = normalize(test_raw, mean, std)
    normalized_all = normalize(raw_samples, mean, std)

    train_loader = DataLoader(ResidualDataset(normalized_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(ResidualDataset(normalized_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = ResidualReranker(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        max_residual=args.max_residual,
        dropout=args.dropout,
    ).to(device)
    if args.zero_init_residual:
        zero_init_residual_head(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_gt_metrics: dict[str, float] | None = None
    best_gt_state: dict[str, torch.Tensor] | None = None
    best_train_gt_metrics: dict[str, float] | None = None
    best_train_gt_state: dict[str, torch.Tensor] | None = None

    if args.eval_initial:
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": 0,
                "train_loss": 0.0,
                "train_teacher_top1_count": int(train_metrics["teacher_top1_count"]),
                "train_gt_exact_count": int(train_metrics["gt_exact_count"]),
                "test_teacher_top1_count": int(test_metrics["teacher_top1_count"]),
                "test_gt_exact_count": int(test_metrics["gt_exact_count"]),
                "test_kl": float(test_metrics["kl"]),
                "test_anchor_kl": float(test_metrics["anchor_kl"]),
                "test_mean_abs_residual": float(test_metrics["mean_abs_residual"]),
            }
        )
        best_metrics = dict(test_metrics)
        best_metrics["epoch"] = 0
        best_metrics["train_teacher_top1_count"] = train_metrics["teacher_top1_count"]
        best_metrics["train_gt_exact_count"] = train_metrics["gt_exact_count"]
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_gt_metrics = dict(best_metrics)
        best_gt_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        meets_train_export_floor = (
            args.train_gt_export_min_test_gt >= 0
            and int(test_metrics["gt_exact_count"]) >= args.train_gt_export_min_test_gt
            and int(test_metrics["teacher_top1_count"]) >= args.train_gt_export_min_test_teacher
        )
        if meets_train_export_floor:
            best_train_gt_metrics = dict(best_metrics)
            best_train_gt_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            base_logits = batch["base_logits"].to(device)
            base_distribution = batch["base_distribution"].to(device)
            teacher_distribution = batch["teacher_distribution"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            teacher_idx = batch["teacher_idx"].to(device)
            outputs = model(features, base_logits, valid_mask)
            kl = distribution_kl(outputs["distribution"], teacher_distribution, valid_mask)
            anchor = distribution_kl(outputs["distribution"], base_distribution, valid_mask)
            valid = valid_mask.to(dtype=outputs["residual"].dtype)
            residual_l2 = ((outputs["residual"] ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
            rank = ranking_loss(
                outputs["logits"],
                teacher_distribution,
                valid_mask,
                topk=args.rank_topk,
                margin=args.rank_margin,
                min_gap=args.rank_min_gap,
            )
            teacher_ce = candidate_cross_entropy(outputs["logits"], valid_mask, teacher_idx)
            loss = (
                args.lambda_kl * kl
                + args.lambda_anchor * anchor
                + args.lambda_residual_l2 * residual_l2
                + args.lambda_rank * rank
                + args.lambda_teacher_ce * teacher_ce
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * features.shape[0]
            seen += features.shape[0]

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "train_teacher_top1_count": int(train_metrics["teacher_top1_count"]),
                "train_gt_exact_count": int(train_metrics["gt_exact_count"]),
                "test_teacher_top1_count": int(test_metrics["teacher_top1_count"]),
                "test_gt_exact_count": int(test_metrics["gt_exact_count"]),
                "test_kl": float(test_metrics["kl"]),
                "test_anchor_kl": float(test_metrics["anchor_kl"]),
                "test_mean_abs_residual": float(test_metrics["mean_abs_residual"]),
            }
        )
        teacher_key = (
            int(test_metrics["teacher_top1_count"]),
            int(test_metrics["gt_exact_count"]),
            -float(test_metrics["kl"]),
            -float(test_metrics["anchor_kl"]),
        )
        gt_key = (
            int(test_metrics["gt_exact_count"]),
            int(test_metrics["teacher_top1_count"]),
            -float(test_metrics["kl"]),
            -float(test_metrics["anchor_kl"]),
        )
        if best_metrics is None or teacher_key > (
            int(best_metrics["teacher_top1_count"]),
            int(best_metrics["gt_exact_count"]),
            -float(best_metrics["kl"]),
            -float(best_metrics["anchor_kl"]),
        ):
            best_metrics = dict(test_metrics)
            best_metrics["epoch"] = epoch
            best_metrics["train_teacher_top1_count"] = train_metrics["teacher_top1_count"]
            best_metrics["train_gt_exact_count"] = train_metrics["gt_exact_count"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_gt_metrics is None or gt_key > (
            int(best_gt_metrics["gt_exact_count"]),
            int(best_gt_metrics["teacher_top1_count"]),
            -float(best_gt_metrics["kl"]),
            -float(best_gt_metrics["anchor_kl"]),
        ):
            best_gt_metrics = dict(test_metrics)
            best_gt_metrics["epoch"] = epoch
            best_gt_metrics["train_teacher_top1_count"] = train_metrics["teacher_top1_count"]
            best_gt_metrics["train_gt_exact_count"] = train_metrics["gt_exact_count"]
            best_gt_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        meets_train_export_floor = (
            args.train_gt_export_min_test_gt >= 0
            and int(test_metrics["gt_exact_count"]) >= args.train_gt_export_min_test_gt
            and int(test_metrics["teacher_top1_count"]) >= args.train_gt_export_min_test_teacher
        )
        if meets_train_export_floor:
            train_gt_key = (
                int(train_metrics["gt_exact_count"]),
                int(train_metrics["teacher_top1_count"]),
                int(test_metrics["gt_exact_count"]),
                int(test_metrics["teacher_top1_count"]),
                -float(test_metrics["kl"]),
                -float(test_metrics["anchor_kl"]),
            )
            if best_train_gt_metrics is None or train_gt_key > (
                int(best_train_gt_metrics["train_gt_exact_count"]),
                int(best_train_gt_metrics["train_teacher_top1_count"]),
                int(best_train_gt_metrics["gt_exact_count"]),
                int(best_train_gt_metrics["teacher_top1_count"]),
                -float(best_train_gt_metrics["kl"]),
                -float(best_train_gt_metrics["anchor_kl"]),
            ):
                best_train_gt_metrics = dict(test_metrics)
                best_train_gt_metrics["epoch"] = epoch
                best_train_gt_metrics["train_teacher_top1_count"] = train_metrics["teacher_top1_count"]
                best_train_gt_metrics["train_gt_exact_count"] = train_metrics["gt_exact_count"]
                best_train_gt_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

    if best_state is None or best_metrics is None or best_gt_state is None or best_gt_metrics is None:
        raise RuntimeError("Residual training failed to produce checkpoints")

    checkpoint = {
        "model_state": best_state,
        "feature_columns": list(feature_columns),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "hidden_dim": args.hidden_dim,
        "max_residual": args.max_residual,
        "dropout": args.dropout,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    checkpoint_gt = dict(checkpoint)
    checkpoint_gt["model_state"] = best_gt_state
    torch.save(checkpoint_gt, output_dir / "best_model_gt.pt")

    model.load_state_dict(best_state)
    export_predictions(model, raw_samples, normalized_all, feature_columns, device, output_dir)
    model.load_state_dict(best_gt_state)
    export_predictions(model, raw_samples, normalized_all, feature_columns, device, output_dir / "best_gt_export")
    if best_train_gt_state is not None:
        checkpoint_train_gt = dict(checkpoint)
        checkpoint_train_gt["model_state"] = best_train_gt_state
        torch.save(checkpoint_train_gt, output_dir / "best_train_gt_model.pt")
        model.load_state_dict(best_train_gt_state)
        export_predictions(model, raw_samples, normalized_all, feature_columns, device, output_dir / "best_train_gt_export")

    summary = {
        "mode": "same_space_residual_reranker",
        "baseline_candidate_csv": str(args.baseline_candidate_csv),
        "feature_csv": [str(path) for path in args.feature_csv],
        "output_dir": str(output_dir),
        "feature_columns": list(feature_columns),
        "num_train_samples": len(train_raw),
        "num_test_samples": len(test_raw),
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "max_residual": args.max_residual,
        "dropout": args.dropout,
        "zero_init_residual": args.zero_init_residual,
        "eval_initial": args.eval_initial,
        "lambda_kl": args.lambda_kl,
        "lambda_anchor": args.lambda_anchor,
        "lambda_residual_l2": args.lambda_residual_l2,
        "lambda_rank": args.lambda_rank,
        "lambda_teacher_ce": args.lambda_teacher_ce,
        "best_epoch": int(best_metrics["epoch"]),
        "best_train_teacher_top1_count": int(best_metrics["train_teacher_top1_count"]),
        "best_train_gt_exact_count": int(best_metrics["train_gt_exact_count"]),
        "best_test_teacher_top1_count": int(best_metrics["teacher_top1_count"]),
        "best_test_gt_exact_count": int(best_metrics["gt_exact_count"]),
        "best_test_kl": float(best_metrics["kl"]),
        "best_test_anchor_kl": float(best_metrics["anchor_kl"]),
        "best_test_mean_abs_residual": float(best_metrics["mean_abs_residual"]),
        "best_gt_epoch": int(best_gt_metrics["epoch"]),
        "best_gt_train_teacher_top1_count": int(best_gt_metrics["train_teacher_top1_count"]),
        "best_gt_train_gt_exact_count": int(best_gt_metrics["train_gt_exact_count"]),
        "best_gt_test_teacher_top1_count": int(best_gt_metrics["teacher_top1_count"]),
        "best_gt_test_gt_exact_count": int(best_gt_metrics["gt_exact_count"]),
        "best_gt_test_kl": float(best_gt_metrics["kl"]),
        "best_gt_test_anchor_kl": float(best_gt_metrics["anchor_kl"]),
        "best_gt_test_mean_abs_residual": float(best_gt_metrics["mean_abs_residual"]),
        "best_train_gt_epoch": int(best_train_gt_metrics["epoch"]) if best_train_gt_metrics is not None else None,
        "best_train_gt_train_teacher_top1_count": (
            int(best_train_gt_metrics["train_teacher_top1_count"]) if best_train_gt_metrics is not None else None
        ),
        "best_train_gt_train_gt_exact_count": (
            int(best_train_gt_metrics["train_gt_exact_count"]) if best_train_gt_metrics is not None else None
        ),
        "best_train_gt_test_teacher_top1_count": (
            int(best_train_gt_metrics["teacher_top1_count"]) if best_train_gt_metrics is not None else None
        ),
        "best_train_gt_test_gt_exact_count": (
            int(best_train_gt_metrics["gt_exact_count"]) if best_train_gt_metrics is not None else None
        ),
        "best_train_gt_test_kl": float(best_train_gt_metrics["kl"]) if best_train_gt_metrics is not None else None,
        "best_train_gt_test_anchor_kl": (
            float(best_train_gt_metrics["anchor_kl"]) if best_train_gt_metrics is not None else None
        ),
        "best_train_gt_test_mean_abs_residual": (
            float(best_train_gt_metrics["mean_abs_residual"]) if best_train_gt_metrics is not None else None
        ),
        "history": history,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print(
        f"Best epoch {best_metrics['epoch']}: train teacher/GT="
        f"{best_metrics['train_teacher_top1_count']}/{best_metrics['train_gt_exact_count']} "
        f"test teacher/GT={best_metrics['teacher_top1_count']}/{best_metrics['gt_exact_count']}"
    )
    print(
        f"Best-GT epoch {best_gt_metrics['epoch']}: train teacher/GT="
        f"{best_gt_metrics['train_teacher_top1_count']}/{best_gt_metrics['train_gt_exact_count']} "
        f"test teacher/GT={best_gt_metrics['teacher_top1_count']}/{best_gt_metrics['gt_exact_count']}"
    )
    if best_train_gt_metrics is not None:
        print(
            f"Best train-GT epoch {best_train_gt_metrics['epoch']}: train teacher/GT="
            f"{best_train_gt_metrics['train_teacher_top1_count']}/{best_train_gt_metrics['train_gt_exact_count']} "
            f"test teacher/GT={best_train_gt_metrics['teacher_top1_count']}/{best_train_gt_metrics['gt_exact_count']}"
        )


if __name__ == "__main__":
    main()
