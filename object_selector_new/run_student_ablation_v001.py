#!/usr/bin/env python3
"""Run width/depth ablations for StudentSelectorAblation.

All outputs are written to a versioned run directory. Existing files and runs
are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.model import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    StudentDataset,
    StudentSample,
    build_samples_from_csv,
    collate_student_samples,
    compute_feature_stats,
    normalize_samples,
    parse_feature_columns,
    save_feature_stats,
)
from object_selector_new.model_ablation_v001 import StudentSelectorAblation  # noqa: E402


DEFAULT_INPUT_CSV = REPO_ROOT / "runs/teacher/candidates.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student object selector capacity ablation.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/student_selector_ablation_v001"))
    parser.add_argument("--ablation", choices=("width", "depth", "both"), default="both")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fixed-hidden-dim", type=int, default=256)
    parser.add_argument("--fixed-depth", type=int, default=2)
    parser.add_argument("--feature-columns", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--teacher-target-field", type=str, default="fusion_total")
    parser.add_argument("--teacher-target-mode", choices=("score_softmax", "probability"), default="score_softmax")
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--teacher-selected-field", type=str, default="is_teacher_selected")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-ce", type=float, default=0.1)
    parser.add_argument("--lambda-rank", type=float, default=0.2)
    parser.add_argument("--lambda-pairgap", type=float, default=0.5)
    parser.add_argument("--lambda-gt-ce", type=float, default=0.0)
    parser.add_argument("--rank-topk", type=int, default=3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-min-gap", type=float, default=0.01)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeat", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def next_versioned_path(path: Path) -> Path:
    if not path.exists():
        return path
    match = re.search(r"_v(\d+)$", path.name)
    if match:
        stem = path.name[: match.start()]
        width = len(match.group(1))
        start = int(match.group(1)) + 1
    else:
        stem = path.name
        width = 3
        start = 1
    for version in range(start, 10000):
        candidate = path.with_name(f"{stem}_v{version:0{width}d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find free versioned path for {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def read_csv_header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return set(reader.fieldnames or [])


def validate_input_csv(path: Path, feature_columns: tuple[str, ...], teacher_field: str | None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"input CSV does not exist: {path}")
    header = read_csv_header(path)
    missing_features = [col for col in feature_columns if col not in header]
    if missing_features:
        preview = ", ".join(missing_features[:8])
        raise ValueError(
            f"input CSV is missing {len(missing_features)} feature columns, e.g. {preview}. "
            "Pass a same-space feature CSV with --input-csv."
        )
    if teacher_field and teacher_field not in header:
        raise ValueError(f"teacher target field '{teacher_field}' is missing from {path}")


def distribution_kl(student_dist: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=student_dist.dtype)
    teacher = teacher_dist * valid
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    student = student_dist * valid
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (teacher * (teacher.clamp_min(1.0e-8).log() - student.clamp_min(1.0e-8).log())).sum(dim=1).mean()


def distribution_ce(student_dist: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=student_dist.dtype)
    teacher = teacher_dist * valid
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    student = student_dist * valid
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return -(teacher * student.clamp_min(1.0e-8).log()).sum(dim=1).mean()


def candidate_cross_entropy(logits: torch.Tensor, valid_mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.masked_fill(~valid_mask, -1.0e9), target_idx)


def topk_pair_indices(teacher_dist: torch.Tensor, valid_mask: torch.Tensor, topk: int) -> torch.Tensor:
    masked = teacher_dist.masked_fill(~valid_mask, -1.0)
    actual_k = min(max(int(topk), 1), teacher_dist.shape[1])
    return masked.topk(actual_k, dim=1).indices


def teacher_topk_ranking_loss(
    logits: torch.Tensor,
    teacher_dist: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
    margin: float,
    min_gap: float,
) -> torch.Tensor:
    actual_k = min(max(int(topk), 2), logits.shape[1])
    top_idx = topk_pair_indices(teacher_dist, valid_mask, actual_k)
    top_logits = logits.gather(1, top_idx)
    top_teacher = teacher_dist.gather(1, top_idx)
    top_valid = valid_mask.gather(1, top_idx)
    diff_logits = top_logits[:, :, None] - top_logits[:, None, :]
    diff_teacher = top_teacher[:, :, None] - top_teacher[:, None, :]
    upper = torch.triu(torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool), diagonal=1)
    pair_mask = upper[None] & top_valid[:, :, None] & top_valid[:, None, :] & (diff_teacher > float(min_gap))
    if not pair_mask.any():
        return logits.new_zeros(())
    hinge = F.relu(float(margin) - diff_logits)
    return (hinge * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def teacher_topk_pairgap_loss(
    logits: torch.Tensor,
    teacher_dist: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    actual_k = min(max(int(topk), 2), logits.shape[1])
    top_idx = topk_pair_indices(teacher_dist, valid_mask, actual_k)
    top_logits = logits.gather(1, top_idx)
    top_teacher = teacher_dist.gather(1, top_idx).clamp_min(1.0e-8)
    top_valid = valid_mask.gather(1, top_idx)
    diff_student = top_logits[:, :, None] - top_logits[:, None, :]
    diff_teacher = top_teacher.log()[:, :, None] - top_teacher.log()[:, None, :]
    upper = torch.triu(torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool), diagonal=1)
    pair_mask = upper[None] & top_valid[:, :, None] & top_valid[:, None, :]
    if not pair_mask.any():
        return logits.new_zeros(())
    loss = F.smooth_l1_loss(diff_student, diff_teacher, reduction="none")
    return (loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def score_softmax_target(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(scores / max(float(temperature), 1.0e-6), dim=0)


def probability_target(values: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = values.clamp_min(0.0)
    total = probs.sum()
    if float(total.item()) <= 0.0:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    probs = probs / total
    if abs(float(temperature) - 1.0) <= 1.0e-8:
        return probs
    positive = probs > 0.0
    out = torch.zeros_like(probs)
    out[positive] = torch.softmax(
        probs[positive].clamp_min(1.0e-8).log() / max(float(temperature), 1.0e-6),
        dim=0,
    )
    return out


def build_teacher_target_samples(
    samples: list[StudentSample],
    *,
    target_mode: str,
    temperature: float,
) -> list[StudentSample]:
    converted: list[StudentSample] = []
    for sample in samples:
        teacher_scores = sample.teacher_scores.clone() if sample.teacher_scores is not None else None
        teacher_posterior = sample.teacher_posterior.clone() if sample.teacher_posterior is not None else None
        if target_mode == "score_softmax":
            if teacher_scores is None:
                raise RuntimeError("score_softmax requires teacher_scores")
            teacher_posterior = score_softmax_target(teacher_scores, temperature)
        elif target_mode == "probability":
            source = teacher_posterior if teacher_posterior is not None else teacher_scores
            if source is None:
                raise RuntimeError("probability requires teacher_posterior or teacher_scores")
            teacher_posterior = probability_target(source, temperature)
        else:
            raise ValueError(f"unknown target mode: {target_mode}")

        converted.append(
            StudentSample(
                sample_id=sample.sample_id,
                split=sample.split,
                candidate_ids=list(sample.candidate_ids),
                candidate_names=list(sample.candidate_names),
                candidate_categories=list(sample.candidate_categories),
                features=sample.features.clone(),
                gt_idx=sample.gt_idx,
                gt_object_id=sample.gt_object_id,
                teacher_posterior=teacher_posterior,
                teacher_scores=teacher_scores,
                teacher_selected_idx=int(torch.argmax(teacher_posterior).item()),
            )
        )
    return converted


def split_train_val(
    samples: list[StudentSample],
    *,
    val_ratio: float,
    split_seed: int,
) -> tuple[list[StudentSample], list[StudentSample]]:
    if val_ratio <= 0 or len(samples) < 2:
        return samples, []
    shuffled = list(samples)
    random.Random(split_seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(val_ratio))))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def make_loader(samples: list[StudentSample], batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        StudentDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_student_samples,
        generator=generator,
    )


def compute_loss(
    model: StudentSelectorAblation,
    batch: dict[str, object],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = batch["features"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    teacher_dist = batch["teacher_posterior"]
    if teacher_dist is None:
        raise RuntimeError("teacher_posterior is required")
    teacher_dist = teacher_dist.to(device)
    teacher_idx = batch["teacher_selected_idx"].to(device)
    gt_idx = batch["gt_idx"].to(device)

    out = model(features, valid_mask)
    kl = distribution_kl(out["probs"], teacher_dist, valid_mask)
    teacher_ce = candidate_cross_entropy(out["logits"], valid_mask, teacher_idx)
    gt_ce = candidate_cross_entropy(out["logits"], valid_mask, gt_idx)
    rank = teacher_topk_ranking_loss(
        out["logits"],
        teacher_dist,
        valid_mask,
        topk=args.rank_topk,
        margin=args.rank_margin,
        min_gap=args.rank_min_gap,
    )
    pairgap = teacher_topk_pairgap_loss(out["logits"], teacher_dist, valid_mask, topk=args.rank_topk)
    loss = (
        args.lambda_kl * kl
        + args.lambda_teacher_ce * teacher_ce
        + args.lambda_rank * rank
        + args.lambda_pairgap * pairgap
        + args.lambda_gt_ce * gt_ce
    )
    return loss, {
        "kl": float(kl.detach().item()),
        "ce": float(distribution_ce(out["probs"], teacher_dist, valid_mask).detach().item()),
        "teacher_ce": float(teacher_ce.detach().item()),
        "rank": float(rank.detach().item()),
        "pairgap": float(pairgap.detach().item()),
        "gt_ce": float(gt_ce.detach().item()),
    }


@torch.no_grad()
def evaluate(
    model: StudentSelectorAblation,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "kl": 0.0,
        "ce": 0.0,
        "teacher_top1_hits": 0.0,
        "top3_hits": 0.0,
        "samples": 0.0,
    }
    for batch in loader:
        loss, pieces = compute_loss(model, batch, device, args)
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        teacher_dist = batch["teacher_posterior"].to(device)
        out = model(features, valid_mask)
        student_top1 = out["probs"].argmax(dim=1)
        teacher_top1 = teacher_dist.masked_fill(~valid_mask, -1.0).argmax(dim=1)

        k = min(3, out["probs"].shape[1])
        student_top3 = out["probs"].masked_fill(~valid_mask, -1.0).topk(k, dim=1).indices
        top3_hit = (student_top3 == teacher_top1[:, None]).any(dim=1)

        batch_size = features.shape[0]
        totals["loss"] += float(loss.item()) * batch_size
        totals["kl"] += pieces["kl"] * batch_size
        totals["ce"] += pieces["ce"] * batch_size
        totals["teacher_top1_hits"] += float((student_top1 == teacher_top1).sum().item())
        totals["top3_hits"] += float(top3_hit.sum().item())
        totals["samples"] += float(batch_size)

    denom = max(totals["samples"], 1.0)
    return {
        "loss": totals["loss"] / denom,
        "KL": totals["kl"] / denom,
        "CE": totals["ce"] / denom,
        "teacher_top1_agreement": totals["teacher_top1_hits"] / denom,
        "top3_recall": totals["top3_hits"] / denom,
        "num_samples": denom,
    }


@torch.no_grad()
def benchmark_latency(
    model: StudentSelectorAblation,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return 0.0, 0.0
    features = batch["features"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    for _ in range(max(int(warmup), 0)):
        _ = model(features, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(int(repeat), 1)):
        _ = model(features, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(int(repeat), 1)
    return elapsed_ms, elapsed_ms / max(int(features.shape[0]), 1)


def parameter_count(model: StudentSelectorAblation) -> int:
    return sum(p.numel() for p in model.parameters())


def better_metric(current: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    key = (
        current["teacher_top1_agreement"],
        current["top3_recall"],
        -current["KL"],
        -current["loss"],
    )
    best_key = (
        best["teacher_top1_agreement"],
        best["top3_recall"],
        -best["KL"],
        -best["loss"],
    )
    return key > best_key


def train_one_run(
    *,
    ablation_name: str,
    hidden_dim: int,
    depth: int,
    seed: int,
    train_samples: list[StudentSample],
    val_samples: list[StudentSample],
    test_samples: list[StudentSample],
    feature_columns: tuple[str, ...],
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    set_seed(seed)
    run_name = f"{ablation_name}_h{hidden_dim}_d{depth}_seed{seed}"
    checkpoint_dir = output_dir / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    train_loader = make_loader(train_samples, args.batch_size, True, seed, args.num_workers)
    val_loader = make_loader(val_samples or test_samples, args.batch_size, False, seed + 10_000, args.num_workers)
    test_loader = make_loader(test_samples, args.batch_size, False, seed + 20_000, args.num_workers)

    model = StudentSelectorAblation(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = -1
    best_val: dict[str, float] | None = None
    last_train_loss = float("nan")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_loss(model, batch, device, args)
            loss.backward()
            optimizer.step()
            batch_size = int(batch["features"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        last_train_loss = total_loss / max(total_samples, 1)

        val_metrics = evaluate(model, val_loader, device, args)
        if better_metric(val_metrics, best_val):
            best_val = val_metrics
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "depth": depth,
                    "dropout": args.dropout,
                    "feature_columns": list(feature_columns),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "val_metrics": best_val,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = evaluate(model, train_loader, device, args)
    val_metrics = evaluate(model, val_loader, device, args)
    test_metrics = evaluate(model, test_loader, device, args)
    latency_batch, latency_sample = benchmark_latency(
        model,
        test_loader,
        device,
        warmup=args.latency_warmup,
        repeat=args.latency_repeat,
    )

    return {
        "ablation": ablation_name,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "teacher_top1_agreement": test_metrics["teacher_top1_agreement"],
        "top3_recall": test_metrics["top3_recall"],
        "KL": test_metrics["KL"],
        "CE": test_metrics["CE"],
        "inference_latency_ms_per_batch": latency_batch,
        "inference_latency_ms_per_sample": latency_sample,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "last_epoch_train_loss": last_train_loss,
        "train_teacher_top1_agreement": train_metrics["teacher_top1_agreement"],
        "val_teacher_top1_agreement": val_metrics["teacher_top1_agreement"],
        "test_num_samples": test_metrics["num_samples"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["hidden_dim"]), int(row["depth"])), []).append(row)

    summary: list[dict[str, object]] = []
    for (hidden_dim, depth), group in sorted(grouped.items()):
        top1_mean, top1_std = mean_std(row["teacher_top1_agreement"] for row in group)
        top3_mean, top3_std = mean_std(row["top3_recall"] for row in group)
        kl_mean, kl_std = mean_std(row["KL"] for row in group)
        ce_mean, ce_std = mean_std(row["CE"] for row in group)
        latency_mean, _ = mean_std(row["inference_latency_ms_per_sample"] for row in group)
        summary.append(
            {
                "hidden_dim": hidden_dim,
                "depth": depth,
                "num_seeds": len(group),
                "mean_teacher_top1_agreement": top1_mean,
                "std_teacher_top1_agreement": top1_std,
                "mean_top3_recall": top3_mean,
                "std_top3_recall": top3_std,
                "mean_KL": kl_mean,
                "std_KL": kl_std,
                "mean_CE": ce_mean,
                "std_CE": ce_std,
                "mean_latency_ms_per_sample": latency_mean,
                "parameter_count": int(group[0]["parameter_count"]),
            }
        )
    return summary


def write_readme(output_dir: Path, args: argparse.Namespace, command: str) -> None:
    readme = f"""# Student Selector Ablation

This run measures the capacity-performance trade-off of the student object selector.

## Varied

- Width ablation: hidden_dim in `{args.hidden_dims}` with depth fixed to `{args.fixed_depth}`.
- Depth ablation: depth in `{args.depths}` with hidden_dim fixed to `{args.fixed_hidden_dim}`.
- Seeds: `{args.seeds}`.

## Fixed

- Input CSV: `{args.input_csv}`
- Feature columns: `{args.feature_columns}`
- Teacher target mode: `{args.teacher_target_mode}`
- Teacher target field: `{args.teacher_target_field}`
- Teacher temperature: `{args.teacher_temperature}`
- Batch size: `{args.batch_size}`
- Epochs: `{args.epochs}`
- Learning rate: `{args.lr}`
- Weight decay: `{args.weight_decay}`
- Dropout: `{args.dropout}`
- Train/val split seed: `{args.split_seed}`

## Metrics

- `teacher_top1_agreement`: mean agreement between student top-1 and teacher top-1.
- `top3_recall`: fraction where teacher top-1 appears inside student top-3.
- `KL`: KL(teacher_probs || student_probs).
- `CE`: cross entropy from teacher posterior to student posterior.
- `parameter_count`: trainable plus non-trainable model parameter count.
- `inference_latency_ms_per_batch`: average forward latency on one test batch.
- `inference_latency_ms_per_sample`: batch latency divided by batch size.

## Reproduce

```bash
{command}
```

## Outputs

- Width results: `width_ablation_results.csv`
- Width summary: `width_ablation_summary.csv`
- Width config: `width_ablation_config.json`
- Depth results: `depth_ablation_results.csv`
- Depth summary: `depth_ablation_summary.csv`
- Depth config: `depth_ablation_config.json`
- Checkpoints: `checkpoints/*/best_model.pt`

All files are inside `{output_dir}`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def config_payload(args: argparse.Namespace, output_dir: Path, feature_columns: tuple[str, ...]) -> dict[str, object]:
    payload = vars(args).copy()
    payload["input_csv"] = str(args.input_csv)
    payload["output_dir"] = str(output_dir)
    payload["feature_columns_resolved"] = list(feature_columns)
    return payload


def main() -> None:
    args = parse_args()
    feature_columns = parse_feature_columns(args.feature_columns)
    validate_input_csv(args.input_csv, feature_columns, args.teacher_target_field)

    output_dir = next_versioned_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = choose_device(args.device)

    raw_samples = build_samples_from_csv(
        args.input_csv,
        feature_columns=feature_columns,
        teacher_score_field=args.teacher_target_field if args.teacher_target_mode == "score_softmax" else None,
        teacher_posterior_field=args.teacher_target_field if args.teacher_target_mode == "probability" else None,
        teacher_selected_field=args.teacher_selected_field,
    )
    raw_samples = build_teacher_target_samples(
        raw_samples,
        target_mode=args.teacher_target_mode,
        temperature=args.teacher_temperature,
    )

    train_raw = [sample for sample in raw_samples if sample.split == "train"]
    test_raw = [sample for sample in raw_samples if sample.split == "test"]
    if not train_raw or not test_raw:
        raise RuntimeError(f"expected train/test splits, got train={len(train_raw)} test={len(test_raw)}")
    train_split_raw, val_raw = split_train_val(train_raw, val_ratio=args.val_ratio, split_seed=args.split_seed)

    mean, std = compute_feature_stats(train_split_raw)
    save_feature_stats(output_dir / "feature_stats.json", feature_columns, mean, std)
    train_samples = normalize_samples(train_split_raw, mean, std)
    val_samples = normalize_samples(val_raw, mean, std) if val_raw else []
    test_samples = normalize_samples(test_raw, mean, std)

    command = "python " + " ".join(sys.argv)
    write_readme(output_dir, args, command)

    width_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []

    if args.ablation in {"width", "both"}:
        write_json(output_dir / "width_ablation_config.json", config_payload(args, output_dir, feature_columns))
        for hidden_dim in args.hidden_dims:
            for seed in args.seeds:
                row = train_one_run(
                    ablation_name="width",
                    hidden_dim=int(hidden_dim),
                    depth=int(args.fixed_depth),
                    seed=int(seed),
                    train_samples=train_samples,
                    val_samples=val_samples,
                    test_samples=test_samples,
                    feature_columns=feature_columns,
                    output_dir=output_dir,
                    device=device,
                    args=args,
                )
                width_rows.append(row)
                write_csv(output_dir / "width_ablation_results.csv", width_rows)
                write_csv(output_dir / "width_ablation_summary.csv", summarize(width_rows))

    if args.ablation in {"depth", "both"}:
        write_json(output_dir / "depth_ablation_config.json", config_payload(args, output_dir, feature_columns))
        for depth in args.depths:
            for seed in args.seeds:
                row = train_one_run(
                    ablation_name="depth",
                    hidden_dim=int(args.fixed_hidden_dim),
                    depth=int(depth),
                    seed=int(seed),
                    train_samples=train_samples,
                    val_samples=val_samples,
                    test_samples=test_samples,
                    feature_columns=feature_columns,
                    output_dir=output_dir,
                    device=device,
                    args=args,
                )
                depth_rows.append(row)
                write_csv(output_dir / "depth_ablation_results.csv", depth_rows)
                write_csv(output_dir / "depth_ablation_summary.csv", summarize(depth_rows))

    print(f"Saved ablation outputs to {output_dir}")


if __name__ == "__main__":
    main()
