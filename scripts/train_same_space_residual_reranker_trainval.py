#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_same_space_residual_reranker import (  # noqa: E402
    DEFAULT_SCENE_GRAPH_ROOT,
    ResidualDataset,
    ResidualReranker,
    build_samples,
    candidate_cross_entropy,
    choose_device,
    collate,
    compute_stats,
    distribution_kl,
    evaluate,
    export_predictions,
    load_feature_rows,
    load_scene_graph_metrics,
    normalize,
    parse_feature_columns,
    ranking_loss,
    read_csv,
    set_seed,
)


DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline10_observed_hand_reach_onefeature_seed7"
    / "hand_last"
    / "best_train_gt_export"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline14_trainval_feature_mlp_distill_seed7"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the same-space residual MLP reranker with a train-scene validation split. "
            "Checkpoint selection uses validation scenes from the training split only; test is reported once."
        )
    )
    parser.add_argument("--baseline-candidate-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--feature-csv", action="append", type=Path, default=[])
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-columns", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-scenes",
        type=str,
        default="",
        help="Optional comma-separated scene names for validation. If omitted, scenes are sampled by seed.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-residual", type=float, default=0.6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.7)
    parser.add_argument("--lambda-residual-l2", type=float, default=0.015)
    parser.add_argument("--lambda-rank", type=float, default=0.7)
    parser.add_argument("--lambda-teacher-ce", type=float, default=0.8)
    parser.add_argument(
        "--lambda-gt-ce",
        type=float,
        default=0.0,
        help="Optional GT CE on train split only. Leave 0.0 for pure teacher distillation.",
    )
    parser.add_argument("--rank-topk", type=int, default=3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-min-gap", type=float, default=0.01)
    parser.add_argument(
        "--select-by",
        choices=("val_gt", "val_teacher", "val_kl"),
        default="val_gt",
        help="Validation-only checkpoint selection target.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def scene_name(sample_id: str) -> str:
    return sample_id.split("|", 1)[0]


def choose_val_scenes(train_samples, *, val_fraction: float, seed: int, explicit: str) -> set[str]:
    if explicit.strip():
        return {part.strip() for part in explicit.split(",") if part.strip()}

    scene_counts: dict[str, int] = {}
    for sample in train_samples:
        scene_counts[scene_name(sample.sample_id)] = scene_counts.get(scene_name(sample.sample_id), 0) + 1

    scenes = sorted(scene_counts)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    target = max(1, round(len(train_samples) * val_fraction))
    chosen: set[str] = set()
    total = 0
    for scene in scenes:
        chosen.add(scene)
        total += scene_counts[scene]
        if total >= target:
            break
    return chosen


def baseline_metrics(samples) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for sample in samples:
        pred_idx = int(torch.argmax(sample.base_distribution).item())
        bucket = output.setdefault(sample.split, {"total": 0, "teacher": 0, "gt": 0})
        bucket["total"] += 1
        bucket["teacher"] += int(pred_idx == sample.teacher_idx)
        bucket["gt"] += int(pred_idx == sample.gt_idx)
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)

    feature_columns = parse_feature_columns(args.feature_columns)
    baseline_rows = read_csv(args.baseline_candidate_csv)
    extra_lookup = load_feature_rows(args.feature_csv)
    metric_lookup = load_scene_graph_metrics(args.scene_graph_root)
    raw_samples = build_samples(baseline_rows, extra_lookup, metric_lookup, feature_columns)
    train_all = [sample for sample in raw_samples if sample.split == "train"]
    test_raw = [sample for sample in raw_samples if sample.split == "test"]
    val_scenes = choose_val_scenes(
        train_all,
        val_fraction=args.val_fraction,
        seed=args.seed,
        explicit=args.val_scenes,
    )
    train_raw = [sample for sample in train_all if scene_name(sample.sample_id) not in val_scenes]
    val_raw = [sample for sample in train_all if scene_name(sample.sample_id) in val_scenes]
    if not train_raw or not val_raw or not test_raw:
        raise RuntimeError("Expected non-empty train/val/test splits")

    mean, std = compute_stats(train_raw)
    normalized_train = normalize(train_raw, mean, std)
    normalized_val = normalize(val_raw, mean, std)
    normalized_test = normalize(test_raw, mean, std)
    normalized_all = normalize(raw_samples, mean, std)

    train_loader = DataLoader(
        ResidualDataset(normalized_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        ResidualDataset(normalized_val),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ResidualDataset(normalized_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = ResidualReranker(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        max_residual=args.max_residual,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_train_metrics: dict[str, float] | None = None

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
            gt_idx = batch["gt_idx"].to(device)

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
            gt_ce = candidate_cross_entropy(outputs["logits"], valid_mask, gt_idx)
            loss = (
                args.lambda_kl * kl
                + args.lambda_anchor * anchor
                + args.lambda_residual_l2 * residual_l2
                + args.lambda_rank * rank
                + args.lambda_teacher_ce * teacher_ce
                + args.lambda_gt_ce * gt_ce
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * features.shape[0]
            seen += features.shape[0]

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_teacher_top1_count": int(train_metrics["teacher_top1_count"]),
            "train_gt_exact_count": int(train_metrics["gt_exact_count"]),
            "val_teacher_top1_count": int(val_metrics["teacher_top1_count"]),
            "val_gt_exact_count": int(val_metrics["gt_exact_count"]),
            "test_teacher_top1_count": int(test_metrics["teacher_top1_count"]),
            "test_gt_exact_count": int(test_metrics["gt_exact_count"]),
            "val_kl": float(val_metrics["kl"]),
            "val_anchor_kl": float(val_metrics["anchor_kl"]),
            "test_kl": float(test_metrics["kl"]),
            "test_anchor_kl": float(test_metrics["anchor_kl"]),
        }
        history.append(record)

        if args.select_by == "val_teacher":
            val_key = (
                int(val_metrics["teacher_top1_count"]),
                int(val_metrics["gt_exact_count"]),
                -float(val_metrics["kl"]),
                -float(val_metrics["anchor_kl"]),
            )
        elif args.select_by == "val_kl":
            val_key = (
                -float(val_metrics["kl"]),
                int(val_metrics["teacher_top1_count"]),
                int(val_metrics["gt_exact_count"]),
                -float(val_metrics["anchor_kl"]),
            )
        else:
            val_key = (
                int(val_metrics["gt_exact_count"]),
                int(val_metrics["teacher_top1_count"]),
                -float(val_metrics["kl"]),
                -float(val_metrics["anchor_kl"]),
            )
        best_key = None
        if best_val_metrics is not None:
            if args.select_by == "val_teacher":
                best_key = (
                    int(best_val_metrics["teacher_top1_count"]),
                    int(best_val_metrics["gt_exact_count"]),
                    -float(best_val_metrics["kl"]),
                    -float(best_val_metrics["anchor_kl"]),
                )
            elif args.select_by == "val_kl":
                best_key = (
                    -float(best_val_metrics["kl"]),
                    int(best_val_metrics["teacher_top1_count"]),
                    int(best_val_metrics["gt_exact_count"]),
                    -float(best_val_metrics["anchor_kl"]),
                )
            else:
                best_key = (
                    int(best_val_metrics["gt_exact_count"]),
                    int(best_val_metrics["teacher_top1_count"]),
                    -float(best_val_metrics["kl"]),
                    -float(best_val_metrics["anchor_kl"]),
                )
        if best_key is None or val_key > best_key:
            best_val_metrics = dict(val_metrics)
            best_val_metrics["epoch"] = epoch
            best_train_metrics = dict(train_metrics)
            best_test_metrics = dict(test_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_val_metrics is None or best_train_metrics is None or best_test_metrics is None:
        raise RuntimeError("No best checkpoint was selected")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "feature_columns": list(feature_columns),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "hidden_dim": args.hidden_dim,
            "max_residual": args.max_residual,
            "dropout": args.dropout,
            "val_scenes": sorted(val_scenes),
        },
        output_dir / "best_model.pt",
    )
    export_predictions(model, raw_samples, normalized_all, feature_columns, device, output_dir)

    base_metrics = baseline_metrics(raw_samples)
    summary = {
        "mode": "same_space_residual_reranker_train_scene_val",
        "baseline_candidate_csv": str(args.baseline_candidate_csv),
        "feature_csv": [str(path) for path in args.feature_csv],
        "output_dir": str(output_dir),
        "feature_columns": list(feature_columns),
        "val_scenes": sorted(val_scenes),
        "num_train_fit_samples": len(train_raw),
        "num_val_samples": len(val_raw),
        "num_test_samples": len(test_raw),
        "baseline_metrics": base_metrics,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "max_residual": args.max_residual,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_kl": args.lambda_kl,
        "lambda_anchor": args.lambda_anchor,
        "lambda_residual_l2": args.lambda_residual_l2,
        "lambda_rank": args.lambda_rank,
        "lambda_teacher_ce": args.lambda_teacher_ce,
        "lambda_gt_ce": args.lambda_gt_ce,
        "select_by": args.select_by,
        "best_epoch_by_val": int(best_val_metrics["epoch"]),
        "best_train_teacher_top1_count": int(best_train_metrics["teacher_top1_count"]),
        "best_train_gt_exact_count": int(best_train_metrics["gt_exact_count"]),
        "best_train_total": int(best_train_metrics["total_samples"]),
        "best_val_teacher_top1_count": int(best_val_metrics["teacher_top1_count"]),
        "best_val_gt_exact_count": int(best_val_metrics["gt_exact_count"]),
        "best_val_total": int(best_val_metrics["total_samples"]),
        "best_test_teacher_top1_count": int(best_test_metrics["teacher_top1_count"]),
        "best_test_gt_exact_count": int(best_test_metrics["gt_exact_count"]),
        "best_test_total": int(best_test_metrics["total_samples"]),
        "best_test_kl": float(best_test_metrics["kl"]),
        "best_test_anchor_kl": float(best_test_metrics["anchor_kl"]),
        "best_test_mean_abs_residual": float(best_test_metrics["mean_abs_residual"]),
        "history": history,
        "sample_summary_csv": str(output_dir / "sample_summary.csv"),
        "candidate_scores_csv": str(output_dir / "candidate_scores.csv"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(
        f"Baseline test teacher/GT="
        f"{base_metrics.get('test', {}).get('teacher', 0)}/{base_metrics.get('test', {}).get('gt', 0)} "
        f"of {base_metrics.get('test', {}).get('total', 0)}"
    )
    print(
        f"Best val epoch {best_val_metrics['epoch']}: "
        f"fit teacher/GT={best_train_metrics['teacher_top1_count']}/{best_train_metrics['gt_exact_count']} "
        f"val teacher/GT={best_val_metrics['teacher_top1_count']}/{best_val_metrics['gt_exact_count']} "
        f"test teacher/GT={best_test_metrics['teacher_top1_count']}/{best_test_metrics['gt_exact_count']} "
        f"of {best_test_metrics['total_samples']}"
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
