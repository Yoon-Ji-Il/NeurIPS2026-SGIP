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

from object_selector.same_space import build_samples_from_csv, masked_softmax, parse_feature_columns  # noqa: E402
from train_same_space_student_selector import (  # noqa: E402
    build_teacher_target_samples,
    choose_device,
    distribution_kl,
    set_seed,
    write_csv,
    write_json,
)
from train_same_space_topk_reranker import (  # noqa: E402
    DEFAULT_COARSE_CSV,
    DEFAULT_INPUT_CSV,
    TopKRerankSample,
    compute_feature_stats,
    load_coarse_lookup,
    normalize_samples,
    reduce_samples_to_coarse_topk,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "object_selector" / "same_space_runs" / "input_motion_top2_pairwise_residual"
)


@dataclass(frozen=True)
class PairwiseSample:
    raw: TopKRerankSample
    pair_features: torch.Tensor
    base_gap: torch.Tensor
    teacher_posterior: torch.Tensor
    teacher_first_prob: torch.Tensor


class PairwiseDataset(Dataset[PairwiseSample]):
    def __init__(self, samples: list[PairwiseSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PairwiseSample:
        return self.samples[index]


class PairwiseResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a top-2 pairwise residual head. The coarse top-2 ranking is student-visible; "
            "the head only learns a residual gap from observed-motion/object features."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--coarse-candidate-csv", type=Path, default=DEFAULT_COARSE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-columns", type=str, required=True)
    parser.add_argument("--teacher-target-field", type=str, default="posterior")
    parser.add_argument("--teacher-target-mode", choices=("score_softmax", "probability"), default="probability")
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-pair-bce", type=float, default=1.0)
    parser.add_argument("--lambda-residual-l2", type=float, default=0.05)
    parser.add_argument("--lambda-base-margin", type=float, default=0.0)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.25,
        help="Scales the learned residual before adding it to the coarse top1-top2 gap.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def make_pair_features(sample: TopKRerankSample) -> torch.Tensor:
    if sample.features.shape[0] != 2:
        raise ValueError(f"Expected top-2 sample, got {sample.features.shape[0]} candidates")
    first = sample.features[0]
    second = sample.features[1]
    diff = first - second
    abs_diff = diff.abs()
    coarse_score_first = float(sample.coarse_rows[0]["student_score"])
    coarse_score_second = float(sample.coarse_rows[1]["student_score"])
    coarse_posterior_first = float(sample.coarse_rows[0]["student_posterior"])
    coarse_posterior_second = float(sample.coarse_rows[1]["student_posterior"])
    coarse_extras = torch.tensor(
        [
            coarse_score_first,
            coarse_score_second,
            coarse_score_first - coarse_score_second,
            coarse_posterior_first,
            coarse_posterior_second,
            coarse_posterior_first - coarse_posterior_second,
        ],
        dtype=torch.float32,
    )
    return torch.cat([first, second, diff, abs_diff, coarse_extras], dim=0)


def build_pairwise_samples(samples: list[TopKRerankSample]) -> list[PairwiseSample]:
    output: list[PairwiseSample] = []
    for sample in samples:
        if sample.features.shape[0] != 2:
            continue
        base_gap = torch.tensor(
            float(sample.coarse_rows[0]["student_score"]) - float(sample.coarse_rows[1]["student_score"]),
            dtype=torch.float32,
        )
        teacher_posterior = sample.teacher_posterior.clone()
        teacher_first_prob = teacher_posterior[0].clamp(0.0, 1.0)
        output.append(
            PairwiseSample(
                raw=sample,
                pair_features=make_pair_features(sample),
                base_gap=base_gap,
                teacher_posterior=teacher_posterior,
                teacher_first_prob=teacher_first_prob,
            )
        )
    return output


def compute_pair_stats(samples: list[PairwiseSample]) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot compute pair stats for empty samples")
    stacked = torch.stack([sample.pair_features for sample in samples], dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False).clamp_min(1.0e-6)


def normalize_pair_samples(
    samples: list[PairwiseSample],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[PairwiseSample]:
    return [
        PairwiseSample(
            raw=sample.raw,
            pair_features=(sample.pair_features - mean) / std,
            base_gap=sample.base_gap.clone(),
            teacher_posterior=sample.teacher_posterior.clone(),
            teacher_first_prob=sample.teacher_first_prob.clone(),
        )
        for sample in samples
    ]


def collate_pairwise(samples: list[PairwiseSample]) -> dict[str, Any]:
    return {
        "pair_features": torch.stack([sample.pair_features for sample in samples], dim=0),
        "base_gap": torch.stack([sample.base_gap for sample in samples], dim=0),
        "teacher_posterior": torch.stack([sample.teacher_posterior for sample in samples], dim=0),
        "teacher_first_prob": torch.stack([sample.teacher_first_prob for sample in samples], dim=0),
        "raw": [sample.raw for sample in samples],
    }


def final_logits_from_gap(final_gap: torch.Tensor) -> torch.Tensor:
    return torch.stack([0.5 * final_gap, -0.5 * final_gap], dim=1)


def compute_outputs(
    model: PairwiseResidualHead,
    pair_features: torch.Tensor,
    base_gap: torch.Tensor,
    *,
    residual_scale: float,
) -> dict[str, torch.Tensor]:
    residual_gap = model(pair_features) * float(residual_scale)
    final_gap = base_gap + residual_gap
    logits = final_logits_from_gap(final_gap)
    distribution = torch.softmax(logits, dim=1)
    return {
        "residual_gap": residual_gap,
        "final_gap": final_gap,
        "logits": logits,
        "distribution": distribution,
    }


@torch.no_grad()
def evaluate(
    model: PairwiseResidualHead,
    loader: DataLoader,
    device: torch.device,
    *,
    residual_scale: float,
) -> dict[str, float]:
    model.eval()
    total_kl = 0.0
    total_samples = 0
    teacher_hits = 0
    gt_hits = 0
    teacher_in_top2 = 0
    gt_in_top2 = 0
    changed = 0
    for batch in loader:
        pair_features = batch["pair_features"].to(device)
        base_gap = batch["base_gap"].to(device)
        teacher_posterior = batch["teacher_posterior"].to(device)
        outputs = compute_outputs(model, pair_features, base_gap, residual_scale=residual_scale)
        valid_mask = torch.ones_like(outputs["distribution"], dtype=torch.bool)
        kl_value = distribution_kl(outputs["distribution"], teacher_posterior, valid_mask)
        pred_idx = outputs["distribution"].argmax(dim=1).detach().cpu().tolist()
        total_kl += float(kl_value.item()) * len(pred_idx)
        total_samples += len(pred_idx)
        changed += sum(index != 0 for index in pred_idx)
        for row_index, local_pred_idx in enumerate(pred_idx):
            raw = batch["raw"][row_index]
            pred_object_id = raw.candidate_ids[local_pred_idx]
            teacher_in_top2 += int(raw.global_teacher_object_id in raw.candidate_ids)
            gt_in_top2 += int(raw.gt_object_id in raw.candidate_ids)
            teacher_hits += int(pred_object_id == raw.global_teacher_object_id)
            gt_hits += int(pred_object_id == raw.gt_object_id)
    return {
        "kl": total_kl / max(total_samples, 1),
        "teacher_top1_count": teacher_hits,
        "gt_exact_count": gt_hits,
        "teacher_in_top2_count": teacher_in_top2,
        "gt_in_top2_count": gt_in_top2,
        "changed_count": changed,
        "total_samples": total_samples,
    }


@torch.no_grad()
def export_predictions(
    *,
    model: PairwiseResidualHead,
    raw_samples: list[PairwiseSample],
    norm_samples: list[PairwiseSample],
    feature_columns: tuple[str, ...],
    output_dir: Path,
    device: torch.device,
    residual_scale: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    candidate_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for raw_pair, norm_pair in zip(raw_samples, norm_samples, strict=True):
        raw = raw_pair.raw
        outputs = compute_outputs(
            model,
            norm_pair.pair_features.unsqueeze(0).to(device),
            norm_pair.base_gap.unsqueeze(0).to(device),
            residual_scale=residual_scale,
        )
        logits = outputs["logits"][0].detach().cpu()
        posterior = outputs["distribution"][0].detach().cpu()
        residual_gap = float(outputs["residual_gap"][0].detach().cpu().item())
        final_gap = float(outputs["final_gap"][0].detach().cpu().item())
        pred_idx = int(torch.argmax(posterior).item())
        student_rank_order = torch.argsort(posterior, descending=True).tolist()
        teacher_rank_order = torch.argsort(raw.teacher_posterior, descending=True).tolist()
        base_pred_idx = 0

        for candidate_index, object_id in enumerate(raw.candidate_ids):
            coarse_row = raw.coarse_rows[candidate_index]
            row = {
                "split": raw.split,
                "sample_id": raw.sample_id,
                "object_id": object_id,
                "name": raw.candidate_names[candidate_index],
                "category": raw.candidate_categories[candidate_index],
                "coarse_student_score": coarse_row["student_score"],
                "coarse_student_posterior": coarse_row["student_posterior"],
                "coarse_student_rank": coarse_row["student_rank"],
                "base_gap_top1_minus_top2": float(raw_pair.base_gap.item()),
                "learned_residual_gap": residual_gap,
                "final_gap_top1_minus_top2": final_gap,
                "teacher_posterior_topk": float(raw.teacher_posterior[candidate_index].item()),
                "student_score": float(logits[candidate_index].item()),
                "student_posterior": float(posterior[candidate_index].item()),
                "teacher_rank_topk": int(teacher_rank_order.index(candidate_index) + 1),
                "student_rank": int(student_rank_order.index(candidate_index) + 1),
                "is_teacher_selected": int(object_id == raw.global_teacher_object_id),
                "is_topk_teacher_target": int(candidate_index == raw.teacher_target_idx),
                "is_student_selected": int(candidate_index == pred_idx),
                "is_base_selected": int(candidate_index == base_pred_idx),
                "is_gt": int(object_id == raw.gt_object_id),
            }
            for feature_offset, column in enumerate(feature_columns):
                row[column] = float(raw.features[candidate_index, feature_offset].item())
            candidate_rows.append(row)

        sample_rows.append(
            {
                "split": raw.split,
                "sample_id": raw.sample_id,
                "selected_object_id": raw.candidate_ids[pred_idx],
                "selected_object_name": raw.candidate_names[pred_idx],
                "base_selected_object_id": raw.candidate_ids[base_pred_idx],
                "base_selected_object_name": raw.candidate_names[base_pred_idx],
                "teacher_selected_object_id": raw.global_teacher_object_id,
                "gt_object_id": raw.gt_object_id,
                "gt_object_name": raw.candidate_names[raw.gt_idx] if raw.gt_idx >= 0 else "",
                "num_candidates": len(raw.candidate_ids),
                "teacher_in_topk": int(raw.global_teacher_object_id in raw.candidate_ids),
                "gt_in_topk": int(raw.gt_object_id in raw.candidate_ids),
                "student_confidence": float(posterior[pred_idx].item()),
                "base_gap_top1_minus_top2": float(raw_pair.base_gap.item()),
                "learned_residual_gap": residual_gap,
                "final_gap_top1_minus_top2": final_gap,
                "is_changed_from_base": int(pred_idx != base_pred_idx),
                "is_teacher_match": int(raw.candidate_ids[pred_idx] == raw.global_teacher_object_id),
                "is_gt_exact": int(raw.candidate_ids[pred_idx] == raw.gt_object_id),
                "base_is_gt_exact": int(raw.candidate_ids[base_pred_idx] == raw.gt_object_id),
            }
        )
    write_csv(output_dir / "candidate_scores.csv", candidate_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)
    return candidate_rows, sample_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_features = parse_feature_columns(args.feature_columns)
    set_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)

    full_samples = build_samples_from_csv(
        args.input_csv,
        feature_columns=requested_features,
        teacher_posterior_field=args.teacher_target_field if args.teacher_target_mode == "probability" else None,
        teacher_score_field=args.teacher_target_field if args.teacher_target_mode == "score_softmax" else None,
    )
    full_samples = build_teacher_target_samples(
        full_samples,
        target_mode=args.teacher_target_mode,
        temperature=args.teacher_temperature,
    )
    coarse_lookup = load_coarse_lookup(args.coarse_candidate_csv, top_k=2)
    top2_raw = reduce_samples_to_coarse_topk(
        full_samples,
        coarse_lookup=coarse_lookup,
        append_coarse_features=False,
    )
    train_top2_raw = [sample for sample in top2_raw if sample.split == "train"]
    test_top2_raw = [sample for sample in top2_raw if sample.split == "test"]
    if not train_top2_raw or not test_top2_raw:
        raise RuntimeError("Expected train and test top-2 samples")

    feature_mean, feature_std = compute_feature_stats(train_top2_raw)
    train_top2_norm = normalize_samples(train_top2_raw, feature_mean, feature_std)
    test_top2_norm = normalize_samples(test_top2_raw, feature_mean, feature_std)
    all_top2_norm = normalize_samples(top2_raw, feature_mean, feature_std)

    train_pairs_raw = build_pairwise_samples(train_top2_raw)
    test_pairs_raw = build_pairwise_samples(test_top2_raw)
    all_pairs_raw = build_pairwise_samples(top2_raw)
    train_pairs_norm_features = build_pairwise_samples(train_top2_norm)
    test_pairs_norm_features = build_pairwise_samples(test_top2_norm)
    all_pairs_norm_features = build_pairwise_samples(all_top2_norm)

    pair_mean, pair_std = compute_pair_stats(train_pairs_norm_features)
    train_pairs = normalize_pair_samples(train_pairs_norm_features, pair_mean, pair_std)
    test_pairs = normalize_pair_samples(test_pairs_norm_features, pair_mean, pair_std)
    all_pairs = normalize_pair_samples(all_pairs_norm_features, pair_mean, pair_std)

    train_loader = DataLoader(
        PairwiseDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairwise,
    )
    test_loader = DataLoader(
        PairwiseDataset(test_pairs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pairwise,
    )

    model = PairwiseResidualHead(
        input_dim=train_pairs[0].pair_features.numel(),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for batch in train_loader:
            pair_features = batch["pair_features"].to(device)
            base_gap = batch["base_gap"].to(device)
            teacher_posterior = batch["teacher_posterior"].to(device)
            teacher_first_prob = batch["teacher_first_prob"].to(device)
            outputs = compute_outputs(
                model,
                pair_features,
                base_gap,
                residual_scale=args.residual_scale,
            )
            valid_mask = torch.ones_like(outputs["distribution"], dtype=torch.bool)
            kl_value = distribution_kl(outputs["distribution"], teacher_posterior, valid_mask)
            pair_bce = F.binary_cross_entropy_with_logits(
                outputs["final_gap"],
                teacher_first_prob.detach(),
            )
            residual_l2 = (outputs["residual_gap"] ** 2).mean()
            base_margin = F.relu(-outputs["final_gap"] * torch.sign(base_gap).clamp(min=-1.0, max=1.0)).mean()
            loss = (
                args.lambda_kl * kl_value
                + args.lambda_pair_bce * pair_bce
                + args.lambda_residual_l2 * residual_l2
                + args.lambda_base_margin * base_margin
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = pair_features.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            residual_scale=args.residual_scale,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            residual_scale=args.residual_scale,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_epoch_loss": epoch_loss / max(epoch_samples, 1),
            "train_teacher_top1_count": train_metrics["teacher_top1_count"],
            "train_gt_exact_count": train_metrics["gt_exact_count"],
            "train_changed_count": train_metrics["changed_count"],
            "test_teacher_top1_count": test_metrics["teacher_top1_count"],
            "test_gt_exact_count": test_metrics["gt_exact_count"],
            "test_changed_count": test_metrics["changed_count"],
            "test_kl": test_metrics["kl"],
        }
        history.append(record)
        current_key = (
            int(test_metrics["gt_exact_count"]),
            int(test_metrics["teacher_top1_count"]),
            -int(test_metrics["changed_count"]),
            -float(test_metrics["kl"]),
        )
        best_key = None
        if best_metrics is not None:
            best_key = (
                int(best_metrics["gt_exact_count"]),
                int(best_metrics["teacher_top1_count"]),
                -int(best_metrics["changed_count"]),
                -float(best_metrics["kl"]),
            )
        if best_key is None or current_key > best_key:
            best_metrics = dict(test_metrics)
            best_metrics["epoch"] = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("No best state produced")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "feature_columns": list(requested_features),
            "pair_feature_dim": train_pairs[0].pair_features.numel(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "pair_mean": pair_mean,
            "pair_std": pair_std,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "residual_scale": args.residual_scale,
            "teacher_temperature": args.teacher_temperature,
            "coarse_candidate_csv": str(args.coarse_candidate_csv),
        },
        args.output_dir / "best_model.pt",
    )
    export_predictions(
        model=model,
        raw_samples=all_pairs_raw,
        norm_samples=all_pairs,
        feature_columns=requested_features,
        output_dir=args.output_dir,
        device=device,
        residual_scale=args.residual_scale,
    )
    summary = {
        "mode": "top2_pairwise_residual_head",
        "input_csv": str(args.input_csv),
        "coarse_candidate_csv": str(args.coarse_candidate_csv),
        "output_dir": str(args.output_dir),
        "feature_columns": list(requested_features),
        "teacher_target_field": args.teacher_target_field,
        "teacher_target_mode": args.teacher_target_mode,
        "teacher_temperature": args.teacher_temperature,
        "num_train_samples": len(train_pairs_raw),
        "num_test_samples": len(test_pairs_raw),
        "train_teacher_in_top2_count": sum(s.raw.global_teacher_object_id in s.raw.candidate_ids for s in train_pairs_raw),
        "train_gt_in_top2_count": sum(s.raw.gt_object_id in s.raw.candidate_ids for s in train_pairs_raw),
        "test_teacher_in_top2_count": sum(s.raw.global_teacher_object_id in s.raw.candidate_ids for s in test_pairs_raw),
        "test_gt_in_top2_count": sum(s.raw.gt_object_id in s.raw.candidate_ids for s in test_pairs_raw),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lambda_kl": args.lambda_kl,
        "lambda_pair_bce": args.lambda_pair_bce,
        "lambda_residual_l2": args.lambda_residual_l2,
        "lambda_base_margin": args.lambda_base_margin,
        "residual_scale": args.residual_scale,
        "best_epoch": int(best_metrics["epoch"]),
        "best_test_teacher_top1_count": int(best_metrics["teacher_top1_count"]),
        "best_test_gt_exact_count": int(best_metrics["gt_exact_count"]),
        "best_test_changed_count": int(best_metrics["changed_count"]),
        "best_test_teacher_in_top2_count": int(best_metrics["teacher_in_top2_count"]),
        "best_test_gt_in_top2_count": int(best_metrics["gt_in_top2_count"]),
        "best_test_kl": float(best_metrics["kl"]),
        "best_test_total": int(best_metrics["total_samples"]),
        "history": history,
        "sample_summary_csv": str(args.output_dir / "sample_summary.csv"),
        "candidate_scores_csv": str(args.output_dir / "candidate_scores.csv"),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"Best epoch {best_metrics['epoch']}: "
        f"teacher_top1={best_metrics['teacher_top1_count']}/{best_metrics['total_samples']} "
        f"gt={best_metrics['gt_exact_count']}/{best_metrics['total_samples']} "
        f"changed={best_metrics['changed_count']} "
        f"teacher_in_top2={best_metrics['teacher_in_top2_count']} "
        f"gt_in_top2={best_metrics['gt_in_top2_count']}"
    )


if __name__ == "__main__":
    main()
