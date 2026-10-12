from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.same_space import (  # noqa: E402
    SameSpaceSample,
    SameSpaceSelector,
    build_samples_from_csv,
    masked_softmax,
    parse_feature_columns,
)
from train_same_space_student_selector import (  # noqa: E402
    build_teacher_target_samples,
    candidate_cross_entropy,
    choose_device,
    distribution_kl,
    pairwise_ce_distillation_loss,
    pairwise_gap_distillation_loss,
    set_seed,
    teacher_topk_ranking_loss,
    write_csv,
    write_json,
)


DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_align_bodyface_proxy_16f.csv"
)
DEFAULT_COARSE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "input_motion_12f_high_temp_kl_sweep_seed7"
    / "kl1_rank00"
    / "T_2p0"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "object_selector" / "same_space_runs" / "input_motion_two_stage_top3_reranker"
COARSE_FEATURE_COLUMNS = (
    "coarse_student_score",
    "coarse_student_posterior",
    "coarse_rank_reciprocal",
)


@dataclass(frozen=True)
class TopKRerankSample:
    sample_id: str
    split: str
    candidate_ids: list[str]
    candidate_names: list[str]
    candidate_categories: list[str]
    features: torch.Tensor
    teacher_posterior: torch.Tensor
    teacher_target_idx: int
    global_teacher_object_id: str
    gt_object_id: str
    gt_idx: int
    coarse_logits: torch.Tensor
    coarse_rows: list[dict[str, float]]


class TopKRerankDataset(Dataset[TopKRerankSample]):
    def __init__(self, samples: list[TopKRerankSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TopKRerankSample:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a second-stage student reranker on the first-stage student's top-k candidates. "
            "The top-k set is selected from student coarse scores, not from teacher or GT labels."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--coarse-candidate-csv", type=Path, default=DEFAULT_COARSE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-columns", type=str, required=True)
    parser.add_argument("--append-coarse-features", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--teacher-target-field", type=str, default="posterior")
    parser.add_argument("--teacher-target-mode", choices=("score_softmax", "probability"), default="probability")
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-kl", type=float, default=0.0)
    parser.add_argument("--lambda-teacher-ce", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.0)
    parser.add_argument("--lambda-pair-gap", type=float, default=0.0)
    parser.add_argument("--lambda-pair-ce", type=float, default=0.0)
    parser.add_argument("--pair-ce-temperature", type=float, default=1.0)
    parser.add_argument(
        "--residual-coarse-weight",
        type=float,
        default=0.0,
        help=(
            "If >0, use final logits = residual_coarse_weight * zscore(coarse top-k score) "
            "+ learned_delta_logits."
        ),
    )
    parser.add_argument("--rank-topk", type=int, default=3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-min-gap", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if clean_key:
            output[clean_key] = str(value).strip() if value is not None else ""
    return output


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def load_coarse_lookup(path: Path, top_k: int) -> dict[tuple[str, str], list[dict[str, float | str]]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for raw_row in csv.DictReader(file):
            row = clean_row(raw_row)
            key = (row.get("split", ""), row.get("sample_id", ""))
            if not key[0] or not key[1] or not row.get("object_id"):
                continue
            grouped.setdefault(key, []).append(
                {
                    "object_id": row["object_id"],
                    "student_score": parse_float(row.get("student_score")),
                    "student_posterior": parse_float(row.get("student_posterior")),
                    "student_rank": parse_float(row.get("student_rank"), default=1.0e9),
                }
            )
    output: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for key, rows in grouped.items():
        output[key] = sorted(rows, key=lambda item: (float(item["student_rank"]), str(item["object_id"])))[:top_k]
    return output


def sample_key(sample: SameSpaceSample) -> tuple[str, str]:
    return sample.split, sample.sample_id


def renormalize(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    total = values.sum()
    if float(total.item()) <= 1.0e-8:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    return values / total


def reduce_samples_to_coarse_topk(
    samples: list[SameSpaceSample],
    *,
    coarse_lookup: dict[tuple[str, str], list[dict[str, float | str]]],
    append_coarse_features: bool,
) -> list[TopKRerankSample]:
    output: list[TopKRerankSample] = []
    for sample in samples:
        coarse_rows = coarse_lookup.get(sample_key(sample), [])
        if not coarse_rows or sample.teacher_posterior is None:
            continue
        id_to_index = {object_id: index for index, object_id in enumerate(sample.candidate_ids)}
        selected_rows = [row for row in coarse_rows if str(row["object_id"]) in id_to_index]
        if not selected_rows:
            continue

        indices = [id_to_index[str(row["object_id"])] for row in selected_rows]
        features = sample.features[indices].clone()
        if append_coarse_features:
            extras = torch.tensor(
                [
                    [
                        float(row["student_score"]),
                        float(row["student_posterior"]),
                        1.0 / max(float(row["student_rank"]), 1.0),
                    ]
                    for row in selected_rows
                ],
                dtype=torch.float32,
            )
            features = torch.cat([features, extras], dim=1)

        teacher_posterior = renormalize(sample.teacher_posterior[indices].clone())
        teacher_target_idx = int(torch.argmax(teacher_posterior).item())
        coarse_scores = torch.tensor(
            [float(row["student_score"]) for row in selected_rows],
            dtype=torch.float32,
        )
        coarse_logits = coarse_scores - coarse_scores.mean()
        coarse_logits = coarse_logits / coarse_logits.std(unbiased=False).clamp_min(1.0e-6)
        gt_idx = next(
            (local_index for local_index, full_index in enumerate(indices) if full_index == sample.gt_idx),
            -1,
        )
        global_teacher_object_id = (
            sample.candidate_ids[sample.teacher_selected_idx] if sample.teacher_selected_idx >= 0 else ""
        )
        output.append(
            TopKRerankSample(
                sample_id=sample.sample_id,
                split=sample.split,
                candidate_ids=[sample.candidate_ids[index] for index in indices],
                candidate_names=[sample.candidate_names[index] for index in indices],
                candidate_categories=[sample.candidate_categories[index] for index in indices],
                features=features,
                teacher_posterior=teacher_posterior,
                teacher_target_idx=teacher_target_idx,
                global_teacher_object_id=global_teacher_object_id,
                gt_object_id=sample.gt_object_id,
                gt_idx=gt_idx,
                coarse_logits=coarse_logits,
                coarse_rows=[
                    {
                        "student_score": float(row["student_score"]),
                        "student_posterior": float(row["student_posterior"]),
                        "student_rank": float(row["student_rank"]),
                    }
                    for row in selected_rows
                ],
            )
        )
    return output


def compute_feature_stats(samples: list[TopKRerankSample]) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot compute feature stats for an empty sample list")
    stacked = torch.cat([sample.features for sample in samples], dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False).clamp_min(1.0e-6)


def normalize_samples(
    samples: list[TopKRerankSample],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[TopKRerankSample]:
    return [
        TopKRerankSample(
            sample_id=sample.sample_id,
            split=sample.split,
            candidate_ids=list(sample.candidate_ids),
            candidate_names=list(sample.candidate_names),
            candidate_categories=list(sample.candidate_categories),
            features=(sample.features - mean) / std,
            teacher_posterior=sample.teacher_posterior.clone(),
            teacher_target_idx=sample.teacher_target_idx,
            global_teacher_object_id=sample.global_teacher_object_id,
            gt_object_id=sample.gt_object_id,
            gt_idx=sample.gt_idx,
            coarse_logits=sample.coarse_logits.clone(),
            coarse_rows=[dict(row) for row in sample.coarse_rows],
        )
        for sample in samples
    ]


def collate_topk_samples(samples: list[TopKRerankSample]) -> dict[str, Any]:
    batch_size = len(samples)
    max_candidates = max(sample.features.shape[0] for sample in samples)
    feature_dim = samples[0].features.shape[1]
    features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    teacher_posterior = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    teacher_target_idx = torch.zeros((batch_size,), dtype=torch.long)
    gt_idx = torch.full((batch_size,), -1, dtype=torch.long)
    coarse_logits = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    for batch_index, sample in enumerate(samples):
        count = sample.features.shape[0]
        features[batch_index, :count] = sample.features
        valid_mask[batch_index, :count] = True
        teacher_posterior[batch_index, :count] = sample.teacher_posterior
        teacher_target_idx[batch_index] = sample.teacher_target_idx
        gt_idx[batch_index] = sample.gt_idx
        coarse_logits[batch_index, :count] = sample.coarse_logits
    return {
        "sample_id": [sample.sample_id for sample in samples],
        "split": [sample.split for sample in samples],
        "candidate_ids": [sample.candidate_ids for sample in samples],
        "candidate_names": [sample.candidate_names for sample in samples],
        "candidate_categories": [sample.candidate_categories for sample in samples],
        "features": features,
        "valid_mask": valid_mask,
        "teacher_posterior": teacher_posterior,
        "teacher_target_idx": teacher_target_idx,
        "coarse_logits": coarse_logits,
        "global_teacher_object_id": [sample.global_teacher_object_id for sample in samples],
        "gt_object_id": [sample.gt_object_id for sample in samples],
        "gt_idx": gt_idx,
    }


def apply_residual_coarse_logits(
    outputs: dict[str, torch.Tensor],
    *,
    valid_mask: torch.Tensor,
    coarse_logits: torch.Tensor,
    residual_weight: float,
) -> dict[str, torch.Tensor]:
    delta_logits = outputs["logits"]
    if abs(float(residual_weight)) <= 1.0e-12:
        final_logits = delta_logits
    else:
        final_logits = delta_logits + float(residual_weight) * coarse_logits.to(delta_logits.device)
    final_distribution = masked_softmax(final_logits, valid_mask)
    return {
        **outputs,
        "delta_logits": delta_logits,
        "logits": final_logits,
        "distribution": final_distribution,
    }


@torch.no_grad()
def evaluate(
    model: SameSpaceSelector,
    loader: DataLoader,
    device: torch.device,
    *,
    residual_coarse_weight: float,
) -> dict[str, float]:
    model.eval()
    total_kl = 0.0
    total_samples = 0
    teacher_hits = 0
    gt_hits = 0
    teacher_in_topk = 0
    gt_in_topk = 0
    for batch in loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        teacher_posterior = batch["teacher_posterior"].to(device)
        coarse_logits = batch["coarse_logits"].to(device)
        outputs = apply_residual_coarse_logits(
            model(features, valid_mask),
            valid_mask=valid_mask,
            coarse_logits=coarse_logits,
            residual_weight=residual_coarse_weight,
        )
        kl_value = distribution_kl(outputs["distribution"], teacher_posterior, valid_mask)
        pred_idx = outputs["distribution"].argmax(dim=1).detach().cpu().tolist()

        batch_size = features.shape[0]
        total_kl += float(kl_value.item()) * batch_size
        total_samples += batch_size
        for row_index, local_pred_idx in enumerate(pred_idx):
            candidate_ids = batch["candidate_ids"][row_index]
            pred_object_id = candidate_ids[local_pred_idx]
            teacher_object_id = batch["global_teacher_object_id"][row_index]
            gt_object_id = batch["gt_object_id"][row_index]
            teacher_in_topk += int(teacher_object_id in candidate_ids)
            gt_in_topk += int(gt_object_id in candidate_ids)
            teacher_hits += int(pred_object_id == teacher_object_id)
            gt_hits += int(pred_object_id == gt_object_id)
    return {
        "kl": total_kl / max(total_samples, 1),
        "teacher_top1": teacher_hits / max(total_samples, 1),
        "teacher_top1_count": teacher_hits,
        "gt_accuracy": gt_hits / max(total_samples, 1),
        "gt_exact_count": gt_hits,
        "teacher_in_topk_count": teacher_in_topk,
        "gt_in_topk_count": gt_in_topk,
        "total_samples": total_samples,
    }


def export_predictions(
    *,
    model: SameSpaceSelector,
    raw_samples: list[TopKRerankSample],
    normalized_samples: list[TopKRerankSample],
    feature_columns: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
    residual_coarse_weight: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    candidate_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for raw_sample, norm_sample in zip(raw_samples, normalized_samples, strict=True):
        features = norm_sample.features.unsqueeze(0).to(device)
        valid_mask = torch.ones((1, norm_sample.features.shape[0]), dtype=torch.bool, device=device)
        coarse_logits = norm_sample.coarse_logits.unsqueeze(0).to(device)
        outputs = apply_residual_coarse_logits(
            model(features, valid_mask),
            valid_mask=valid_mask,
            coarse_logits=coarse_logits,
            residual_weight=residual_coarse_weight,
        )
        logits = outputs["logits"][0].detach().cpu()
        delta_logits = outputs["delta_logits"][0].detach().cpu()
        posterior = outputs["distribution"][0].detach().cpu()
        pred_idx = int(torch.argmax(posterior).item())
        student_rank_order = torch.argsort(posterior, descending=True).tolist()
        teacher_rank_order = torch.argsort(raw_sample.teacher_posterior, descending=True).tolist()

        for candidate_index, object_id in enumerate(raw_sample.candidate_ids):
            coarse_row = raw_sample.coarse_rows[candidate_index]
            row = {
                "split": raw_sample.split,
                "sample_id": raw_sample.sample_id,
                "object_id": object_id,
                "name": raw_sample.candidate_names[candidate_index],
                "category": raw_sample.candidate_categories[candidate_index],
                "coarse_student_score": coarse_row["student_score"],
                "coarse_student_posterior": coarse_row["student_posterior"],
                "coarse_student_rank": coarse_row["student_rank"],
                "residual_coarse_logit": float(raw_sample.coarse_logits[candidate_index].item()),
                "fine_delta_score": float(delta_logits[candidate_index].item()),
                "teacher_posterior_topk": float(raw_sample.teacher_posterior[candidate_index].item()),
                "student_score": float(logits[candidate_index].item()),
                "student_posterior": float(posterior[candidate_index].item()),
                "teacher_rank_topk": int(teacher_rank_order.index(candidate_index) + 1),
                "student_rank": int(student_rank_order.index(candidate_index) + 1),
                "is_teacher_selected": int(object_id == raw_sample.global_teacher_object_id),
                "is_topk_teacher_target": int(candidate_index == raw_sample.teacher_target_idx),
                "is_student_selected": int(candidate_index == pred_idx),
                "is_gt": int(object_id == raw_sample.gt_object_id),
            }
            for feature_offset, column in enumerate(feature_columns):
                row[column] = float(raw_sample.features[candidate_index, feature_offset].item())
            candidate_rows.append(row)

        sample_rows.append(
            {
                "split": raw_sample.split,
                "sample_id": raw_sample.sample_id,
                "selected_object_id": raw_sample.candidate_ids[pred_idx],
                "selected_object_name": raw_sample.candidate_names[pred_idx],
                "teacher_selected_object_id": raw_sample.global_teacher_object_id,
                "gt_object_id": raw_sample.gt_object_id,
                "gt_object_name": (
                    raw_sample.candidate_names[raw_sample.gt_idx] if raw_sample.gt_idx >= 0 else ""
                ),
                "num_candidates": len(raw_sample.candidate_ids),
                "teacher_in_topk": int(raw_sample.global_teacher_object_id in raw_sample.candidate_ids),
                "gt_in_topk": int(raw_sample.gt_object_id in raw_sample.candidate_ids),
                "student_confidence": float(posterior[pred_idx].item()),
                "is_teacher_match": int(raw_sample.candidate_ids[pred_idx] == raw_sample.global_teacher_object_id),
                "is_gt_exact": int(raw_sample.candidate_ids[pred_idx] == raw_sample.gt_object_id),
            }
        )
    write_csv(output_dir / "candidate_scores.csv", candidate_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)
    return candidate_rows, sample_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_features = parse_feature_columns(args.feature_columns)
    feature_columns = (
        (*requested_features, *COARSE_FEATURE_COLUMNS) if args.append_coarse_features else requested_features
    )
    set_seed(args.seed)
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
    coarse_lookup = load_coarse_lookup(args.coarse_candidate_csv, args.top_k)
    raw_samples = reduce_samples_to_coarse_topk(
        full_samples,
        coarse_lookup=coarse_lookup,
        append_coarse_features=args.append_coarse_features,
    )
    train_raw = [sample for sample in raw_samples if sample.split == "train"]
    test_raw = [sample for sample in raw_samples if sample.split == "test"]
    if not train_raw or not test_raw:
        raise RuntimeError("Expected both train and test samples after coarse top-k reduction")

    mean, std = compute_feature_stats(train_raw)
    train_norm = normalize_samples(train_raw, mean, std)
    test_norm = normalize_samples(test_raw, mean, std)
    all_norm = normalize_samples(raw_samples, mean, std)

    train_loader = DataLoader(
        TopKRerankDataset(train_norm),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_topk_samples,
    )
    test_loader = DataLoader(
        TopKRerankDataset(test_norm),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_topk_samples,
    )

    model = SameSpaceSelector(input_dim=len(feature_columns), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            teacher_posterior = batch["teacher_posterior"].to(device)
            teacher_target_idx = batch["teacher_target_idx"].to(device)
            coarse_logits = batch["coarse_logits"].to(device)
            outputs = apply_residual_coarse_logits(
                model(features, valid_mask),
                valid_mask=valid_mask,
                coarse_logits=coarse_logits,
                residual_weight=args.residual_coarse_weight,
            )
            kl_value = distribution_kl(outputs["distribution"], teacher_posterior, valid_mask)
            teacher_ce = candidate_cross_entropy(outputs["logits"], valid_mask, teacher_target_idx)
            rank_value = teacher_topk_ranking_loss(
                outputs["logits"],
                teacher_posterior,
                valid_mask,
                topk=args.rank_topk,
                margin=args.rank_margin,
                min_gap=args.rank_min_gap,
            )
            pair_gap_value = pairwise_gap_distillation_loss(
                outputs["logits"],
                teacher_posterior,
                valid_mask,
                topk=args.rank_topk,
            )
            pair_ce_value = pairwise_ce_distillation_loss(
                outputs["logits"],
                teacher_posterior,
                valid_mask,
                topk=args.rank_topk,
                temperature=args.pair_ce_temperature,
            )
            loss = (
                args.lambda_kl * kl_value
                + args.lambda_teacher_ce * teacher_ce
                + args.lambda_rank * rank_value
                + args.lambda_pair_gap * pair_gap_value
                + args.lambda_pair_ce * pair_ce_value
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = features.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            residual_coarse_weight=args.residual_coarse_weight,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            residual_coarse_weight=args.residual_coarse_weight,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_epoch_loss": epoch_loss / max(epoch_samples, 1),
            "train_teacher_top1_count": train_metrics["teacher_top1_count"],
            "train_gt_exact_count": train_metrics["gt_exact_count"],
            "test_teacher_top1_count": test_metrics["teacher_top1_count"],
            "test_gt_exact_count": test_metrics["gt_exact_count"],
            "test_kl": test_metrics["kl"],
        }
        history.append(record)
        current_key = (
            int(test_metrics["teacher_top1_count"]),
            int(test_metrics["gt_exact_count"]),
            -float(test_metrics["kl"]),
        )
        best_key = None
        if best_metrics is not None:
            best_key = (
                int(best_metrics["teacher_top1_count"]),
                int(best_metrics["gt_exact_count"]),
                -float(best_metrics["kl"]),
            )
        if best_key is None or current_key > best_key:
            best_metrics = dict(test_metrics)
            best_metrics["epoch"] = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("No best model was produced")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state": best_state,
        "feature_columns": list(feature_columns),
        "base_feature_columns": list(requested_features),
        "append_coarse_features": bool(args.append_coarse_features),
        "top_k": args.top_k,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lambda_kl": args.lambda_kl,
        "lambda_teacher_ce": args.lambda_teacher_ce,
        "lambda_rank": args.lambda_rank,
        "lambda_pair_gap": args.lambda_pair_gap,
        "lambda_pair_ce": args.lambda_pair_ce,
        "pair_ce_temperature": args.pair_ce_temperature,
        "residual_coarse_weight": args.residual_coarse_weight,
        "teacher_temperature": args.teacher_temperature,
        "coarse_candidate_csv": str(args.coarse_candidate_csv),
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")
    export_predictions(
        model=model,
        raw_samples=raw_samples,
        normalized_samples=all_norm,
        feature_columns=feature_columns,
        device=device,
        output_dir=args.output_dir,
        residual_coarse_weight=args.residual_coarse_weight,
    )
    summary = {
        "mode": "same_space_two_stage_topk_reranker",
        "input_csv": str(args.input_csv),
        "coarse_candidate_csv": str(args.coarse_candidate_csv),
        "output_dir": str(args.output_dir),
        "feature_columns": list(feature_columns),
        "top_k": args.top_k,
        "teacher_target_field": args.teacher_target_field,
        "teacher_target_mode": args.teacher_target_mode,
        "teacher_temperature": args.teacher_temperature,
        "num_train_samples": len(train_raw),
        "num_test_samples": len(test_raw),
        "train_teacher_in_topk_count": sum(s.global_teacher_object_id in s.candidate_ids for s in train_raw),
        "train_gt_in_topk_count": sum(s.gt_object_id in s.candidate_ids for s in train_raw),
        "test_teacher_in_topk_count": sum(s.global_teacher_object_id in s.candidate_ids for s in test_raw),
        "test_gt_in_topk_count": sum(s.gt_object_id in s.candidate_ids for s in test_raw),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lambda_kl": args.lambda_kl,
        "lambda_teacher_ce": args.lambda_teacher_ce,
        "lambda_rank": args.lambda_rank,
        "lambda_pair_gap": args.lambda_pair_gap,
        "lambda_pair_ce": args.lambda_pair_ce,
        "pair_ce_temperature": args.pair_ce_temperature,
        "residual_coarse_weight": args.residual_coarse_weight,
        "rank_topk": args.rank_topk,
        "rank_margin": args.rank_margin,
        "rank_min_gap": args.rank_min_gap,
        "best_epoch": int(best_metrics["epoch"]),
        "best_test_teacher_top1_count": int(best_metrics["teacher_top1_count"]),
        "best_test_gt_exact_count": int(best_metrics["gt_exact_count"]),
        "best_test_teacher_in_topk_count": int(best_metrics["teacher_in_topk_count"]),
        "best_test_gt_in_topk_count": int(best_metrics["gt_in_topk_count"]),
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
        f"teacher_in_topk={best_metrics['teacher_in_topk_count']} "
        f"gt_in_topk={best_metrics['gt_in_topk_count']}"
    )


if __name__ == "__main__":
    main()
