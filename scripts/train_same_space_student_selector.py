from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.same_space import (  # noqa: E402
    DEFAULT_SAME_SPACE_FEATURE_COLUMNS,
    DEFAULT_SOURCE_CSV,
    SameSpaceDataset,
    SameSpaceSample,
    SameSpaceSelector,
    build_samples_from_csv,
    collate_same_space_samples,
    compute_feature_stats,
    load_feature_stats,
    normalize_samples,
    parse_feature_columns,
    save_feature_stats,
    masked_softmax,
)


DEFAULT_INPUT_CSV = DEFAULT_SOURCE_CSV
DEFAULT_OUTPUT_DIR = REPO_ROOT / "object_selector" / "same_space_runs" / "student_no_want_align_rank_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a same-space no-want student from teacher posterior targets."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=",".join(DEFAULT_SAME_SPACE_FEATURE_COLUMNS),
        help="Comma-separated numeric columns shared by teacher and student.",
    )
    parser.add_argument(
        "--feature-stats-json",
        type=Path,
        default=None,
        help="Optional teacher feature_stats.json to keep normalization identical.",
    )
    parser.add_argument(
        "--teacher-target-field",
        type=str,
        default="fusion_total",
        help=(
            "Teacher column used to build the distillation target. "
            "Use a raw score such as fusion_total with --teacher-target-mode score_softmax."
        ),
    )
    parser.add_argument(
        "--teacher-target-mode",
        choices=("score_softmax", "probability"),
        default="score_softmax",
        help="How to convert the teacher target field into a candidate distribution.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=2.0,
        help="Temperature used when converting teacher scores/probabilities into a target distribution.",
    )
    parser.add_argument(
        "--distillation-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature applied to student logits inside KL. Use the same "
            "temperature that produced the stored teacher posterior."
        ),
    )
    parser.add_argument(
        "--teacher-selected-field",
        type=str,
        default="",
        help=(
            "Optional teacher selected-object indicator column. "
            "Leave empty to derive teacher top-1 from the distillation target argmax."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument(
        "--lambda-teacher-ce",
        type=float,
        default=0.0,
        help="Cross-entropy weight for the teacher top-1 pseudo-label.",
    )
    parser.add_argument("--lambda-gt-ce", type=float, default=0.0)
    parser.add_argument("--lambda-rank", type=float, default=0.2)
    parser.add_argument(
        "--lambda-pairgap",
        type=float,
        default=0.0,
        help="Gap-matching weight over teacher top-k candidate pairs.",
    )
    parser.add_argument("--rank-topk", type=int, default=3)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-min-gap", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--checkpoint-metric",
        choices=("train_kl", "test_teacher_top1"),
        default="train_kl",
        help=(
            "train_kl avoids selecting a checkpoint on the test set. "
            "test_teacher_top1 is retained only for reproducing legacy runs."
        ),
    )
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


def candidate_cross_entropy(logits: torch.Tensor, valid_mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    usable = target_idx >= 0
    if not usable.any():
        return logits.new_zeros(())
    return F.cross_entropy(masked_logits[usable], target_idx[usable])


def candidate_set_cross_entropy(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability assigned to any accepted target descendant."""
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    log_probs = F.log_softmax(masked_logits, dim=1)
    accepted = target_mask & valid_mask
    usable = accepted.any(dim=1)
    if not usable.any():
        return logits.new_zeros(())
    accepted_log_probs = log_probs.masked_fill(~accepted, -torch.inf)
    return -torch.logsumexp(accepted_log_probs[usable], dim=1).mean()


def distribution_kl(
    student_distribution: torch.Tensor,
    teacher_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.to(dtype=student_distribution.dtype)
    teacher = teacher_distribution * valid
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    student = student_distribution * valid
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    teacher_safe = teacher.clamp_min(1.0e-8)
    student_safe = student.clamp_min(1.0e-8)
    return (teacher * (teacher_safe.log() - student_safe.log())).sum(dim=1).mean()


def teacher_topk_ranking_loss(
    student_logits: torch.Tensor,
    teacher_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
    margin: float,
    min_gap: float,
) -> torch.Tensor:
    total_loss = student_logits.new_zeros(())
    total_pairs = 0

    for batch_index in range(student_logits.shape[0]):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() <= 1:
            continue
        teacher_values = teacher_distribution[batch_index, valid_indices]
        sorted_local = torch.argsort(teacher_values, descending=True)
        ranked = valid_indices[sorted_local[: max(int(topk), 1)]]
        ranked_teacher = teacher_distribution[batch_index, ranked]

        pair_loss = student_logits.new_zeros(())
        pair_count = 0
        for i in range(ranked.numel()):
            for j in range(i + 1, ranked.numel()):
                if float((ranked_teacher[i] - ranked_teacher[j]).item()) < float(min_gap):
                    continue
                hi = ranked[i]
                lo = ranked[j]
                diff = student_logits[batch_index, hi] - student_logits[batch_index, lo]
                pair_loss = pair_loss + F.relu(float(margin) - diff)
                pair_count += 1
        if pair_count > 0:
            total_loss = total_loss + (pair_loss / pair_count)
            total_pairs += 1

    if total_pairs == 0:
        return student_logits.new_zeros(())
    return total_loss / total_pairs


def pairwise_gap_distillation_loss(
    student_logits: torch.Tensor,
    teacher_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    total_loss = student_logits.new_zeros(())
    total_pairs = 0

    for batch_index in range(student_logits.shape[0]):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() <= 1:
            continue
        teacher_values = teacher_distribution[batch_index, valid_indices]
        sorted_local = torch.argsort(teacher_values, descending=True)
        ranked = valid_indices[sorted_local[: max(int(topk), 2)]]
        if ranked.numel() <= 1:
            continue

        teacher_log = teacher_distribution[batch_index, ranked].clamp_min(1.0e-8).log()
        student_ranked = student_logits[batch_index, ranked]
        pair_loss = student_logits.new_zeros(())
        pair_count = 0
        for i in range(ranked.numel()):
            for j in range(i + 1, ranked.numel()):
                teacher_gap = teacher_log[i] - teacher_log[j]
                student_gap = student_ranked[i] - student_ranked[j]
                pair_loss = pair_loss + F.smooth_l1_loss(
                    student_gap,
                    teacher_gap,
                    reduction="sum",
                )
                pair_count += 1
        if pair_count > 0:
            total_loss = total_loss + pair_loss / pair_count
            total_pairs += 1

    if total_pairs == 0:
        return student_logits.new_zeros(())
    return total_loss / total_pairs


def pairwise_ce_distillation_loss(
    student_logits: torch.Tensor,
    teacher_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    total_loss = student_logits.new_zeros(())
    total_pairs = 0
    safe_temperature = max(float(temperature), 1.0e-6)

    for batch_index in range(student_logits.shape[0]):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() <= 1:
            continue
        teacher_values = teacher_distribution[batch_index, valid_indices]
        sorted_local = torch.argsort(teacher_values, descending=True)
        ranked = valid_indices[sorted_local[: max(int(topk), 2)]]
        if ranked.numel() <= 1:
            continue

        teacher_ranked = teacher_distribution[batch_index, ranked].clamp_min(1.0e-8)
        student_ranked = student_logits[batch_index, ranked]
        pair_loss = student_logits.new_zeros(())
        pair_count = 0
        for i in range(ranked.numel()):
            for j in range(i + 1, ranked.numel()):
                teacher_gap = (teacher_ranked[i].log() - teacher_ranked[j].log()) / safe_temperature
                teacher_pair_prob = torch.sigmoid(teacher_gap)
                student_gap = student_ranked[i] - student_ranked[j]
                pair_loss = pair_loss + F.binary_cross_entropy_with_logits(
                    student_gap,
                    teacher_pair_prob.detach(),
                    reduction="sum",
                )
                pair_count += 1
        if pair_count > 0:
            total_loss = total_loss + pair_loss / pair_count
            total_pairs += 1

    if total_pairs == 0:
        return student_logits.new_zeros(())
    return total_loss / total_pairs


def score_softmax_target(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(scores / max(float(temperature), 1.0e-6), dim=0)


def probability_target(values: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = values.clamp_min(0.0)
    total = probs.sum()
    if float(total.item()) <= 0.0:
        probs = torch.full_like(values, 1.0 / max(values.numel(), 1))
    else:
        probs = probs / total
    if abs(float(temperature) - 1.0) <= 1.0e-8:
        return probs
    positive = probs > 0.0
    if not positive.any():
        return probs
    scaled = torch.zeros_like(probs)
    scaled[positive] = torch.softmax(probs[positive].clamp_min(1.0e-8).log() / max(float(temperature), 1.0e-6), dim=0)
    return scaled


def build_teacher_target_samples(
    samples: list[SameSpaceSample],
    *,
    target_mode: str,
    temperature: float,
) -> list[SameSpaceSample]:
    converted: list[SameSpaceSample] = []
    for sample in samples:
        teacher_scores = sample.teacher_scores.clone() if sample.teacher_scores is not None else None
        teacher_posterior = sample.teacher_posterior.clone() if sample.teacher_posterior is not None else None
        if target_mode == "score_softmax":
            if teacher_scores is None:
                raise RuntimeError("score_softmax mode requires teacher scores")
            teacher_posterior = score_softmax_target(teacher_scores, temperature)
        elif target_mode == "probability":
            if teacher_posterior is None:
                if teacher_scores is None:
                    raise RuntimeError("probability mode requires teacher probabilities or scores")
                teacher_posterior = probability_target(teacher_scores, temperature)
            else:
                teacher_posterior = probability_target(teacher_posterior, temperature)
        else:
            raise ValueError(f"Unsupported teacher target mode: {target_mode}")
        teacher_selected_idx = int(torch.argmax(teacher_posterior).item()) if teacher_posterior.numel() > 0 else -1
        converted.append(
            SameSpaceSample(
                sample_id=sample.sample_id,
                split=sample.split,
                split_index=sample.split_index,
                candidate_ids=list(sample.candidate_ids),
                candidate_names=list(sample.candidate_names),
                candidate_categories=list(sample.candidate_categories),
                features=sample.features.clone(),
                gt_idx=sample.gt_idx,
                gt_object_id=sample.gt_object_id,
                gt_indices=list(sample.gt_indices),
                gt_object_ids=list(sample.gt_object_ids),
                teacher_posterior=teacher_posterior,
                teacher_scores=teacher_scores,
                teacher_selected_idx=teacher_selected_idx,
            )
        )
    return converted


@torch.no_grad()
def evaluate(
    model: SameSpaceSelector,
    loader: DataLoader,
    device: torch.device,
    *,
    distillation_temperature: float,
) -> dict[str, float]:
    model.eval()
    total_kl = 0.0
    total_ce = 0.0
    total_samples = 0
    total_gt_samples = 0
    teacher_hits = 0
    gt_hits = 0

    for batch in loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        gt_idx = batch["gt_idx"].to(device)
        gt_mask = batch["gt_mask"].to(device)
        teacher_posterior = batch["teacher_posterior"]
        if teacher_posterior is None:
            raise RuntimeError("Student evaluation requires teacher_posterior in the batch")
        teacher_posterior = teacher_posterior.to(device)
        teacher_selected_idx = batch["teacher_selected_idx"].to(device)

        outputs = model(features, valid_mask)
        temperature = max(float(distillation_temperature), 1.0e-6)
        distill_distribution = masked_softmax(
            outputs["logits"] / temperature,
            valid_mask,
        )
        kl_value = (
            distribution_kl(
                distill_distribution,
                teacher_posterior,
                valid_mask,
            )
            * (temperature**2)
        )
        ce_value = candidate_set_cross_entropy(outputs["logits"], valid_mask, gt_mask)
        pred_idx = outputs["distribution"].argmax(dim=1)

        batch_size = features.shape[0]
        total_kl += float(kl_value.item()) * batch_size
        total_ce += float(ce_value.item()) * batch_size
        total_samples += batch_size
        teacher_hits += int((pred_idx == teacher_selected_idx).sum().item())
        has_gt = gt_mask.any(dim=1)
        total_gt_samples += int(has_gt.sum().item())
        if has_gt.any():
            gt_hits += int(
                gt_mask.gather(1, pred_idx[:, None]).squeeze(1)[has_gt].sum().item()
            )

    return {
        "kl": total_kl / max(total_samples, 1),
        "gt_ce": total_ce / max(total_samples, 1),
        "teacher_top1": teacher_hits / max(total_samples, 1),
        "teacher_top1_count": teacher_hits,
        "gt_accuracy": gt_hits / max(total_gt_samples, 1),
        "gt_exact_count": gt_hits,
        "gt_total": total_gt_samples,
        "total_samples": total_samples,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty csv: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def export_predictions(
    *,
    model: SameSpaceSelector,
    raw_samples: list,
    normalized_samples: list,
    feature_columns: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    candidate_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for raw_sample, norm_sample in zip(raw_samples, normalized_samples, strict=True):
        features = norm_sample.features.unsqueeze(0).to(device)
        valid_mask = torch.ones((1, norm_sample.features.shape[0]), dtype=torch.bool, device=device)
        outputs = model(features, valid_mask)
        student_logits = outputs["logits"][0].detach().cpu()
        student_posterior = outputs["distribution"][0].detach().cpu()
        pred_idx = int(torch.argmax(student_posterior).item())

        teacher_posterior = raw_sample.teacher_posterior
        if teacher_posterior is None:
            raise RuntimeError("Student export requires teacher_posterior")

        teacher_rank_order = torch.argsort(teacher_posterior, descending=True).tolist()
        student_rank_order = torch.argsort(student_posterior, descending=True).tolist()

        for candidate_index, object_id in enumerate(raw_sample.candidate_ids):
            row = {
                "split": raw_sample.split,
                "split_index": raw_sample.split_index,
                "sample_id": raw_sample.sample_id,
                "object_id": object_id,
                "name": raw_sample.candidate_names[candidate_index],
                "category": raw_sample.candidate_categories[candidate_index],
                "teacher_score": (
                    float(raw_sample.teacher_scores[candidate_index].item())
                    if raw_sample.teacher_scores is not None
                    else 0.0
                ),
                "teacher_posterior": float(teacher_posterior[candidate_index].item()),
                "student_score": float(student_logits[candidate_index].item()),
                "student_posterior": float(student_posterior[candidate_index].item()),
                "teacher_rank": int(teacher_rank_order.index(candidate_index) + 1),
                "student_rank": int(student_rank_order.index(candidate_index) + 1),
                "is_teacher_selected": int(candidate_index == raw_sample.teacher_selected_idx),
                "is_student_selected": int(candidate_index == pred_idx),
                "is_gt": int(candidate_index in raw_sample.gt_indices),
            }
            for feature_offset, column in enumerate(feature_columns):
                row[column] = float(raw_sample.features[candidate_index, feature_offset].item())
            candidate_rows.append(row)

        teacher_selected_idx = raw_sample.teacher_selected_idx
        sample_rows.append(
            {
                "split": raw_sample.split,
                "split_index": raw_sample.split_index,
                "sample_id": raw_sample.sample_id,
                "selected_object_id": raw_sample.candidate_ids[pred_idx],
                "selected_object_name": raw_sample.candidate_names[pred_idx],
                "selected_label": raw_sample.candidate_names[pred_idx],
                "teacher_selected_object_id": (
                    raw_sample.candidate_ids[teacher_selected_idx] if teacher_selected_idx >= 0 else ""
                ),
                "gt_object_id": raw_sample.gt_object_id,
                "gt_object_ids": json.dumps(raw_sample.gt_object_ids),
                "gt_object_name": (
                    raw_sample.candidate_names[raw_sample.gt_idx]
                    if raw_sample.gt_idx >= 0
                    else ""
                ),
                "num_candidates": len(raw_sample.candidate_ids),
                "student_confidence": float(student_posterior[pred_idx].item()),
                "teacher_confidence": (
                    float(teacher_posterior[teacher_selected_idx].item()) if teacher_selected_idx >= 0 else 0.0
                ),
                "is_teacher_match": int(pred_idx == teacher_selected_idx),
                "is_gt_exact": int(pred_idx in raw_sample.gt_indices),
            }
        )

    write_csv(output_dir / "candidate_scores.csv", candidate_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)
    return candidate_rows, sample_rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_features = parse_feature_columns(args.feature_columns)
    set_seed(args.seed)
    device = choose_device(args.device)

    raw_samples = build_samples_from_csv(
        args.input_csv,
        feature_columns=requested_features,
        teacher_posterior_field=(
            args.teacher_target_field if args.teacher_target_mode == "probability" else None
        ),
        teacher_score_field=(
            args.teacher_target_field if args.teacher_target_mode == "score_softmax" else None
        ),
        teacher_selected_field=args.teacher_selected_field or None,
    )
    raw_samples = build_teacher_target_samples(
        raw_samples,
        target_mode=args.teacher_target_mode,
        temperature=args.teacher_temperature,
    )
    train_raw = [sample for sample in raw_samples if sample.split == "train"]
    test_raw = [sample for sample in raw_samples if sample.split == "test"]
    if not train_raw or not test_raw:
        raise RuntimeError("Expected both train and test samples in the student input csv")

    if args.feature_stats_json:
        feature_columns, mean, std = load_feature_stats(args.feature_stats_json)
        if feature_columns != requested_features:
            raise ValueError(
                f"Requested feature columns {requested_features} do not match stats file {feature_columns}"
            )
    else:
        feature_columns = requested_features
        mean, std = compute_feature_stats(train_raw)
    save_feature_stats(
        output_dir / "feature_stats.json",
        feature_columns=feature_columns,
        mean=mean,
        std=std,
    )

    normalized_train = normalize_samples(train_raw, mean, std)
    normalized_test = normalize_samples(test_raw, mean, std)

    train_loader = DataLoader(
        SameSpaceDataset(normalized_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_same_space_samples,
    )
    test_loader = DataLoader(
        SameSpaceDataset(normalized_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_same_space_samples,
    )

    model = SameSpaceSelector(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_selection_key: tuple[float, ...] | None = None
    best_gt_metrics: dict[str, float] | None = None
    best_gt_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            gt_idx = batch["gt_idx"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            teacher_posterior = batch["teacher_posterior"]
            if teacher_posterior is None:
                raise RuntimeError("Student training requires teacher_posterior in the batch")
            teacher_posterior = teacher_posterior.to(device)
            teacher_selected_idx = batch["teacher_selected_idx"].to(device)

            outputs = model(features, valid_mask)
            distillation_temperature = max(
                float(args.distillation_temperature), 1.0e-6
            )
            distill_distribution = masked_softmax(
                outputs["logits"] / distillation_temperature,
                valid_mask,
            )
            kl_value = (
                distribution_kl(
                    distill_distribution,
                    teacher_posterior,
                    valid_mask,
                )
                * (distillation_temperature**2)
            )
            rank_value = teacher_topk_ranking_loss(
                outputs["logits"],
                teacher_posterior,
                valid_mask,
                topk=args.rank_topk,
                margin=args.rank_margin,
                min_gap=args.rank_min_gap,
            )
            pairgap_value = pairwise_gap_distillation_loss(
                outputs["logits"],
                teacher_posterior,
                valid_mask,
                topk=args.rank_topk,
            )
            teacher_ce = candidate_cross_entropy(outputs["logits"], valid_mask, teacher_selected_idx)
            gt_ce = candidate_set_cross_entropy(
                outputs["logits"], valid_mask, gt_mask
            )
            loss = (
                args.lambda_kl * kl_value
                + args.lambda_teacher_ce * teacher_ce
                + args.lambda_rank * rank_value
                + args.lambda_pairgap * pairgap_value
                + args.lambda_gt_ce * gt_ce
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
            distillation_temperature=args.distillation_temperature,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            distillation_temperature=args.distillation_temperature,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_epoch_loss": epoch_loss / max(epoch_samples, 1),
            "train_teacher_top1": train_metrics["teacher_top1"],
            "test_teacher_top1": test_metrics["teacher_top1"],
            "test_teacher_top1_count": test_metrics["teacher_top1_count"],
            "test_gt_accuracy": test_metrics["gt_accuracy"],
            "test_gt_exact_count": test_metrics["gt_exact_count"],
            "test_kl": test_metrics["kl"],
        }
        history.append(record)

        if args.checkpoint_metric == "train_kl":
            current_key = (
                -float(train_metrics["kl"]),
                float(train_metrics["teacher_top1_count"]),
            )
        else:
            current_key = (
                float(test_metrics["teacher_top1_count"]),
                float(test_metrics["gt_exact_count"]),
                -float(test_metrics["kl"]),
            )
        is_better = best_selection_key is None or current_key > best_selection_key

        if is_better:
            best_selection_key = current_key
            best_metrics = dict(test_metrics)
            best_metrics["epoch"] = epoch
            best_metrics["train_kl_at_selection"] = float(train_metrics["kl"])
            best_metrics["train_teacher_top1_at_selection"] = float(
                train_metrics["teacher_top1"]
            )
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        is_better_gt = False
        if best_gt_metrics is None:
            is_better_gt = True
        else:
            current_gt_key = (
                int(test_metrics["gt_exact_count"]),
                int(test_metrics["teacher_top1_count"]),
                -float(test_metrics["kl"]),
            )
            best_gt_key = (
                int(best_gt_metrics["gt_exact_count"]),
                int(best_gt_metrics["teacher_top1_count"]),
                -float(best_gt_metrics["kl"]),
            )
            is_better_gt = current_gt_key > best_gt_key

        if is_better_gt:
            best_gt_metrics = dict(test_metrics)
            best_gt_metrics["epoch"] = epoch
            best_gt_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None or best_gt_state is None or best_gt_metrics is None:
        raise RuntimeError("Student training did not produce best checkpoints")

    checkpoint = {
        "model_state": best_state,
        "feature_columns": list(feature_columns),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lambda_kl": args.lambda_kl,
        "lambda_teacher_ce": args.lambda_teacher_ce,
        "lambda_rank": args.lambda_rank,
        "lambda_pairgap": args.lambda_pairgap,
        "lambda_gt_ce": args.lambda_gt_ce,
        "rank_topk": args.rank_topk,
        "rank_margin": args.rank_margin,
        "rank_min_gap": args.rank_min_gap,
        "feature_stats_json": str(args.feature_stats_json) if args.feature_stats_json else "",
        "saved_feature_stats_json": str(output_dir / "feature_stats.json"),
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    torch.save(checkpoint, output_dir / "best_model_teacher.pt")

    checkpoint_gt = dict(checkpoint)
    checkpoint_gt["model_state"] = best_gt_state
    torch.save(checkpoint_gt, output_dir / "best_model_gt.pt")

    model.load_state_dict(best_state)
    normalized_all = normalize_samples(raw_samples, mean, std)
    _, sample_rows = export_predictions(
        model=model,
        raw_samples=raw_samples,
        normalized_samples=normalized_all,
        feature_columns=feature_columns,
        device=device,
        output_dir=output_dir,
    )

    model.load_state_dict(best_gt_state)
    gt_output_dir = output_dir / "best_gt_export"
    _, sample_rows_gt = export_predictions(
        model=model,
        raw_samples=raw_samples,
        normalized_samples=normalized_all,
        feature_columns=feature_columns,
        device=device,
        output_dir=gt_output_dir,
    )

    summary = {
        "mode": "same_space_student_no_want_align_rank",
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
        "feature_columns": list(feature_columns),
        "teacher_target_field": args.teacher_target_field,
        "teacher_target_mode": args.teacher_target_mode,
        "teacher_temperature": args.teacher_temperature,
        "distillation_temperature": args.distillation_temperature,
        "checkpoint_metric": args.checkpoint_metric,
        "num_train_samples": len(train_raw),
        "num_test_samples": len(test_raw),
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
        "lambda_pairgap": args.lambda_pairgap,
        "lambda_gt_ce": args.lambda_gt_ce,
        "rank_topk": args.rank_topk,
        "rank_margin": args.rank_margin,
        "rank_min_gap": args.rank_min_gap,
        "feature_stats_json": str(args.feature_stats_json) if args.feature_stats_json else "",
        "best_epoch": int(best_metrics["epoch"]),
        "best_test_teacher_top1": float(best_metrics["teacher_top1"]),
        "best_test_teacher_top1_count": int(best_metrics["teacher_top1_count"]),
        "best_test_gt_accuracy": float(best_metrics["gt_accuracy"]),
        "best_test_gt_exact_count": int(best_metrics["gt_exact_count"]),
        "best_test_gt_total": int(best_metrics["gt_total"]),
        "best_test_kl": float(best_metrics["kl"]),
        "best_test_total": int(best_metrics["total_samples"]),
        "best_gt_epoch": int(best_gt_metrics["epoch"]),
        "best_gt_test_teacher_top1": float(best_gt_metrics["teacher_top1"]),
        "best_gt_test_teacher_top1_count": int(best_gt_metrics["teacher_top1_count"]),
        "best_gt_test_gt_accuracy": float(best_gt_metrics["gt_accuracy"]),
        "best_gt_test_gt_exact_count": int(best_gt_metrics["gt_exact_count"]),
        "best_gt_test_kl": float(best_gt_metrics["kl"]),
        "history": history,
        "sample_summary_csv": str(output_dir / "sample_summary.csv"),
        "candidate_scores_csv": str(output_dir / "candidate_scores.csv"),
        "best_gt_sample_summary_csv": str(gt_output_dir / "sample_summary.csv"),
        "best_gt_candidate_scores_csv": str(gt_output_dir / "candidate_scores.csv"),
    }
    write_json(output_dir / "summary.json", summary)

    test_selected = [row for row in sample_rows if row["split"] == "test"]
    print(
        f"Best epoch {best_metrics['epoch']}: "
        f"teacher_top1={best_metrics['teacher_top1_count']}/{best_metrics['total_samples']} "
        f"({best_metrics['teacher_top1']:.4f}), "
        f"gt={best_metrics['gt_exact_count']}/{best_metrics['gt_total']} "
        f"({best_metrics['gt_accuracy']:.4f})"
    )
    print(f"Exported {len(test_selected)} test predictions to {output_dir}")


if __name__ == "__main__":
    main()
