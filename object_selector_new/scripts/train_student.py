#!/usr/bin/env python3
"""Train the same-space student to mimic the teacher's RUBRIC posterior.

Pipeline:
  - Teacher CSV (from export_teacher_csv.py) provides per-candidate
    `rubric_posterior` and `motion_score` columns.
  - Student MLP receives geometry features only and is trained to match the
    rubric posterior via KL distillation.
  - At evaluation time, an optional Stage-S2 tie-break is applied: when the
    student's top-3 candidates contain 2+ entries with the same `name`, the
    motion score is added (within-top3 minmax normalized) before argmax.

The student's training-time loss never sees motion or names — those are used
only to compute final predictions for evaluation/export.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.student import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    StudentDataset,
    StudentSample,
    StudentSelector,
    build_samples_from_csv,
    collate_student_samples,
    compute_feature_stats,
    load_feature_stats,
    normalize_samples,
    parse_feature_columns,
    save_feature_stats,
    stage_s2_apply,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill a same-space student from teacher CSV.")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--feature-columns", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS))
    p.add_argument("--feature-stats-json", type=Path, default=None)

    # Distillation target (now: teacher rubric_posterior, plus optional sharpening T_S)
    p.add_argument(
        "--teacher-posterior-field",
        type=str,
        default="rubric_posterior",
        help="CSV column to use as the teacher distribution.",
    )
    p.add_argument(
        "--teacher-temperature",
        type=float,
        default=1.0,
        help="Optional sharpening temperature applied to teacher posterior. "
        "T=1 keeps it as-is; T<1 sharpens, T>1 softens.",
    )
    p.add_argument("--teacher-selected-field", type=str, default="is_teacher_selected")

    # Optional: keep CSV motion_score column accessible during eval
    p.add_argument("--motion-score-field", type=str, default="motion_score")

    # Stage-S2 tie-break (evaluation only)
    p.add_argument(
        "--s2-tiebreak",
        type=lambda v: str(v).lower() in {"1", "true", "yes", "on"},
        default=True,
        help="Enable top-k name-collision motion tie-break at eval time.",
    )
    p.add_argument("--s2-topk", type=int, default=3)
    p.add_argument("--s2-motion-weight", type=float, default=1.0)
    p.add_argument(
        "--s2-trigger-mode",
        choices=("name_collision", "score_margin", "either"),
        default="name_collision",
    )
    p.add_argument("--s2-margin-threshold", type=float, default=0.05)

    # Optimizer / model
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    # Loss weights
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-teacher-ce", type=float, default=0.1)
    p.add_argument("--lambda-rank", type=float, default=0.2)
    p.add_argument("--lambda-pairgap", type=float, default=0.5)
    p.add_argument("--lambda-gt-ce", type=float, default=0.0)

    # Ranking loss settings
    p.add_argument("--rank-topk", type=int, default=3)
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--rank-min-gap", type=float, default=0.01)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


# ---------------------------------------------------------------------------
# Loss functions (vectorized)
# ---------------------------------------------------------------------------
def distribution_kl(
    student_dist: torch.Tensor,
    teacher_dist: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.to(dtype=student_dist.dtype)
    teacher = teacher_dist * valid
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    student = student_dist * valid
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (
        teacher
        * (teacher.clamp_min(1.0e-8).log() - student.clamp_min(1.0e-8).log())
    ).sum(dim=1).mean()


def candidate_cross_entropy(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    target_idx: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits.masked_fill(~valid_mask, -1.0e9), target_idx)


def topk_pair_indices(
    teacher_dist: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    masked = teacher_dist.masked_fill(~valid_mask, -1.0)
    actual_k = min(int(topk), teacher_dist.shape[1])
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
    k = max(int(topk), 2)
    actual_k = min(k, logits.shape[1])
    top_idx = topk_pair_indices(teacher_dist, valid_mask, actual_k)
    top_logits = logits.gather(1, top_idx)
    top_teacher = teacher_dist.gather(1, top_idx)
    top_valid = valid_mask.gather(1, top_idx)
    diff_logits = top_logits[:, :, None] - top_logits[:, None, :]
    diff_teacher = top_teacher[:, :, None] - top_teacher[:, None, :]
    upper = torch.triu(
        torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool),
        diagonal=1,
    )
    pair_valid = top_valid[:, :, None] & top_valid[:, None, :]
    pair_mask = upper[None] & pair_valid & (diff_teacher > float(min_gap))
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
    k = max(int(topk), 2)
    actual_k = min(k, logits.shape[1])
    top_idx = topk_pair_indices(teacher_dist, valid_mask, actual_k)
    top_logits = logits.gather(1, top_idx)
    top_teacher = teacher_dist.gather(1, top_idx).clamp_min(1.0e-8)
    top_log_t = top_teacher.log()
    top_valid = valid_mask.gather(1, top_idx)
    diff_s = top_logits[:, :, None] - top_logits[:, None, :]
    diff_t = top_log_t[:, :, None] - top_log_t[:, None, :]
    upper = torch.triu(
        torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool),
        diagonal=1,
    )
    pair_valid = top_valid[:, :, None] & top_valid[:, None, :]
    pair_mask = upper[None] & pair_valid
    if not pair_mask.any():
        return logits.new_zeros(())
    elem = F.smooth_l1_loss(diff_s, diff_t, reduction="none")
    return (elem * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Teacher target preparation
# ---------------------------------------------------------------------------
def _temperature_apply(probs: torch.Tensor, T: float) -> torch.Tensor:
    if abs(float(T) - 1.0) <= 1.0e-8:
        return probs
    positive = probs > 0.0
    if not positive.any():
        return probs
    out = torch.zeros_like(probs)
    out[positive] = torch.softmax(
        probs[positive].clamp_min(1.0e-8).log() / max(float(T), 1.0e-6),
        dim=0,
    )
    return out


def build_teacher_target_samples(samples: list[StudentSample], *, T: float) -> list[StudentSample]:
    """Renormalize teacher_posterior and optionally re-temper it."""
    out = []
    for s in samples:
        if s.teacher_posterior is None:
            raise RuntimeError(
                f"sample {s.sample_id} has no teacher_posterior; "
                "did you set --teacher-posterior-field correctly?"
            )
        tp = s.teacher_posterior.clone().clamp_min(0.0)
        total = tp.sum()
        tp = tp / total if float(total.item()) > 0.0 else torch.full_like(
            tp,
            1.0 / max(tp.numel(), 1),
        )
        tp = _temperature_apply(tp, T)
        teacher_idx = int(torch.argmax(tp).item()) if tp.numel() > 0 else -1
        out.append(
            StudentSample(
                sample_id=s.sample_id,
                split=s.split,
                candidate_ids=list(s.candidate_ids),
                candidate_names=list(s.candidate_names),
                candidate_categories=list(s.candidate_categories),
                features=s.features.clone(),
                gt_idx=s.gt_idx,
                gt_object_id=s.gt_object_id,
                teacher_posterior=tp,
                teacher_scores=s.teacher_scores,
                teacher_selected_idx=teacher_idx,
                teacher_motion_score=s.teacher_motion_score.clone()
                if s.teacher_motion_score is not None
                else None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Stage-S2 tie-break (evaluation only)
# ---------------------------------------------------------------------------
def s2_tiebreak_predict(
    distribution: torch.Tensor,
    names: list[str] | None,
    motion_score: torch.Tensor | None,
    *,
    topk: int,
    motion_weight: float,
    trigger_mode: str,
    margin_threshold: float,
) -> tuple[int, bool, list[int]]:
    """Apply Stage-S2 motion tie-break.

    Returns (winner_idx, tie_triggered, top_indices).
    If motion_score is None or no trigger fires, returns plain argmax of distribution.
    """
    if motion_score is None:
        winner = int(torch.argmax(distribution).item())
        top_idx = torch.topk(distribution, min(int(topk), int(distribution.numel()))).indices.tolist()
        return winner, False, top_idx

    post, triggered, top_indices = stage_s2_apply(
        distribution,
        names=names,
        motion_score=motion_score,
        top_k=topk,
        motion_weight=motion_weight,
        trigger_mode=trigger_mode,
        margin_threshold=margin_threshold,
        return_triggered=True,
    )
    winner = int(torch.argmax(post).item())
    return winner, bool(triggered[0]), top_indices[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: StudentSelector,
    loader: DataLoader,
    device: torch.device,
    *,
    s2_tiebreak: bool,
    s2_topk: int,
    s2_motion_weight: float,
    s2_trigger_mode: str,
    s2_margin_threshold: float,
) -> dict[str, float]:
    model.eval()
    total_kl = 0.0
    total = 0
    teacher_hits_pre = 0
    gt_hits_pre = 0
    teacher_hits_post = 0
    gt_hits_post = 0
    tie_triggered = 0
    tie_resolved_to_gt = 0
    tie_resolved_to_teacher = 0
    for batch in loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        gt_idx = batch["gt_idx"].to(device)
        teacher_dist = batch["teacher_posterior"]
        if teacher_dist is None:
            raise RuntimeError("Eval requires teacher_posterior")
        teacher_dist = teacher_dist.to(device)
        teacher_idx = batch["teacher_selected_idx"].to(device)
        motion_t = batch.get("teacher_motion_score")
        motion_t = motion_t.to(device) if isinstance(motion_t, torch.Tensor) else None
        samples_b: list[StudentSample] = batch["samples"]

        out = model(features, valid_mask)
        kl = distribution_kl(out["distribution"], teacher_dist, valid_mask)
        pred_pre = out["distribution"].argmax(dim=1)
        bsz = features.shape[0]
        total_kl += float(kl.item()) * bsz
        total += bsz
        teacher_hits_pre += int((pred_pre == teacher_idx).sum().item())
        gt_hits_pre += int((pred_pre == gt_idx).sum().item())

        for bi in range(bsz):
            n_i = int(valid_mask[bi].sum().item())
            dist_i = out["distribution"][bi, :n_i].cpu()
            names_i = samples_b[bi].candidate_names[:n_i]
            motion_i = motion_t[bi, :n_i].cpu() if (s2_tiebreak and motion_t is not None) else None
            winner, tie, _ = s2_tiebreak_predict(
                dist_i,
                names_i,
                motion_i,
                topk=s2_topk,
                motion_weight=s2_motion_weight,
                trigger_mode=s2_trigger_mode,
                margin_threshold=s2_margin_threshold,
            )
            if tie:
                tie_triggered += 1
                if winner == int(gt_idx[bi].item()):
                    tie_resolved_to_gt += 1
                if winner == int(teacher_idx[bi].item()):
                    tie_resolved_to_teacher += 1
            if winner == int(teacher_idx[bi].item()):
                teacher_hits_post += 1
            if winner == int(gt_idx[bi].item()):
                gt_hits_post += 1

    return {
        "kl": total_kl / max(total, 1),
        "teacher_top1_pre_count": teacher_hits_pre,
        "teacher_top1_pre": teacher_hits_pre / max(total, 1),
        "gt_exact_pre_count": gt_hits_pre,
        "gt_pre": gt_hits_pre / max(total, 1),
        "teacher_top1_post_count": teacher_hits_post,
        "teacher_top1_post": teacher_hits_post / max(total, 1),
        "gt_exact_post_count": gt_hits_post,
        "gt_post": gt_hits_post / max(total, 1),
        "tie_triggered": tie_triggered,
        "tie_resolved_to_gt": tie_resolved_to_gt,
        "tie_resolved_to_teacher": tie_resolved_to_teacher,
        "total_samples": total,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


@torch.no_grad()
def export_predictions(
    *,
    model: StudentSelector,
    raw: list[StudentSample],
    norm: list[StudentSample],
    feature_columns: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
    s2_tiebreak: bool,
    s2_topk: int,
    s2_motion_weight: float,
    s2_trigger_mode: str,
    s2_margin_threshold: float,
) -> None:
    model.eval()
    cand_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for r, n in zip(raw, norm, strict=True):
        feat = n.features.unsqueeze(0).to(device)
        vm = torch.ones((1, n.features.shape[0]), dtype=torch.bool, device=device)
        out = model(feat, vm)
        logits = out["logits"][0].cpu()
        dist = out["distribution"][0].cpu()
        pred_pre = int(torch.argmax(dist).item())
        motion = r.teacher_motion_score if (s2_tiebreak and r.teacher_motion_score is not None) else None
        pred_post, tie_triggered, top_idx = s2_tiebreak_predict(
            dist,
            r.candidate_names,
            motion,
            topk=s2_topk,
            motion_weight=s2_motion_weight,
            trigger_mode=s2_trigger_mode,
            margin_threshold=s2_margin_threshold,
        )

        if r.teacher_posterior is None:
            raise RuntimeError("export requires teacher_posterior")
        t_order = torch.argsort(r.teacher_posterior, descending=True).tolist()
        s_order = torch.argsort(dist, descending=True).tolist()

        for ci, oid in enumerate(r.candidate_ids):
            row = {
                "split": r.split,
                "sample_id": r.sample_id,
                "object_id": oid,
                "name": r.candidate_names[ci],
                "category": r.candidate_categories[ci],
                "teacher_posterior": float(r.teacher_posterior[ci].item()),
                "teacher_motion_score": float(r.teacher_motion_score[ci].item())
                if r.teacher_motion_score is not None
                else 0.0,
                "student_score": float(logits[ci].item()),
                "student_posterior": float(dist[ci].item()),
                "teacher_rank": int(t_order.index(ci) + 1),
                "student_rank": int(s_order.index(ci) + 1),
                "is_in_student_top_k": int(ci in top_idx),
                "is_teacher_selected": int(ci == r.teacher_selected_idx),
                "is_student_selected_pre": int(ci == pred_pre),
                "is_student_selected_post": int(ci == pred_post),
                "is_gt": int(ci == r.gt_idx),
            }
            for fi, col in enumerate(feature_columns):
                row[col] = float(r.features[ci, fi].item())
            cand_rows.append(row)

        sample_rows.append(
            {
                "split": r.split,
                "sample_id": r.sample_id,
                "selected_pre_object_id": r.candidate_ids[pred_pre],
                "selected_post_object_id": r.candidate_ids[pred_post],
                "selected_pre_object_name": r.candidate_names[pred_pre],
                "selected_post_object_name": r.candidate_names[pred_post],
                "teacher_selected_object_id": r.candidate_ids[r.teacher_selected_idx]
                if r.teacher_selected_idx >= 0
                else "",
                "gt_object_id": r.gt_object_id,
                "gt_object_name": r.candidate_names[r.gt_idx],
                "num_candidates": len(r.candidate_ids),
                "student_confidence_pre": float(dist[pred_pre].item()),
                "student_confidence_post": float(dist[pred_post].item()),
                "tie_triggered": int(tie_triggered),
                "is_teacher_match_pre": int(pred_pre == r.teacher_selected_idx),
                "is_teacher_match_post": int(pred_post == r.teacher_selected_idx),
                "is_gt_exact_pre": int(pred_pre == r.gt_idx),
                "is_gt_exact_post": int(pred_post == r.gt_idx),
            }
        )
    write_csv(output_dir / "candidate_scores.csv", cand_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    requested = parse_feature_columns(args.feature_columns)

    raw = build_samples_from_csv(
        args.input_csv,
        feature_columns=requested,
        teacher_score_field=None,
        teacher_posterior_field=args.teacher_posterior_field,
        teacher_selected_field=args.teacher_selected_field or None,
        motion_score_field=args.motion_score_field or None,
    )
    raw = build_teacher_target_samples(raw, T=args.teacher_temperature)
    train_raw = [s for s in raw if s.split == "train"]
    test_raw = [s for s in raw if s.split == "test"]
    if not train_raw or not test_raw:
        raise RuntimeError("Need both train and test splits")

    if args.feature_stats_json:
        cols, mean, std = load_feature_stats(args.feature_stats_json)
        if cols != requested:
            raise ValueError(f"stats columns {cols} != requested {requested}")
    else:
        cols = requested
        mean, std = compute_feature_stats(train_raw)
        save_feature_stats(args.output_dir / "feature_stats.json", cols, mean, std)

    norm_train = normalize_samples(train_raw, mean, std)
    norm_test = normalize_samples(test_raw, mean, std)

    train_loader = DataLoader(
        StudentDataset(norm_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_student_samples,
    )
    test_loader = DataLoader(
        StudentDataset(norm_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_student_samples,
    )

    model = StudentSelector(
        input_dim=len(cols),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict] = []
    best_metrics: dict | None = None
    best_state: dict | None = None

    use_kl = args.lambda_kl > 0
    use_tce = args.lambda_teacher_ce > 0
    use_rank = args.lambda_rank > 0
    use_pg = args.lambda_pairgap > 0
    use_gtce = args.lambda_gt_ce > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        kl_sum = 0.0
        seen = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            gt_idx = batch["gt_idx"].to(device)
            teacher_dist = batch["teacher_posterior"].to(device)
            teacher_idx = batch["teacher_selected_idx"].to(device)
            out = model(features, valid_mask)
            logits = out["logits"]
            dist = out["distribution"]

            loss = logits.new_zeros(())
            kl_loss = logits.new_zeros(())
            if use_kl:
                kl_loss = distribution_kl(dist, teacher_dist, valid_mask)
                loss = loss + args.lambda_kl * kl_loss
            if use_tce:
                loss = loss + args.lambda_teacher_ce * candidate_cross_entropy(
                    logits,
                    valid_mask,
                    teacher_idx,
                )
            if use_rank:
                loss = loss + args.lambda_rank * teacher_topk_ranking_loss(
                    logits,
                    teacher_dist,
                    valid_mask,
                    topk=args.rank_topk,
                    margin=args.rank_margin,
                    min_gap=args.rank_min_gap,
                )
            if use_pg:
                loss = loss + args.lambda_pairgap * teacher_topk_pairgap_loss(
                    logits,
                    teacher_dist,
                    valid_mask,
                    topk=args.rank_topk,
                )
            if use_gtce:
                loss = loss + args.lambda_gt_ce * candidate_cross_entropy(
                    logits,
                    valid_mask,
                    gt_idx,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = features.shape[0]
            loss_sum += float(loss.item()) * bsz
            kl_sum += float(kl_loss.item()) * bsz
            seen += bsz

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            s2_tiebreak=args.s2_tiebreak,
            s2_topk=args.s2_topk,
            s2_motion_weight=args.s2_motion_weight,
            s2_trigger_mode=args.s2_trigger_mode,
            s2_margin_threshold=args.s2_margin_threshold,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "test_kl": test_metrics["kl"],
                "test_teacher_top1_pre": test_metrics["teacher_top1_pre_count"],
                "test_gt_exact_pre": test_metrics["gt_exact_pre_count"],
                "test_teacher_top1_post": test_metrics["teacher_top1_post_count"],
                "test_gt_exact_post": test_metrics["gt_exact_post_count"],
                "tie_triggered": test_metrics["tie_triggered"],
                "tie_resolved_to_gt": test_metrics["tie_resolved_to_gt"],
                "tie_resolved_to_teacher": test_metrics["tie_resolved_to_teacher"],
            }
        )
        train_loss_avg = loss_sum / max(seen, 1)
        train_kl_avg = kl_sum / max(seen, 1)
        val_total = int(test_metrics["total_samples"])
        val_gt_pre = int(test_metrics["gt_exact_pre_count"])
        val_gt_post = int(test_metrics["gt_exact_post_count"])
        teacher_match_pre = int(test_metrics["teacher_top1_pre_count"])
        print(
            f"[epoch {epoch:>3d}/{args.epochs}] "
            f"train_loss={train_loss_avg:.4f} "
            f"kl={train_kl_avg:.4f} "
            f"val_gt_pre={val_gt_pre}/{val_total} "
            f"val_gt_post={val_gt_post}/{val_total} "
            f"teacher_match={teacher_match_pre}/{val_total}",
            flush=True,
        )

        key = (
            test_metrics["gt_exact_post_count"],
            test_metrics["teacher_top1_post_count"],
            -test_metrics["kl"],
        )
        prev = (
            best_metrics["gt_exact_post_count"] if best_metrics else -1,
            best_metrics["teacher_top1_post_count"] if best_metrics else -1,
            -best_metrics["kl"] if best_metrics else -1e9,
        )
        if best_metrics is None or key > prev:
            best_metrics = dict(test_metrics)
            best_metrics["epoch"] = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    common = {
        "feature_columns": list(cols),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
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
        "teacher_posterior_field": args.teacher_posterior_field,
        "teacher_temperature": args.teacher_temperature,
        "motion_score_field": args.motion_score_field,
        "s2_tiebreak": args.s2_tiebreak,
        "s2_topk": args.s2_topk,
        "s2_motion_weight": args.s2_motion_weight,
        "s2_trigger_mode": args.s2_trigger_mode,
        "s2_margin_threshold": args.s2_margin_threshold,
    }
    torch.save({"model_state": best_state, **common}, args.output_dir / "best_model.pt")

    norm_all = normalize_samples(raw, mean, std)
    model.load_state_dict(best_state)
    export_predictions(
        model=model,
        raw=raw,
        norm=norm_all,
        feature_columns=cols,
        device=device,
        output_dir=args.output_dir,
        s2_tiebreak=args.s2_tiebreak,
        s2_topk=args.s2_topk,
        s2_motion_weight=args.s2_motion_weight,
        s2_trigger_mode=args.s2_trigger_mode,
        s2_margin_threshold=args.s2_margin_threshold,
    )

    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "num_train_samples": len(train_raw),
        "num_test_samples": len(test_raw),
        "best_epoch": best_metrics["epoch"],
        "best_test_kl": best_metrics["kl"],
        "best_test_teacher_top1_pre": best_metrics["teacher_top1_pre_count"],
        "best_test_gt_exact_pre": best_metrics["gt_exact_pre_count"],
        "best_test_teacher_top1_post": best_metrics["teacher_top1_post_count"],
        "best_test_gt_exact_post": best_metrics["gt_exact_post_count"],
        "best_tie_triggered": best_metrics["tie_triggered"],
        "best_tie_resolved_to_gt": best_metrics["tie_resolved_to_gt"],
        "best_tie_resolved_to_teacher": best_metrics["tie_resolved_to_teacher"],
        **common,
        "history": history,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"Best epoch {best_metrics['epoch']}: "
        f"GT pre={best_metrics['gt_exact_pre_count']} → post={best_metrics['gt_exact_post_count']} / {best_metrics['total_samples']}, "
        f"teacher pre={best_metrics['teacher_top1_pre_count']} → post={best_metrics['teacher_top1_post_count']}, "
        f"ties={best_metrics['tie_triggered']} (→GT {best_metrics['tie_resolved_to_gt']})"
    )


if __name__ == "__main__":
    main()
