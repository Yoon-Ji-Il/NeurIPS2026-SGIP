#!/usr/bin/env python3
"""Train the same-space student to mimic the teacher's fusion posterior."""

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
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill a same-space student from teacher CSV.")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--feature-columns", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS))
    p.add_argument("--feature-stats-json", type=Path, default=None)

    p.add_argument("--teacher-target-field", type=str, default="fusion_total")
    p.add_argument("--teacher-target-mode", choices=("score_softmax", "probability"), default="score_softmax")
    p.add_argument("--teacher-temperature", type=float, default=0.5)
    p.add_argument("--teacher-selected-field", type=str, default="is_teacher_selected")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-teacher-ce", type=float, default=0.1)
    p.add_argument("--lambda-rank", type=float, default=0.2)
    p.add_argument("--lambda-pairgap", type=float, default=0.5)
    p.add_argument("--lambda-gt-ce", type=float, default=0.0)

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


# ----- Loss functions (vectorized) ------------------------------------------
def distribution_kl(student_dist: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=student_dist.dtype)
    teacher = teacher_dist * valid
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    student = student_dist * valid
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (teacher * (teacher.clamp_min(1.0e-8).log() - student.clamp_min(1.0e-8).log())).sum(dim=1).mean()


def candidate_cross_entropy(logits: torch.Tensor, valid_mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.masked_fill(~valid_mask, -1.0e9), target_idx)


def topk_pair_indices(teacher_dist: torch.Tensor, valid_mask: torch.Tensor, topk: int) -> torch.Tensor:
    masked = teacher_dist.masked_fill(~valid_mask, -1.0)
    actual_k = min(int(topk), teacher_dist.shape[1])
    return masked.topk(actual_k, dim=1).indices


def teacher_topk_ranking_loss(
    logits: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor,
    *, topk: int, margin: float, min_gap: float,
) -> torch.Tensor:
    k = max(int(topk), 2)
    actual_k = min(k, logits.shape[1])
    top_idx = topk_pair_indices(teacher_dist, valid_mask, actual_k)
    top_logits = logits.gather(1, top_idx)
    top_teacher = teacher_dist.gather(1, top_idx)
    top_valid = valid_mask.gather(1, top_idx)
    diff_logits = top_logits[:, :, None] - top_logits[:, None, :]
    diff_teacher = top_teacher[:, :, None] - top_teacher[:, None, :]
    upper = torch.triu(torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool), diagonal=1)
    pair_valid = top_valid[:, :, None] & top_valid[:, None, :]
    pair_mask = upper[None] & pair_valid & (diff_teacher > float(min_gap))
    if not pair_mask.any():
        return logits.new_zeros(())
    hinge = F.relu(float(margin) - diff_logits)
    return (hinge * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def teacher_topk_pairgap_loss(
    logits: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor, *, topk: int,
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
    upper = torch.triu(torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool), diagonal=1)
    pair_valid = top_valid[:, :, None] & top_valid[:, None, :]
    pair_mask = upper[None] & pair_valid
    if not pair_mask.any():
        return logits.new_zeros(())
    elem = F.smooth_l1_loss(diff_s, diff_t, reduction="none")
    return (elem * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


# ----- Teacher target conversion --------------------------------------------
def score_softmax_target(scores: torch.Tensor, T: float) -> torch.Tensor:
    return torch.softmax(scores / max(float(T), 1.0e-6), dim=0)


def probability_target(values: torch.Tensor, T: float) -> torch.Tensor:
    probs = values.clamp_min(0.0)
    total = probs.sum()
    if float(total.item()) <= 0.0:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    probs = probs / total
    if abs(float(T) - 1.0) <= 1.0e-8:
        return probs
    positive = probs > 0.0
    if not positive.any():
        return probs
    out = torch.zeros_like(probs)
    out[positive] = torch.softmax(probs[positive].clamp_min(1.0e-8).log() / max(float(T), 1.0e-6), dim=0)
    return out


def build_teacher_target_samples(samples: list[StudentSample], *, target_mode: str, T: float) -> list[StudentSample]:
    out = []
    for s in samples:
        ts = s.teacher_scores.clone() if s.teacher_scores is not None else None
        tp = s.teacher_posterior.clone() if s.teacher_posterior is not None else None
        if target_mode == "score_softmax":
            if ts is None:
                raise RuntimeError("score_softmax requires teacher_scores")
            tp = score_softmax_target(ts, T)
        elif target_mode == "probability":
            if tp is None:
                if ts is None:
                    raise RuntimeError("probability requires teacher_posterior or scores")
                tp = probability_target(ts, T)
            else:
                tp = probability_target(tp, T)
        else:
            raise ValueError(f"unknown target mode {target_mode}")
        teacher_idx = int(torch.argmax(tp).item()) if tp.numel() > 0 else -1
        out.append(
            StudentSample(
                sample_id=s.sample_id, split=s.split,
                candidate_ids=list(s.candidate_ids),
                candidate_names=list(s.candidate_names),
                candidate_categories=list(s.candidate_categories),
                features=s.features.clone(),
                gt_idx=s.gt_idx, gt_object_id=s.gt_object_id,
                teacher_posterior=tp, teacher_scores=ts,
                teacher_selected_idx=teacher_idx,
                teacher_motion_score=s.teacher_motion_score.clone() if s.teacher_motion_score is not None else None,
            )
        )
    return out


# ----- Evaluation -----------------------------------------------------------
@torch.no_grad()
def evaluate(model: StudentSelector, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_kl = 0.0
    total = 0
    teacher_hits = 0
    gt_hits = 0
    for batch in loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        gt_idx = batch["gt_idx"].to(device)
        teacher_dist = batch["teacher_posterior"]
        if teacher_dist is None:
            raise RuntimeError("Eval requires teacher_posterior")
        teacher_dist = teacher_dist.to(device)
        teacher_idx = batch["teacher_selected_idx"].to(device)
        out = model(features, valid_mask)
        kl = distribution_kl(out["distribution"], teacher_dist, valid_mask)
        pred = out["distribution"].argmax(dim=1)
        bsz = features.shape[0]
        total_kl += float(kl.item()) * bsz
        total += bsz
        teacher_hits += int((pred == teacher_idx).sum().item())
        gt_hits += int((pred == gt_idx).sum().item())
    return {
        "kl": total_kl / max(total, 1),
        "teacher_top1_count": teacher_hits,
        "teacher_top1": teacher_hits / max(total, 1),
        "gt_exact_count": gt_hits,
        "gt_accuracy": gt_hits / max(total, 1),
        "total_samples": total,
    }


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
    *, model: StudentSelector, raw: list[StudentSample], norm: list[StudentSample],
    feature_columns: tuple[str, ...], device: torch.device, output_dir: Path,
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
        pred = int(torch.argmax(dist).item())
        if r.teacher_posterior is None:
            raise RuntimeError("export requires teacher_posterior")
        t_order = torch.argsort(r.teacher_posterior, descending=True).tolist()
        s_order = torch.argsort(dist, descending=True).tolist()
        for ci, oid in enumerate(r.candidate_ids):
            row = {
                "split": r.split, "sample_id": r.sample_id, "object_id": oid,
                "name": r.candidate_names[ci], "category": r.candidate_categories[ci],
                "teacher_score": float(r.teacher_scores[ci].item()) if r.teacher_scores is not None else 0.0,
                "teacher_posterior": float(r.teacher_posterior[ci].item()),
                "student_score": float(logits[ci].item()),
                "student_posterior": float(dist[ci].item()),
                "teacher_rank": int(t_order.index(ci) + 1),
                "student_rank": int(s_order.index(ci) + 1),
                "is_teacher_selected": int(ci == r.teacher_selected_idx),
                "is_student_selected": int(ci == pred),
                "is_gt": int(ci == r.gt_idx),
            }
            for fi, col in enumerate(feature_columns):
                row[col] = float(r.features[ci, fi].item())
            cand_rows.append(row)
        sample_rows.append({
            "split": r.split, "sample_id": r.sample_id,
            "selected_object_id": r.candidate_ids[pred],
            "selected_object_name": r.candidate_names[pred],
            "teacher_selected_object_id": r.candidate_ids[r.teacher_selected_idx] if r.teacher_selected_idx >= 0 else "",
            "gt_object_id": r.gt_object_id,
            "gt_object_name": r.candidate_names[r.gt_idx],
            "num_candidates": len(r.candidate_ids),
            "student_confidence": float(dist[pred].item()),
            "teacher_confidence": float(r.teacher_posterior[r.teacher_selected_idx].item()) if r.teacher_selected_idx >= 0 else 0.0,
            "is_teacher_match": int(pred == r.teacher_selected_idx),
            "is_gt_exact": int(pred == r.gt_idx),
        })
    write_csv(output_dir / "candidate_scores.csv", cand_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    requested = parse_feature_columns(args.feature_columns)

    raw = build_samples_from_csv(
        args.input_csv,
        feature_columns=requested,
        teacher_score_field=args.teacher_target_field if args.teacher_target_mode == "score_softmax" else None,
        teacher_posterior_field=args.teacher_target_field if args.teacher_target_mode == "probability" else None,
        teacher_selected_field=args.teacher_selected_field or None,
    )
    raw = build_teacher_target_samples(raw, target_mode=args.teacher_target_mode, T=args.teacher_temperature)
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

    train_loader = DataLoader(StudentDataset(norm_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_student_samples)
    test_loader = DataLoader(StudentDataset(norm_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_student_samples)

    model = StudentSelector(input_dim=len(cols), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    best_metrics: dict | None = None
    best_state: dict | None = None
    best_gt_metrics: dict | None = None
    best_gt_state: dict | None = None

    use_kl, use_tce = args.lambda_kl > 0, args.lambda_teacher_ce > 0
    use_rank, use_pg, use_gtce = args.lambda_rank > 0, args.lambda_pairgap > 0, args.lambda_gt_ce > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        train_thits = 0
        train_ghits = 0
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
            if use_kl:
                loss = loss + args.lambda_kl * distribution_kl(dist, teacher_dist, valid_mask)
            if use_tce:
                loss = loss + args.lambda_teacher_ce * candidate_cross_entropy(logits, valid_mask, teacher_idx)
            if use_rank:
                loss = loss + args.lambda_rank * teacher_topk_ranking_loss(
                    logits, teacher_dist, valid_mask,
                    topk=args.rank_topk, margin=args.rank_margin, min_gap=args.rank_min_gap,
                )
            if use_pg:
                loss = loss + args.lambda_pairgap * teacher_topk_pairgap_loss(
                    logits, teacher_dist, valid_mask, topk=args.rank_topk,
                )
            if use_gtce:
                loss = loss + args.lambda_gt_ce * candidate_cross_entropy(logits, valid_mask, gt_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = dist.argmax(dim=1)
                train_thits += int((pred == teacher_idx).sum().item())
                train_ghits += int((pred == gt_idx).sum().item())
            bsz = features.shape[0]
            loss_sum += float(loss.item()) * bsz
            seen += bsz

        test_metrics = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_teacher_top1_count": train_thits,
            "train_gt_exact_count": train_ghits,
            "test_teacher_top1_count": test_metrics["teacher_top1_count"],
            "test_gt_exact_count": test_metrics["gt_exact_count"],
            "test_kl": test_metrics["kl"],
        })

        tkey = (test_metrics["teacher_top1_count"], test_metrics["gt_exact_count"], -test_metrics["kl"])
        if best_metrics is None or tkey > (best_metrics["teacher_top1_count"], best_metrics["gt_exact_count"], -best_metrics["kl"]):
            best_metrics = dict(test_metrics); best_metrics["epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        gkey = (test_metrics["gt_exact_count"], test_metrics["teacher_top1_count"], -test_metrics["kl"])
        if best_gt_metrics is None or gkey > (best_gt_metrics["gt_exact_count"], best_gt_metrics["teacher_top1_count"], -best_gt_metrics["kl"]):
            best_gt_metrics = dict(test_metrics); best_gt_metrics["epoch"] = epoch
            best_gt_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
        "teacher_target_field": args.teacher_target_field,
        "teacher_target_mode": args.teacher_target_mode,
        "teacher_temperature": args.teacher_temperature,
    }
    torch.save({"model_state": best_state, **common}, args.output_dir / "best_model.pt")
    torch.save({"model_state": best_gt_state, **common}, args.output_dir / "best_model_gt.pt")

    norm_all = normalize_samples(raw, mean, std)
    model.load_state_dict(best_state)
    export_predictions(model=model, raw=raw, norm=norm_all, feature_columns=cols, device=device, output_dir=args.output_dir)
    model.load_state_dict(best_gt_state)
    export_predictions(model=model, raw=raw, norm=norm_all, feature_columns=cols, device=device, output_dir=args.output_dir / "best_gt_export")

    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "num_train_samples": len(train_raw),
        "num_test_samples": len(test_raw),
        "best_epoch": best_metrics["epoch"],
        "best_teacher_top1_count": best_metrics["teacher_top1_count"],
        "best_gt_exact_count": best_metrics["gt_exact_count"],
        "best_kl": best_metrics["kl"],
        "best_gt_epoch": best_gt_metrics["epoch"],
        "best_gt_test_teacher_top1_count": best_gt_metrics["teacher_top1_count"],
        "best_gt_test_gt_exact_count": best_gt_metrics["gt_exact_count"],
        **common,
        "history": history,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"Best epoch {best_metrics['epoch']}: teacher={best_metrics['teacher_top1_count']}/{best_metrics['total_samples']}, gt={best_metrics['gt_exact_count']}")


if __name__ == "__main__":
    main()
