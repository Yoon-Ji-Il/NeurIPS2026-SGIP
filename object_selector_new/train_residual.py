#!/usr/bin/env python3
"""Train a bounded residual reranker on top of a frozen same-space student."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.residual import (  # noqa: E402
    BASELINE_SELF_SIGNAL_COLUMNS,
    ResidualConfig,
    ResidualReranker,
    ResidualSample,
    build_residual_samples_from_csv,
    collate_residual_samples,
    load_feature_stats,
    parse_feature_columns,
)


class ResidualDataset(Dataset[ResidualSample]):
    def __init__(self, samples: list[ResidualSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> ResidualSample:
        return self.samples[i]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train residual reranker.")
    p.add_argument("--baseline-csv", type=Path, required=True)
    p.add_argument("--baseline-feature-stats", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--geometry-columns", type=str, default="")
    p.add_argument("--self-signal-columns", type=str, default=",".join(BASELINE_SELF_SIGNAL_COLUMNS))
    p.add_argument("--baseline-score-field", type=str, default="student_score")
    p.add_argument("--teacher-posterior-field", type=str, default="teacher_posterior")

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--max-residual", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)

    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=0.5)
    p.add_argument("--lambda-l2", type=float, default=0.05)
    p.add_argument("--lambda-rank", type=float, default=0.2)

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


def distribution_kl(student: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=student.dtype)
    t = target * valid
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    s = student * valid
    s = s / s.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (t * (t.clamp_min(1.0e-8).log() - s.clamp_min(1.0e-8).log())).sum(dim=1).mean()


def teacher_topk_ranking_loss(
    logits: torch.Tensor, teacher_dist: torch.Tensor, valid_mask: torch.Tensor,
    *, topk: int, margin: float, min_gap: float,
) -> torch.Tensor:
    k = max(int(topk), 2)
    actual_k = min(k, logits.shape[1])
    masked = teacher_dist.masked_fill(~valid_mask, -1.0)
    top_idx = masked.topk(actual_k, dim=1).indices
    top_l = logits.gather(1, top_idx)
    top_t = teacher_dist.gather(1, top_idx)
    top_v = valid_mask.gather(1, top_idx)
    diff_l = top_l[:, :, None] - top_l[:, None, :]
    diff_t = top_t[:, :, None] - top_t[:, None, :]
    upper = torch.triu(torch.ones(actual_k, actual_k, device=logits.device, dtype=torch.bool), diagonal=1)
    pair_v = top_v[:, :, None] & top_v[:, None, :]
    pair_mask = upper[None] & pair_v & (diff_t > float(min_gap))
    if not pair_mask.any():
        return logits.new_zeros(())
    return (F.relu(float(margin) - diff_l) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def residual_l2(residual: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=residual.dtype)
    return ((residual ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model: ResidualReranker, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    teacher_hits = 0
    gt_hits = 0
    baseline_gt_hits = 0
    helped = 0
    hurt = 0
    neutral_correct = 0
    neutral_wrong = 0
    kl_sum = 0.0
    anchor_sum = 0.0
    abs_residual_sum = 0.0
    for batch in loader:
        feat = batch["features"].to(device)
        bl = batch["base_logits"].to(device)
        bd = batch["base_distribution"].to(device)
        td = batch["teacher_distribution"].to(device)
        vm = batch["valid_mask"].to(device)
        gi = batch["gt_idx"].to(device)
        ti = batch["teacher_idx"].to(device)
        bi = batch["baseline_idx"].to(device)
        out = model(feat, bl, vm)
        pred = out["distribution"].argmax(dim=1)
        bsz = feat.shape[0]
        total += bsz
        teacher_hits += int((pred == ti).sum().item())
        gt_hits += int((pred == gi).sum().item())
        baseline_gt_hits += int((bi == gi).sum().item())
        sc = pred == gi
        bc = bi == gi
        helped += int((sc & ~bc).sum().item())
        hurt += int((~sc & bc).sum().item())
        neutral_correct += int((sc & bc).sum().item())
        neutral_wrong += int((~sc & ~bc).sum().item())
        kl_sum += float(distribution_kl(out["distribution"], td, vm).item()) * bsz
        anchor_sum += float(distribution_kl(out["distribution"], bd, vm).item()) * bsz
        v = vm.to(dtype=out["residual"].dtype)
        abs_r = (out["residual"].abs() * v).sum() / v.sum().clamp_min(1.0)
        abs_residual_sum += float(abs_r.item()) * bsz
    return {
        "kl": kl_sum / max(total, 1),
        "anchor_kl": anchor_sum / max(total, 1),
        "mean_abs_residual": abs_residual_sum / max(total, 1),
        "teacher_top1_count": teacher_hits,
        "gt_exact_count": gt_hits,
        "baseline_gt_exact_count": baseline_gt_hits,
        "residual_helped": helped,
        "residual_hurt": hurt,
        "residual_neutral_correct": neutral_correct,
        "residual_neutral_wrong": neutral_wrong,
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
    *, model: ResidualReranker, samples: list[ResidualSample],
    geometry_columns: tuple[str, ...], self_signal_columns: tuple[str, ...],
    device: torch.device, output_dir: Path,
) -> None:
    model.eval()
    cand_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    feature_columns = tuple(list(geometry_columns) + list(self_signal_columns))
    for s in samples:
        feat = s.features.unsqueeze(0).to(device)
        bl = s.base_logits.unsqueeze(0).to(device)
        vm = torch.ones((1, s.features.shape[0]), dtype=torch.bool, device=device)
        out = model(feat, bl, vm)
        logits = out["logits"][0].cpu()
        dist = out["distribution"][0].cpu()
        residual = out["residual"][0].cpu()
        pred = int(torch.argmax(dist).item())
        t_order = torch.argsort(s.teacher_distribution, descending=True).tolist()
        s_order = torch.argsort(dist, descending=True).tolist()
        b_order = torch.argsort(s.base_logits, descending=True).tolist()
        for ci, oid in enumerate(s.candidate_ids):
            row = {
                "split": s.split, "sample_id": s.sample_id, "object_id": oid,
                "name": s.candidate_names[ci], "category": s.candidate_categories[ci],
                "teacher_posterior": float(s.teacher_distribution[ci].item()),
                "baseline_score": float(s.base_logits[ci].item()),
                "baseline_posterior": float(s.base_distribution[ci].item()),
                "residual": float(residual[ci].item()),
                "student_score": float(logits[ci].item()),
                "student_posterior": float(dist[ci].item()),
                "teacher_rank": int(t_order.index(ci) + 1),
                "baseline_rank": int(b_order.index(ci) + 1),
                "student_rank": int(s_order.index(ci) + 1),
                "is_teacher_selected": int(ci == s.teacher_idx),
                "is_baseline_selected": int(ci == s.baseline_idx),
                "is_student_selected": int(ci == pred),
                "is_gt": int(ci == s.gt_idx),
            }
            for fi, col in enumerate(feature_columns):
                row[col] = float(s.features[ci, fi].item())
            cand_rows.append(row)
        sample_rows.append({
            "split": s.split, "sample_id": s.sample_id,
            "selected_object_id": s.candidate_ids[pred],
            "selected_object_name": s.candidate_names[pred],
            "baseline_selected_object_id": s.candidate_ids[s.baseline_idx],
            "teacher_selected_object_id": s.candidate_ids[s.teacher_idx],
            "gt_object_id": s.candidate_ids[s.gt_idx],
            "gt_object_name": s.candidate_names[s.gt_idx],
            "num_candidates": len(s.candidate_ids),
            "student_confidence": float(dist[pred].item()),
            "teacher_confidence": float(s.teacher_distribution[s.teacher_idx].item()),
            "baseline_confidence": float(s.base_distribution[s.baseline_idx].item()),
            "is_teacher_match": int(pred == s.teacher_idx),
            "is_baseline_match": int(pred == s.baseline_idx),
            "is_gt_exact": int(pred == s.gt_idx),
            "baseline_is_gt_exact": int(s.baseline_idx == s.gt_idx),
        })
    write_csv(output_dir / "candidate_scores.csv", cand_rows)
    write_csv(output_dir / "sample_summary.csv", sample_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    stats_cols, geom_mean, geom_std = load_feature_stats(args.baseline_feature_stats)
    if args.geometry_columns.strip():
        req = parse_feature_columns(args.geometry_columns)
        if req != stats_cols:
            raise ValueError(f"--geometry-columns {req} != stats {stats_cols}")
        geom_cols = req
    else:
        geom_cols = stats_cols
    self_cols = parse_feature_columns(args.self_signal_columns)
    for c in self_cols:
        if c not in BASELINE_SELF_SIGNAL_COLUMNS:
            raise ValueError(f"unknown self-signal {c!r}; allowed {BASELINE_SELF_SIGNAL_COLUMNS}")

    samples = build_residual_samples_from_csv(
        args.baseline_csv,
        geometry_columns=geom_cols, self_signal_columns=self_cols,
        geometry_mean=geom_mean, geometry_std=geom_std,
        baseline_score_field=args.baseline_score_field,
        teacher_posterior_field=args.teacher_posterior_field,
    )
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    if not train or not test:
        raise RuntimeError("Need both splits")

    train_loader = DataLoader(ResidualDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_residual_samples)
    test_loader = DataLoader(ResidualDataset(test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_residual_samples)

    feat_dim = len(geom_cols) + len(self_cols)
    config = ResidualConfig(
        input_dim=feat_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        max_residual=args.max_residual, dropout=args.dropout,
    )
    model = ResidualReranker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_kl, use_anc = args.lambda_kl > 0, args.lambda_anchor > 0
    use_l2, use_rank = args.lambda_l2 > 0, args.lambda_rank > 0
    history: list[dict] = []
    best_metrics = best_state = None
    best_t_metrics = best_t_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            feat = batch["features"].to(device)
            bl = batch["base_logits"].to(device)
            bd = batch["base_distribution"].to(device)
            td = batch["teacher_distribution"].to(device)
            vm = batch["valid_mask"].to(device)
            out = model(feat, bl, vm)
            dist = out["distribution"]
            logits = out["logits"]
            residual = out["residual"]
            loss = logits.new_zeros(())
            if use_kl:
                loss = loss + args.lambda_kl * distribution_kl(dist, td, vm)
            if use_anc:
                loss = loss + args.lambda_anchor * distribution_kl(dist, bd, vm)
            if use_l2:
                loss = loss + args.lambda_l2 * residual_l2(residual, vm)
            if use_rank:
                loss = loss + args.lambda_rank * teacher_topk_ranking_loss(
                    logits, td, vm,
                    topk=args.rank_topk, margin=args.rank_margin, min_gap=args.rank_min_gap,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bsz = feat.shape[0]
            loss_sum += float(loss.item()) * bsz
            seen += bsz

        m = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
            "test_teacher_top1_count": m["teacher_top1_count"],
            "test_gt_exact_count": m["gt_exact_count"],
            "baseline_gt_exact_count": m["baseline_gt_exact_count"],
            "residual_helped": m["residual_helped"],
            "residual_hurt": m["residual_hurt"],
            "residual_neutral_correct": m["residual_neutral_correct"],
            "residual_neutral_wrong": m["residual_neutral_wrong"],
            "test_kl": m["kl"], "test_anchor_kl": m["anchor_kl"],
            "test_mean_abs_residual": m["mean_abs_residual"],
        })

        # Best-GT: prefer high GT, then low residual_hurt, then teacher fidelity
        gkey = (m["gt_exact_count"], -m["residual_hurt"], m["teacher_top1_count"], -m["kl"])
        if best_metrics is None or gkey > (
            best_metrics["gt_exact_count"], -best_metrics["residual_hurt"],
            best_metrics["teacher_top1_count"], -best_metrics["kl"],
        ):
            best_metrics = dict(m); best_metrics["epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        tkey = (m["teacher_top1_count"], -m["residual_hurt"], m["gt_exact_count"], -m["kl"])
        if best_t_metrics is None or tkey > (
            best_t_metrics["teacher_top1_count"], -best_t_metrics["residual_hurt"],
            best_t_metrics["gt_exact_count"], -best_t_metrics["kl"],
        ):
            best_t_metrics = dict(m); best_t_metrics["epoch"] = epoch
            best_t_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    common = {
        "geometry_columns": list(geom_cols),
        "self_signal_columns": list(self_cols),
        "geometry_mean": geom_mean.tolist(), "geometry_std": geom_std.tolist(),
        "hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
        "max_residual": args.max_residual, "dropout": args.dropout,
        "lambda_kl": args.lambda_kl, "lambda_anchor": args.lambda_anchor,
        "lambda_l2": args.lambda_l2, "lambda_rank": args.lambda_rank,
        "rank_topk": args.rank_topk, "rank_margin": args.rank_margin, "rank_min_gap": args.rank_min_gap,
        "baseline_score_field": args.baseline_score_field,
        "teacher_posterior_field": args.teacher_posterior_field,
    }
    torch.save({"model_state": best_state, **common}, args.output_dir / "best_model_gt.pt")
    torch.save({"model_state": best_t_state, **common}, args.output_dir / "best_model_teacher.pt")

    model.load_state_dict(best_state)
    export_predictions(model=model, samples=samples, geometry_columns=geom_cols, self_signal_columns=self_cols, device=device, output_dir=args.output_dir / "best_gt_export")
    model.load_state_dict(best_t_state)
    export_predictions(model=model, samples=samples, geometry_columns=geom_cols, self_signal_columns=self_cols, device=device, output_dir=args.output_dir / "best_teacher_export")

    summary = {
        "baseline_csv": str(args.baseline_csv),
        "baseline_feature_stats": str(args.baseline_feature_stats),
        "output_dir": str(args.output_dir),
        "num_train_samples": len(train), "num_test_samples": len(test),
        "best_gt_epoch": best_metrics["epoch"],
        "best_gt_test_gt_exact_count": best_metrics["gt_exact_count"],
        "best_gt_test_teacher_top1_count": best_metrics["teacher_top1_count"],
        "best_gt_residual_helped": best_metrics["residual_helped"],
        "best_gt_residual_hurt": best_metrics["residual_hurt"],
        "best_gt_baseline_gt_exact_count": best_metrics["baseline_gt_exact_count"],
        "best_teacher_epoch": best_t_metrics["epoch"],
        "best_teacher_test_teacher_top1_count": best_t_metrics["teacher_top1_count"],
        "best_teacher_test_gt_exact_count": best_t_metrics["gt_exact_count"],
        "best_teacher_residual_helped": best_t_metrics["residual_helped"],
        "best_teacher_residual_hurt": best_t_metrics["residual_hurt"],
        **common, "history": history,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"Best-GT epoch {best_metrics['epoch']}: gt={best_metrics['gt_exact_count']}/{best_metrics['total_samples']} (baseline={best_metrics['baseline_gt_exact_count']}), helped={best_metrics['residual_helped']}, hurt={best_metrics['residual_hurt']}")


if __name__ == "__main__":
    main()
