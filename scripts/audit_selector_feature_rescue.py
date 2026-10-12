#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "best_train_gt_export"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "feature_rescue_audit_seed7"
    / "baseline16"
)
DEFAULT_FEATURE_CSVS = (
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_support_context_geometry.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_bbox_aspect_extras.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_morphology_motion_interactions.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_input_motion_bbox_align_bodyface_proxy_16f.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_input_motion_temporal_attention6.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_baseline10_observed_hand_reach.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_scene32768_observed_trajectory.csv",
    REPO_ROOT / "object_selector" / "same_space_feature_inputs" / "teacher_candidate_scores_scene32768_observed_ray_endpoint.csv",
)

ID_COLUMNS = {"split", "sample_id", "object_id"}
DEFAULT_EXACT_BLOCKLIST = {
    "split",
    "sample_id",
    "object_id",
    "name",
    "category",
    "want",
    "match",
    "ease",
    "posterior",
    "reasoning_trace",
    "in_candidate_pool",
    "teacher_score",
    "teacher_posterior",
    "student_score",
    "student_posterior",
    "baseline_score",
    "baseline_posterior",
    "baseline_student_score",
    "baseline_student_posterior",
    "residual",
    "teacher_rank",
    "student_rank",
    "is_gt",
    "is_teacher_selected",
    "is_student_selected",
    "is_semantic_tie",
    "is_final",
    "semantic_total",
    "fusion_total",
    "masked_fusion_total",
    "face_alignment_norm",
    "body_alignment_norm",
    "face_line_score",
    "future_hand_contact_score",
    "future_root_endpoint_score",
}
DEFAULT_SUBSTRING_BLOCKLIST = (
    "semantic",
    "future",
    "want",
    "gaze",
    "qwen",
    "teacher_",
    "student_",
    "baseline_",
    "posterior",
    "reasoning",
    "fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit candidate features for teacher-rescue potential. A useful feature should separate "
            "teacher-selected objects from the baseline top-1 on baseline teacher-miss samples, while "
            "not destabilizing samples where baseline already matches teacher."
        )
    )
    parser.add_argument("--baseline-candidate-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--feature-csv", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-valid-train-miss", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument(
        "--exclude-substrings",
        type=str,
        default=",".join(DEFAULT_SUBSTRING_BLOCKLIST),
        help="Comma-separated substrings excluded from audited feature names.",
    )
    return parser.parse_args()


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
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        value_float = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def is_blocked_feature(column: str, exclude_substrings: tuple[str, ...]) -> bool:
    if column in DEFAULT_EXACT_BLOCKLIST:
        return True
    if column.startswith("is_"):
        return True
    lowered = column.lower()
    return any(pattern and pattern in lowered for pattern in exclude_substrings)


def selected_row(rows: list[dict[str, str]], flag: str, fallback_score: str | None = None) -> dict[str, str]:
    selected = [row for row in rows if row.get(flag, "").strip() in {"1", "true", "True"}]
    if selected:
        return selected[0]
    if fallback_score is None:
        raise ValueError(f"Missing selected row for {flag}")
    return max(rows, key=lambda row: parse_float(row.get(fallback_score)) or -float("inf"))


def load_feature_lookup(
    baseline_rows: list[dict[str, str]],
    feature_paths: list[Path],
    exclude_substrings: tuple[str, ...],
) -> tuple[dict[tuple[str, str, str], dict[str, float]], set[str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    columns: set[str] = set()
    source_by_column: dict[str, str] = {}
    for source_name, rows in [("baseline", baseline_rows)]:
        for row in rows:
            key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
            if not all(key):
                continue
            for column, value in row.items():
                if is_blocked_feature(column, exclude_substrings):
                    continue
                parsed = parse_float(value)
                if parsed is None:
                    continue
                lookup[key][column] = parsed
                columns.add(column)
                source_by_column.setdefault(column, source_name)

    for path in feature_paths:
        if not path.exists():
            print(f"[warn] missing feature csv: {path}", file=sys.stderr)
            continue
        for row in read_csv(path):
            key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
            if not all(key):
                continue
            for column, value in row.items():
                if column in ID_COLUMNS or is_blocked_feature(column, exclude_substrings):
                    continue
                parsed = parse_float(value)
                if parsed is None:
                    continue
                lookup[key][column] = parsed
                columns.add(column)
                source_by_column[column] = path.name
    return lookup, columns, source_by_column


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def hit_rate(values: list[float], epsilon: float) -> float:
    if not values:
        return 0.0
    return sum(value > epsilon for value in values) / len(values)


def signed_stats(values: list[float], sign: float, epsilon: float) -> tuple[float, float]:
    signed = [sign * value for value in values]
    return hit_rate(signed, epsilon), mean(signed)


def best_sign(train_diffs: list[float], epsilon: float) -> float:
    pos_hits = sum(value > epsilon for value in train_diffs)
    neg_hits = sum(-value > epsilon for value in train_diffs)
    if pos_hits == neg_hits:
        return 1.0 if mean(train_diffs) >= 0.0 else -1.0
    return 1.0 if pos_hits > neg_hits else -1.0


def competitor_risk(
    rows: list[dict[str, str]],
    base: dict[str, str],
    lookup: dict[tuple[str, str, str], dict[str, float]],
    column: str,
    sign: float,
    epsilon: float,
) -> tuple[int, float]:
    key_base = (base.get("split", ""), base.get("sample_id", ""), base.get("object_id", ""))
    base_value = lookup.get(key_base, {}).get(column)
    if base_value is None:
        return 0, 0.0
    max_margin = -float("inf")
    for row in rows:
        if row.get("object_id") == base.get("object_id"):
            continue
        key_other = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        other_value = lookup.get(key_other, {}).get(column)
        if other_value is None:
            continue
        max_margin = max(max_margin, sign * (other_value - base_value))
    if max_margin == -float("inf"):
        return 0, 0.0
    return int(max_margin > epsilon), max_margin


def audit_features(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    lookup: dict[tuple[str, str, str], dict[str, float]],
    columns: set[str],
    source_by_column: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    miss_pairs: list[tuple[str, str, list[dict[str, str]], dict[str, str], dict[str, str], dict[str, str]]] = []
    correct_pairs: list[tuple[str, str, list[dict[str, str]], dict[str, str], dict[str, str]]] = []
    gt_correct_pairs: list[tuple[str, str, list[dict[str, str]], dict[str, str], dict[str, str]]] = []
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "teacher": 0, "gt": 0, "miss": 0})

    for (split, sample_id), rows in grouped.items():
        base = selected_row(rows, "is_student_selected", fallback_score="student_posterior")
        teacher = selected_row(rows, "is_teacher_selected")
        gt = selected_row(rows, "is_gt")
        split_counts[split]["total"] += 1
        split_counts[split]["teacher"] += int(base.get("object_id") == teacher.get("object_id"))
        split_counts[split]["gt"] += int(base.get("object_id") == gt.get("object_id"))
        if base.get("object_id") != teacher.get("object_id"):
            split_counts[split]["miss"] += 1
            miss_pairs.append((split, sample_id, rows, base, teacher, gt))
        else:
            correct_pairs.append((split, sample_id, rows, base, teacher))
        if base.get("object_id") == gt.get("object_id"):
            gt_correct_pairs.append((split, sample_id, rows, base, gt))

    audit_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for column in sorted(columns):
        diffs_by_split: dict[str, list[float]] = defaultdict(list)
        for split, sample_id, _rows, base, teacher, gt in miss_pairs:
            base_key = (split, sample_id, base.get("object_id", ""))
            teacher_key = (split, sample_id, teacher.get("object_id", ""))
            base_value = lookup.get(base_key, {}).get(column)
            teacher_value = lookup.get(teacher_key, {}).get(column)
            if base_value is None or teacher_value is None:
                continue
            diffs_by_split[split].append(teacher_value - base_value)

        train_diffs = diffs_by_split.get("train", [])
        test_diffs = diffs_by_split.get("test", [])
        if len(train_diffs) < args.min_valid_train_miss:
            continue
        sign = best_sign(train_diffs, args.epsilon)
        train_miss_hit, train_miss_margin = signed_stats(train_diffs, sign, args.epsilon)
        test_miss_hit, test_miss_margin = signed_stats(test_diffs, sign, args.epsilon)

        risk_by_split: dict[str, list[int]] = defaultdict(list)
        risk_margin_by_split: dict[str, list[float]] = defaultdict(list)
        gt_risk_by_split: dict[str, list[int]] = defaultdict(list)
        for split, sample_id, rows, base, _teacher in correct_pairs:
            risk, margin = competitor_risk(rows, base, lookup, column, sign, args.epsilon)
            risk_by_split[split].append(risk)
            risk_margin_by_split[split].append(margin)
        for split, sample_id, rows, base, _gt in gt_correct_pairs:
            risk, _margin = competitor_risk(rows, base, lookup, column, sign, args.epsilon)
            gt_risk_by_split[split].append(risk)

        train_teacher_risk = mean(risk_by_split.get("train", []))
        test_teacher_risk = mean(risk_by_split.get("test", []))
        train_gt_risk = mean(gt_risk_by_split.get("train", []))
        test_gt_risk = mean(gt_risk_by_split.get("test", []))
        train_score = train_miss_hit - train_teacher_risk
        test_score = test_miss_hit - test_teacher_risk
        direction = "higher_for_teacher" if sign > 0 else "lower_for_teacher"
        audit_rows.append(
            {
                "feature": column,
                "source": source_by_column.get(column, ""),
                "direction": direction,
                "train_rescue_score": train_score,
                "test_diagnostic_score": test_score,
                "train_miss_hit": train_miss_hit,
                "test_miss_hit": test_miss_hit,
                "train_miss_signed_margin": train_miss_margin,
                "test_miss_signed_margin": test_miss_margin,
                "train_teacher_correct_risk": train_teacher_risk,
                "test_teacher_correct_risk": test_teacher_risk,
                "train_gt_correct_risk": train_gt_risk,
                "test_gt_correct_risk": test_gt_risk,
                "valid_train_miss_pairs": len(train_diffs),
                "valid_test_miss_pairs": len(test_diffs),
                "train_correct_samples": len(risk_by_split.get("train", [])),
                "test_correct_samples": len(risk_by_split.get("test", [])),
                "mean_train_correct_competitor_margin": mean(risk_margin_by_split.get("train", [])),
                "mean_test_correct_competitor_margin": mean(risk_margin_by_split.get("test", [])),
            }
        )

        for split, sample_id, _rows, base, teacher, gt in miss_pairs:
            base_key = (split, sample_id, base.get("object_id", ""))
            teacher_key = (split, sample_id, teacher.get("object_id", ""))
            base_value = lookup.get(base_key, {}).get(column)
            teacher_value = lookup.get(teacher_key, {}).get(column)
            if base_value is None or teacher_value is None:
                continue
            signed_delta = sign * (teacher_value - base_value)
            pair_rows.append(
                {
                    "feature": column,
                    "direction": direction,
                    "split": split,
                    "sample_id": sample_id,
                    "baseline_object_id": base.get("object_id", ""),
                    "baseline_name": base.get("name", ""),
                    "teacher_object_id": teacher.get("object_id", ""),
                    "teacher_name": teacher.get("name", ""),
                    "gt_object_id": gt.get("object_id", ""),
                    "gt_name": gt.get("name", ""),
                    "baseline_value": base_value,
                    "teacher_value": teacher_value,
                    "signed_delta": signed_delta,
                    "would_rescue_pair": int(signed_delta > args.epsilon),
                }
            )

    summary = {
        "baseline_candidate_csv": str(args.baseline_candidate_csv),
        "feature_csv": [str(path) for path in (args.feature_csv or list(DEFAULT_FEATURE_CSVS))],
        "split_counts": split_counts,
        "num_audited_features": len(audit_rows),
        "min_valid_train_miss": args.min_valid_train_miss,
        "epsilon": args.epsilon,
    }
    return audit_rows, pair_rows, summary


def main() -> None:
    args = parse_args()
    feature_paths = args.feature_csv or list(DEFAULT_FEATURE_CSVS)
    exclude_substrings = tuple(part.strip().lower() for part in args.exclude_substrings.split(",") if part.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv(args.baseline_candidate_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        grouped[(row.get("split", ""), row.get("sample_id", ""))].append(row)

    lookup, columns, source_by_column = load_feature_lookup(baseline_rows, feature_paths, exclude_substrings)
    audit_rows, pair_rows, summary = audit_features(grouped, lookup, columns, source_by_column, args)
    if not audit_rows:
        raise RuntimeError("No auditable features survived filtering")

    by_train = sorted(
        audit_rows,
        key=lambda row: (
            float(row["train_rescue_score"]),
            float(row["train_miss_hit"]),
            -float(row["train_teacher_correct_risk"]),
            float(row["test_diagnostic_score"]),
        ),
        reverse=True,
    )
    by_test = sorted(
        audit_rows,
        key=lambda row: (
            float(row["test_diagnostic_score"]),
            float(row["test_miss_hit"]),
            -float(row["test_teacher_correct_risk"]),
            float(row["train_rescue_score"]),
        ),
        reverse=True,
    )
    write_csv(args.output_dir / "feature_rescue_audit_by_train.csv", by_train)
    write_csv(args.output_dir / "feature_rescue_audit_by_test_diagnostic.csv", by_test)
    write_csv(args.output_dir / "feature_pair_deltas.csv", pair_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"wrote {args.output_dir / 'feature_rescue_audit_by_train.csv'}")
    print(f"wrote {args.output_dir / 'feature_rescue_audit_by_test_diagnostic.csv'}")
    print(f"wrote {args.output_dir / 'feature_pair_deltas.csv'}")
    print("Top by train rescue score:")
    for row in by_train[:10]:
        print(
            f"{row['feature']}: train_score={float(row['train_rescue_score']):.3f} "
            f"train_hit={float(row['train_miss_hit']):.3f} "
            f"train_risk={float(row['train_teacher_correct_risk']):.3f} "
            f"test_score={float(row['test_diagnostic_score']):.3f} "
            f"test_hit={float(row['test_miss_hit']):.3f}"
        )
    print("Top by test diagnostic score:")
    for row in by_test[:10]:
        print(
            f"{row['feature']}: test_score={float(row['test_diagnostic_score']):.3f} "
            f"test_hit={float(row['test_miss_hit']):.3f} "
            f"test_risk={float(row['test_teacher_correct_risk']):.3f} "
            f"train_score={float(row['train_rescue_score']):.3f}"
        )


if __name__ == "__main__":
    main()
