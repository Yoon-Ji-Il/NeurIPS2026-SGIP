#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_support_rank_features.csv"
)
DEFAULT_BASELINE_CANDIDATE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_aspect_seed_sweep"
    / "seed7"
    / "best_gt_export"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_support_rank_baseline_confidence.csv"
)

OUTPUT_COLUMNS = (
    "baseline_student_rank_best1",
    "baseline_is_student_top1",
    "baseline_top1_score_margin",
    "baseline_top1_posterior_margin",
    "baseline_selected_confidence",
    "baseline_distribution_entropy",
    "baseline_distribution_norm_entropy",
    "baseline_candidate_score_gap_to_top1",
    "baseline_candidate_posterior_gap_to_top1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append per-sample baseline confidence/margin features derived only from "
            "the already-fair baseline student scores. No teacher, GT, want, semantic, "
            "or future fields are used."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--baseline-candidate-csv", type=Path, default=DEFAULT_BASELINE_CANDIDATE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))


def sample_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("split", ""), row.get("sample_id", ""))


def build_confidence_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[sample_key(row)].append(row)

    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (parse_float(row.get("student_score")), row.get("object_id", "")),
            reverse=True,
        )
        n = len(ordered)
        scores = [parse_float(row.get("student_score")) for row in ordered]
        posteriors = [max(parse_float(row.get("student_posterior")), 0.0) for row in ordered]
        posterior_sum = sum(posteriors)
        if posterior_sum <= 0.0:
            posteriors = [1.0 / max(n, 1) for _ in ordered]
        else:
            posteriors = [value / posterior_sum for value in posteriors]

        top_score = scores[0] if scores else 0.0
        second_score = scores[1] if len(scores) > 1 else top_score
        top_posterior = posteriors[0] if posteriors else 0.0
        second_posterior = posteriors[1] if len(posteriors) > 1 else top_posterior
        entropy = -sum(p * math.log(max(p, 1.0e-8)) for p in posteriors)
        norm_entropy = entropy / max(math.log(max(n, 2)), 1.0e-8)

        for rank, row in enumerate(ordered, start=1):
            denominator = max(n - 1, 1)
            rank_best1 = 1.0 - (rank - 1) / denominator
            score = parse_float(row.get("student_score"))
            posterior = max(parse_float(row.get("student_posterior")), 0.0)
            lookup[row_key(row)] = {
                "baseline_student_rank_best1": rank_best1,
                "baseline_is_student_top1": float(rank == 1),
                "baseline_top1_score_margin": top_score - second_score,
                "baseline_top1_posterior_margin": top_posterior - second_posterior,
                "baseline_selected_confidence": top_posterior,
                "baseline_distribution_entropy": entropy,
                "baseline_distribution_norm_entropy": norm_entropy,
                "baseline_candidate_score_gap_to_top1": top_score - score,
                "baseline_candidate_posterior_gap_to_top1": top_posterior - posterior,
            }
    return lookup


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    baseline_rows = read_csv(args.baseline_candidate_csv)
    confidence_lookup = build_confidence_lookup(baseline_rows)
    output_rows: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        output = dict(row)
        features = confidence_lookup.get(row_key(row))
        if features is None:
            missing += 1
            features = {column: 0.0 for column in OUTPUT_COLUMNS}
        for column in OUTPUT_COLUMNS:
            output[column] = features[column]
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"num_rows={len(output_rows)} missing_rows={missing}")
    print("columns=" + ",".join(OUTPUT_COLUMNS))


if __name__ == "__main__":
    main()
