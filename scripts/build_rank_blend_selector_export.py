#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_seed_sweep"
    / "seed7"
    / "candidate_scores.csv"
)
DEFAULT_AUX_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_neighborhood_seed7"
    / "contained_by_larger"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_contained_rank_blend_a025_seed7"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend two fair selector exports by per-sample candidate rank score. "
            "This uses only student scores from two already-trained selectors."
        )
    )
    parser.add_argument("--base-candidate-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--aux-candidate-csv", type=Path, default=DEFAULT_AUX_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.25, help="Aux selector rank weight.")
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
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


def parse_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def parse_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))


def sample_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("split", ""), row.get("sample_id", ""))


def rank_scores(rows: list[dict[str, Any]], score_key: str) -> list[float]:
    if len(rows) <= 1:
        return [1.0 for _ in rows]
    scores = [parse_float(row.get(score_key)) for row in rows]
    order = sorted(range(len(rows)), key=lambda index: (scores[index], index))
    ranks = [0.0 for _ in rows]
    for rank, index in enumerate(order):
        ranks[index] = rank / max(len(rows) - 1, 1)
    return ranks


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / max(total, 1.0e-8) for value in exps]


def counts_from_samples(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"teacher": 0, "gt": 0, "total": 0})
    for row in rows:
        split = str(row.get("split", ""))
        counts[split]["teacher"] += parse_int(row.get("is_teacher_match"))
        counts[split]["gt"] += parse_int(row.get("is_gt_exact"))
        counts[split]["total"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    alpha = float(args.alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {alpha}")

    base_rows = read_csv(args.base_candidate_csv)
    aux_rows = read_csv(args.aux_candidate_csv)
    if not base_rows or not aux_rows:
        raise RuntimeError("Both input candidate CSVs must be non-empty")

    aux_by_key = {row_key(row): row for row in aux_rows}
    merged_rows: list[dict[str, Any]] = []
    missing_aux: list[tuple[str, str, str]] = []
    for base_row in base_rows:
        key = row_key(base_row)
        aux_row = aux_by_key.get(key)
        if aux_row is None:
            missing_aux.append(key)
            continue
        row: dict[str, Any] = dict(base_row)
        row["base_student_score"] = parse_float(base_row.get("student_score"))
        row["base_student_posterior"] = parse_float(base_row.get("student_posterior"))
        row["contained_student_score"] = parse_float(aux_row.get("student_score"))
        row["contained_student_posterior"] = parse_float(aux_row.get("student_posterior"))
        merged_rows.append(row)
    if missing_aux:
        raise RuntimeError(f"Missing aux rows for {len(missing_aux)} base candidates, e.g. {missing_aux[:5]}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        grouped[sample_key(row)].append(row)

    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: (row.get("object_id", ""), row.get("name", "")))
        base_ranks = rank_scores(rows, "base_student_score")
        aux_ranks = rank_scores(rows, "contained_student_score")
        blend_scores = [
            (1.0 - alpha) * base_ranks[index] + alpha * aux_ranks[index]
            for index in range(len(rows))
        ]
        blend_posterior = softmax(blend_scores)
        student_order = sorted(range(len(rows)), key=lambda index: blend_scores[index], reverse=True)
        pred_idx = student_order[0]
        teacher_idx = next(
            (index for index, row in enumerate(rows) if parse_int(row.get("is_teacher_selected")) == 1),
            -1,
        )
        gt_idx = next((index for index, row in enumerate(rows) if parse_int(row.get("is_gt")) == 1), -1)
        if teacher_idx < 0:
            teacher_idx = max(range(len(rows)), key=lambda index: parse_float(rows[index].get("teacher_posterior")))
        if gt_idx < 0:
            raise RuntimeError(f"Sample {key} has no GT row")

        rank_by_index = {candidate_index: rank + 1 for rank, candidate_index in enumerate(student_order)}
        for index, row in enumerate(rows):
            output = dict(row)
            output["base_rank_score"] = base_ranks[index]
            output["contained_rank_score"] = aux_ranks[index]
            output["student_score"] = blend_scores[index]
            output["student_posterior"] = blend_posterior[index]
            output["student_rank"] = rank_by_index[index]
            output["is_student_selected"] = int(index == pred_idx)
            output["blend_alpha_contained"] = alpha
            candidate_rows.append(output)

        pred = rows[pred_idx]
        teacher = rows[teacher_idx]
        gt = rows[gt_idx]
        sample_rows.append(
            {
                "split": pred.get("split", ""),
                "sample_id": pred.get("sample_id", ""),
                "selected_object_id": pred.get("object_id", ""),
                "selected_object_name": pred.get("name", ""),
                "teacher_selected_object_id": teacher.get("object_id", ""),
                "gt_object_id": gt.get("object_id", ""),
                "gt_object_name": gt.get("name", ""),
                "num_candidates": len(rows),
                "student_confidence": blend_posterior[pred_idx],
                "teacher_confidence": parse_float(teacher.get("teacher_posterior")),
                "is_teacher_match": int(pred_idx == teacher_idx),
                "is_gt_exact": int(pred_idx == gt_idx),
                "blend_alpha_contained": alpha,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_fieldnames = list(candidate_rows[0].keys())
    sample_fieldnames = list(sample_rows[0].keys())
    write_csv(output_dir / "candidate_scores.csv", candidate_rows, candidate_fieldnames)
    write_csv(output_dir / "sample_summary.csv", sample_rows, sample_fieldnames)

    summary = {
        "mode": "rank_blend_selector_export",
        "base_candidate_csv": str(args.base_candidate_csv),
        "aux_candidate_csv": str(args.aux_candidate_csv),
        "output_dir": str(output_dir),
        "alpha_contained": alpha,
        "num_candidate_rows": len(candidate_rows),
        "num_samples": len(sample_rows),
        "counts": counts_from_samples(sample_rows),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"wrote {output_dir / 'candidate_scores.csv'}")
    print(f"wrote {output_dir / 'sample_summary.csv'}")
    for split, values in sorted(summary["counts"].items()):
        print(
            f"{split}: teacher={values['teacher']}/{values['total']} "
            f"gt={values['gt']}/{values['total']}"
        )


if __name__ == "__main__":
    main()
