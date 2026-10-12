#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_geometry.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_relative_occupancy.csv"
)
BASE_COLUMN = "scene32768_expanded_bbox_1p1_fraction"
OUTPUT_COLUMNS = (
    "scene32768_expanded_bbox_1p1_fraction_zscore",
    "scene32768_expanded_bbox_1p1_fraction_rank_percentile",
    "scene32768_expanded_bbox_1p1_fraction_max_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append sample-relative occupancy features from expanded bbox occupancy. "
            "Uses only per-sample candidate scene occupancy values; no want, future, GT, or teacher rank."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--base-column", type=str, default=BASE_COLUMN)
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


def parse_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile_ranks(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.5 for _ in values]
    sorted_indices = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(sorted_indices):
        next_index = index + 1
        value = values[sorted_indices[index]]
        while next_index < len(sorted_indices) and values[sorted_indices[next_index]] == value:
            next_index += 1
        # Average tied ranks, normalized to [0, 1].
        avg_rank = (index + next_index - 1) * 0.5
        percentile = avg_rank / max(len(values) - 1, 1)
        for rank_index in range(index, next_index):
            ranks[sorted_indices[rank_index]] = percentile
        index = next_index
    return ranks


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", args.base_column}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[(row.get("split", ""), row.get("sample_id", ""))].append(row_index)

    output_rows = [dict(row) for row in rows]
    for indices in grouped.values():
        values = [parse_float(rows[index].get(args.base_column)) for index in indices]
        count = len(values)
        mean = sum(values) / max(count, 1)
        variance = sum((value - mean) ** 2 for value in values) / max(count, 1)
        std = variance ** 0.5
        max_value = max(values) if values else 0.0
        ranks = percentile_ranks(values)
        for local_index, row_index in enumerate(indices):
            value = values[local_index]
            output_rows[row_index][OUTPUT_COLUMNS[0]] = (value - mean) / max(std, 1.0e-8)
            output_rows[row_index][OUTPUT_COLUMNS[1]] = ranks[local_index]
            output_rows[row_index][OUTPUT_COLUMNS[2]] = value / max(max_value, 1.0e-8)

    fieldnames = list(rows[0].keys())
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"base_column={args.base_column}")
    print(f"features={','.join(OUTPUT_COLUMNS)}")
    print(f"num_samples={len(grouped)}")
    print(f"num_rows={len(output_rows)}")


if __name__ == "__main__":
    main()
