#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
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
    / "teacher_candidate_scores_scene32768_category_onehot.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append train-vocabulary category one-hot features. "
            "This uses static object category labels only; no want, future, GT, or teacher rank."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--category-column", type=str, default="category")
    parser.add_argument("--min-train-count", type=int, default=1)
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


def sanitize_category(category: str) -> str:
    clean = category.strip().lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown"


def unique_column_names(categories: list[str]) -> dict[str, str]:
    used: set[str] = set()
    output: dict[str, str] = {}
    for category in categories:
        base = f"category_onehot_{sanitize_category(category)}"
        column = base
        suffix = 2
        while column in used:
            column = f"{base}_{suffix}"
            suffix += 1
        used.add(column)
        output[category] = column
    other = "category_onehot_other"
    while other in used:
        other += "_"
    output["__other__"] = other
    return output


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id", args.category_column}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    train_counts = Counter(
        row.get(args.category_column, "").strip()
        for row in rows
        if row.get("split", "") == "train"
    )
    train_categories = sorted(
        category
        for category, count in train_counts.items()
        if category and count >= int(args.min_train_count)
    )
    column_by_category = unique_column_names(train_categories)
    onehot_columns = [column_by_category[category] for category in train_categories]
    onehot_columns.append(column_by_category["__other__"])

    output_rows: list[dict[str, Any]] = []
    unseen_test_categories: set[str] = set()
    for row in rows:
        category = row.get(args.category_column, "").strip()
        output = dict(row)
        active_column = column_by_category.get(category, column_by_category["__other__"])
        if row.get("split", "") == "test" and category not in column_by_category:
            unseen_test_categories.add(category)
        for column in onehot_columns:
            output[column] = 1.0 if column == active_column else 0.0
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in onehot_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)

    meta = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "category_column": args.category_column,
        "min_train_count": int(args.min_train_count),
        "num_train_categories": len(train_categories),
        "train_categories": train_categories,
        "feature_columns": onehot_columns,
        "other_column": column_by_category["__other__"],
        "unseen_test_categories": sorted(category for category in unseen_test_categories if category),
    }
    meta_path = args.output_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {args.output_csv}")
    print(f"wrote {meta_path}")
    print(f"num_train_categories={len(train_categories)}")
    print(f"num_onehot_features={len(onehot_columns)}")
    print(f"other_column={column_by_category['__other__']}")
    print(f"unseen_test_categories={','.join(meta['unseen_test_categories'])}")


if __name__ == "__main__":
    main()
