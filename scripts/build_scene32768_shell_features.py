#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_expanded_bbox.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_shell.csv"
)
SCALES = ("1p1", "1p25", "1p5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair expanded-bbox shell occupancy features. "
            "Shell fraction is expanded bbox fraction minus original bbox fraction."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
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


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")

    base_column = "scene32768_bbox_fraction"
    required_columns = {base_column}
    required_columns.update(f"scene32768_expanded_bbox_{scale}_fraction" for scale in SCALES)
    missing = sorted(column for column in required_columns if column not in rows[0])
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    output_rows: list[dict[str, Any]] = []
    shell_columns = [f"scene32768_expanded_bbox_{scale}_shell_fraction" for scale in SCALES]
    for row in rows:
        output = dict(row)
        base_fraction = parse_float(row.get(base_column))
        for scale in SCALES:
            expanded_column = f"scene32768_expanded_bbox_{scale}_fraction"
            shell_column = f"scene32768_expanded_bbox_{scale}_shell_fraction"
            shell = parse_float(row.get(expanded_column)) - base_fraction
            output[shell_column] = max(shell, 0.0)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in shell_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"features={','.join(shell_columns)}")


if __name__ == "__main__":
    main()
