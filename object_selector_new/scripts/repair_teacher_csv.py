"""Merge selected repair samples into an incomplete long-format teacher CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from object_selector_new.scripts.export_teacher_selection_summary import export_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly merge rerun sample rows into an incomplete teacher CSV."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--base-teacher-csv", type=Path, required=True)
    parser.add_argument("--repair-teacher-csv", type=Path, required=True)
    parser.add_argument("--missing-indices", type=str, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    return parser.parse_args()


def parse_indices(raw: str) -> frozenset[int]:
    indices = frozenset(int(part.strip()) for part in raw.split(",") if part.strip())
    if not indices:
        raise ValueError("--missing-indices must be non-empty")
    if any(index < 0 for index in indices):
        raise ValueError("--missing-indices values must be >= 0")
    return indices


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def consume_sample_group(
    rows: list[dict[str, str]],
    offset: int,
    sample: dict[str, Any],
    *,
    sample_index: int,
    source_name: str,
) -> tuple[list[dict[str, str]], int]:
    candidates = list(sample.get("object_candidates", []))
    count = len(candidates)
    group = rows[offset : offset + count]
    if len(group) != count:
        raise ValueError(
            f"{source_name} ended early at sample_index={sample_index}: "
            f"expected {count} candidate rows, found {len(group)}"
        )
    expected_sample_id = str(sample.get("sample_id", ""))
    expected_ids = [str(candidate.get("object_id", "")) for candidate in candidates]
    actual_ids = [str(row.get("object_id", "")) for row in group]
    if any(row.get("sample_id") != expected_sample_id for row in group):
        raise ValueError(
            f"{source_name} sample_id mismatch at sample_index={sample_index}"
        )
    if actual_ids != expected_ids:
        raise ValueError(
            f"{source_name} candidate order mismatch at sample_index={sample_index}"
        )
    return group, offset + count


def main() -> None:
    args = parse_args()
    missing = parse_indices(args.missing_indices)
    samples = read_jsonl(args.input_jsonl)
    if max(missing) >= len(samples):
        raise ValueError(
            f"missing index {max(missing)} is outside input range 0..{len(samples) - 1}"
        )

    base_fields, base_rows = read_csv(args.base_teacher_csv)
    repair_fields, repair_rows = read_csv(args.repair_teacher_csv)
    if repair_fields != base_fields:
        raise ValueError("base and repair CSV field lists differ")

    merged_rows: list[dict[str, str]] = []
    base_offset = 0
    repair_offset = 0
    for sample_index, sample in enumerate(samples):
        if sample_index in missing:
            group, repair_offset = consume_sample_group(
                repair_rows,
                repair_offset,
                sample,
                sample_index=sample_index,
                source_name="repair CSV",
            )
        else:
            group, base_offset = consume_sample_group(
                base_rows,
                base_offset,
                sample,
                sample_index=sample_index,
                source_name="base CSV",
            )
        merged_rows.extend(group)

    if base_offset != len(base_rows):
        raise ValueError(f"base CSV has {len(base_rows) - base_offset} unconsumed rows")
    if repair_offset != len(repair_rows):
        raise ValueError(f"repair CSV has {len(repair_rows) - repair_offset} unconsumed rows")

    expected_rows = sum(len(sample.get("object_candidates", [])) for sample in samples)
    if len(merged_rows) != expected_rows:
        raise ValueError(
            f"merged row count mismatch: expected {expected_rows}, found {len(merged_rows)}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    metrics = export_summary(
        input_jsonl=args.input_jsonl,
        teacher_csv=args.output_csv,
        output_csv=args.summary_csv,
        metrics_json=args.metrics_json,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
