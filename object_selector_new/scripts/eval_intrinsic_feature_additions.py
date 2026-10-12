#!/usr/bin/env python3
"""Evaluate object-intrinsic geometry feature additions over a base feature set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SHAPE_SOURCE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_openins3d_shape.csv"
)
DEFAULT_SAMPLES_JSONL = REPO_ROOT / "data" / "samples.jsonl"

SHAPE_SOURCE_COLUMNS = (
    "object_bbox_height",
    "object_bbox_log_volume",
    "object_bbox_log_xz_area",
    "object_shape_bbox_aspect_max_over_min",
    "object_shape_pca_linearity",
    "object_shape_pca_planarity",
    "object_shape_vertical_spread_ratio",
    "object_shape_bottom_half_fraction",
)

COMPUTED_INTRINSIC_COLUMNS = (
    "intrinsic_bbox_bottom_height",
    "intrinsic_bbox_top_height",
    "intrinsic_bbox_center_height",
    "intrinsic_bbox_top_above_pelvis",
    "intrinsic_bbox_center_above_pelvis",
    "intrinsic_bbox_top_above_head",
)

DEFAULT_EXTRA_FEATURES = SHAPE_SOURCE_COLUMNS + COMPUTED_INTRINSIC_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate +1 intrinsic feature additions on top of a base set.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--shape-source-csv", type=Path, default=DEFAULT_SHAPE_SOURCE_CSV)
    parser.add_argument("--samples-jsonl", type=Path, default=DEFAULT_SAMPLES_JSONL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-features",
        type=str,
        default="root_horizontal_alignment,face_normal_alignment",
    )
    parser.add_argument(
        "--extra-features",
        type=str,
        default=",".join(DEFAULT_EXTRA_FEATURES),
    )
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def parse_feature_list(raw: str) -> tuple[str, ...]:
    cols = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not cols:
        raise ValueError("feature list cannot be empty")
    return cols


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def key_for_row(row: dict[str, str]) -> tuple[str, str, str]:
    split = str(row.get("split", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    object_id = str(row.get("object_id", "")).strip()
    if not split or not sample_id or not object_id:
        raise ValueError(f"Missing join key columns in row: {row}")
    return split, sample_id, object_id


def parse_float_strict(value: str | float | int | None, *, column: str, key: tuple[str, str, str]) -> float:
    if value is None:
        raise ValueError(f"Missing {column!r} for key={key}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty {column!r} for key={key}")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Non-numeric {column!r}={value!r} for key={key}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {column!r}={value!r} for key={key}")
    return number


def parse_vector3(value: Any, *, field_name: str, key: tuple[str, str, str] | tuple[str, str]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be length-3 for key={key}, got {value!r}")
    numbers = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{index}] must be numeric for key={key}, got {item!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{index}] must be finite for key={key}, got {item!r}")
        numbers.append(number)
    return numbers[0], numbers[1], numbers[2]


def build_shape_source_lookup(
    source_csv: Path,
    extra_features: tuple[str, ...],
) -> dict[tuple[str, str, str], dict[str, float]]:
    requested = tuple(column for column in extra_features if column in SHAPE_SOURCE_COLUMNS)
    if not requested:
        return {}
    fieldnames, rows = load_csv_rows(source_csv)
    missing_columns = [column for column in requested if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"shape source csv is missing required columns: {missing_columns}")

    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = key_for_row(row)
        payload = {
            column: parse_float_strict(row.get(column), column=column, key=key)
            for column in requested
        }
        previous = lookup.get(key)
        if previous is not None:
            if previous != payload:
                raise ValueError(f"Conflicting duplicate shape source rows for key={key}")
            continue
        lookup[key] = payload
    return lookup


def compute_intrinsic_candidate_features(
    *,
    split: str,
    sample_id: str,
    object_id: str,
    bbox_min: Any,
    bbox_max: Any,
    pelvis_position: Any,
    head_position: Any,
) -> dict[str, float]:
    key3 = (split, sample_id, object_id)
    low_x, low_y, low_z = parse_vector3(bbox_min, field_name="bbox_min", key=key3)
    high_x, high_y, high_z = parse_vector3(bbox_max, field_name="bbox_max", key=key3)
    low_y, high_y = sorted((low_y, high_y))
    pelvis_y = parse_vector3(pelvis_position, field_name="pelvis_position", key=(split, sample_id))[1]
    head_y = parse_vector3(head_position, field_name="head_position", key=(split, sample_id))[1]

    bottom_y = float(low_y)
    top_y = float(high_y)
    center_y = 0.5 * (bottom_y + top_y)
    return {
        "intrinsic_bbox_bottom_height": bottom_y,
        "intrinsic_bbox_top_height": top_y,
        "intrinsic_bbox_center_height": center_y,
        "intrinsic_bbox_top_above_pelvis": top_y - pelvis_y,
        "intrinsic_bbox_center_above_pelvis": center_y - pelvis_y,
        "intrinsic_bbox_top_above_head": top_y - head_y,
    }


def build_computed_lookup(
    samples_jsonl: Path,
    extra_features: tuple[str, ...],
) -> dict[tuple[str, str, str], dict[str, float]]:
    requested = tuple(column for column in extra_features if column in COMPUTED_INTRINSIC_COLUMNS)
    if not requested:
        return {}

    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    with samples_jsonl.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            split = str(sample.get("split", "")).strip()
            sample_id = str(sample.get("sample_id", "")).strip()
            if not split or not sample_id:
                raise ValueError(f"Missing split/sample_id in samples.jsonl line {line_index}")
            pelvis_position = sample.get("pelvis_position")
            head_position = sample.get("head_position")
            if pelvis_position is None or head_position is None:
                raise ValueError(f"Missing pelvis/head position in samples.jsonl line {line_index}")
            for candidate in sample.get("object_candidates", []):
                object_id = str(candidate.get("object_id", "")).strip()
                if not object_id:
                    raise ValueError(f"Missing object_id in samples.jsonl line {line_index}")
                metadata = candidate.get("metadata", {}) or {}
                bbox_min = metadata.get("bbox_min")
                bbox_max = metadata.get("bbox_max")
                if bbox_min is None or bbox_max is None:
                    raise ValueError(
                        f"Missing bbox metadata for key={(split, sample_id, object_id)} in samples.jsonl line {line_index}"
                    )
                computed = compute_intrinsic_candidate_features(
                    split=split,
                    sample_id=sample_id,
                    object_id=object_id,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    pelvis_position=pelvis_position,
                    head_position=head_position,
                )
                key = (split, sample_id, object_id)
                payload = {column: computed[column] for column in requested}
                previous = lookup.get(key)
                if previous is not None and previous != payload:
                    raise ValueError(f"Conflicting duplicate computed intrinsic features for key={key}")
                lookup[key] = payload
    return lookup


def write_augmented_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    extra_features: tuple[str, ...],
    shape_lookup: dict[tuple[str, str, str], dict[str, float]],
    computed_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> None:
    fieldnames, rows = load_csv_rows(input_csv)
    out_fieldnames = list(fieldnames)
    for column in extra_features:
        if column not in out_fieldnames:
            out_fieldnames.append(column)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            key = key_for_row(row)
            for column in extra_features:
                if column in SHAPE_SOURCE_COLUMNS:
                    payload = shape_lookup.get(key)
                    if payload is None or column not in payload:
                        raise ValueError(f"Missing shape feature {column!r} for key={key}")
                    row[column] = payload[column]
                elif column in COMPUTED_INTRINSIC_COLUMNS:
                    payload = computed_lookup.get(key)
                    if payload is None or column not in payload:
                        raise ValueError(f"Missing computed intrinsic feature {column!r} for key={key}")
                    row[column] = payload[column]
                else:
                    raise ValueError(f"Unknown intrinsic feature column {column!r}")
            writer.writerow(row)


def run_student(
    *,
    input_csv: Path,
    output_dir: Path,
    feature_columns: tuple[str, ...],
    teacher_temperature: float,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "object_selector_new.scripts.train_student",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--feature-columns",
        ",".join(feature_columns),
        "--teacher-temperature",
        str(teacher_temperature),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    print(f"[run] {output_dir.name}: {', '.join(feature_columns)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    with (output_dir / "summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    total = int(summary["num_test_samples"])
    return {
        "run_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "feature_columns": list(feature_columns),
        "best_epoch": int(summary["best_epoch"]),
        "best_teacher_top1_count": int(summary["best_teacher_top1_count"]),
        "best_teacher_top1_pct": float(summary["best_teacher_top1_count"]) / max(total, 1),
        "best_gt_exact_count": int(summary["best_gt_exact_count"]),
        "best_gt_exact_pct": float(summary["best_gt_exact_count"]) / max(total, 1),
        "best_kl": float(summary["best_kl"]),
        "num_test_samples": total,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized = [
        {**row, "feature_columns": ",".join(row["feature_columns"])}
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)


def score_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row["best_gt_exact_count"]),
        int(row["best_teacher_top1_count"]),
        -float(row["best_kl"]),
    )


def main() -> None:
    args = parse_args()
    base_features = parse_feature_list(args.base_features)
    extra_features = parse_feature_list(args.extra_features)

    shape_lookup = build_shape_source_lookup(args.shape_source_csv, extra_features)
    computed_lookup = build_computed_lookup(args.samples_jsonl, extra_features)

    augmented_csv = args.output_dir / "candidates_with_intrinsic_features.csv"
    write_augmented_csv(
        input_csv=args.input_csv,
        output_csv=augmented_csv,
        extra_features=extra_features,
        shape_lookup=shape_lookup,
        computed_lookup=computed_lookup,
    )
    print(f"[info] wrote augmented csv: {augmented_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    baseline = run_student(
        input_csv=augmented_csv,
        output_dir=args.output_dir / "runs" / "base_pair",
        feature_columns=base_features,
        teacher_temperature=args.teacher_temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    baseline["extra_feature"] = ""
    results.append(baseline)

    for feature in extra_features:
        cols = tuple(base_features) + (feature,)
        result = run_student(
            input_csv=augmented_csv,
            output_dir=args.output_dir / "runs" / f"plus_{sanitize_tag(feature)}",
            feature_columns=cols,
            teacher_temperature=args.teacher_temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["extra_feature"] = feature
        results.append(result)

    ranked = sorted(results, key=score_key, reverse=True)
    write_json(args.output_dir / "results.json", ranked)
    write_csv(args.output_dir / "results.csv", ranked)
    write_json(
        args.output_dir / "summary.json",
        {
            "input_csv": str(args.input_csv),
            "shape_source_csv": str(args.shape_source_csv),
            "samples_jsonl": str(args.samples_jsonl),
            "augmented_csv": str(augmented_csv),
            "base_features": list(base_features),
            "extra_features": list(extra_features),
            "epochs": args.epochs,
            "teacher_temperature": args.teacher_temperature,
            "baseline_result": baseline,
            "best_result": ranked[0] if ranked else None,
        },
    )

    print("[done] ranked results")
    for index, row in enumerate(ranked, start=1):
        label = row["extra_feature"] or "BASE"
        print(
            f"{index}. {label}: gt={row['best_gt_exact_count']}/{row['num_test_samples']}, "
            f"teacher={row['best_teacher_top1_count']}/{row['num_test_samples']}, "
            f"kl={row['best_kl']:.6f}"
        )


if __name__ == "__main__":
    main()
