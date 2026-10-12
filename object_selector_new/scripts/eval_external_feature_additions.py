#!/usr/bin/env python3
"""Evaluate external geometry/input/support feature additions over a base feature set."""

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


DEFAULT_SOURCE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_support_context_geometry.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate +1 external feature additions on top of a base set.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-features",
        type=str,
        default="root_horizontal_alignment,face_normal_alignment",
    )
    parser.add_argument(
        "--extra-features",
        type=str,
        default=(
            "object_support_on_larger_overlap_max,"
            "object_support_on_larger_size_ratio_log,"
            "object_support_on_larger_vertical_gap_abs_min,"
            "object_support_on_larger_score,"
            "object_support_child_on_top_log_count,"
            "object_support_child_on_top_overlap_sum,"
            "object_support_child_on_top_volume_ratio_sum,"
            "object_support_max_child_score,"
            "input_torso_object_forward_coord,"
            "input_torso_object_lateral_coord,"
            "obs_hand_last_bbox_distance_3d,"
            "input_hand_heading_object_fraction,"
            "input_face_ray_perp_mean,"
            "input_head_object_cone_fraction"
        ),
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


def parse_float_strict(value: str | None, *, column: str, key: tuple[str, str, str]) -> float:
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


def build_source_lookup(source_csv: Path, extra_features: tuple[str, ...]) -> dict[tuple[str, str, str], dict[str, float]]:
    fieldnames, rows = load_csv_rows(source_csv)
    missing_columns = [column for column in extra_features if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"source csv is missing required columns: {missing_columns}")

    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = key_for_row(row)
        payload = {
            column: parse_float_strict(row.get(column), column=column, key=key)
            for column in extra_features
        }
        previous = lookup.get(key)
        if previous is not None:
            if previous != payload:
                raise ValueError(f"Conflicting duplicate source rows for key={key}")
            continue
        lookup[key] = payload
    return lookup


def write_augmented_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    source_lookup: dict[tuple[str, str, str], dict[str, float]],
    extra_features: tuple[str, ...],
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
            source_payload = source_lookup.get(key)
            if source_payload is None:
                raise ValueError(f"Missing source feature row for key={key}")
            for column in extra_features:
                row[column] = source_payload[column]
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

    source_lookup = build_source_lookup(args.source_csv, extra_features)
    augmented_csv = args.output_dir / "candidates_with_extra_features.csv"
    write_augmented_csv(
        input_csv=args.input_csv,
        output_csv=augmented_csv,
        source_lookup=source_lookup,
        extra_features=extra_features,
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
            "source_csv": str(args.source_csv),
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
