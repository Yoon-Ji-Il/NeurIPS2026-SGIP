#!/usr/bin/env python3
"""Evaluate one-by-one derived feature additions on top of a base feature set."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.derived_geometry_features import DERIVED_GEOMETRY_FEATURES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate derived feature additions on top of a base feature set.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-features",
        type=str,
        default="root_horizontal_alignment,face_normal_alignment",
        help="Comma-separated base feature set.",
    )
    parser.add_argument(
        "--derived-features",
        type=str,
        default=",".join(DERIVED_GEOMETRY_FEATURES.keys()),
        help="Comma-separated subset of derived feature names to evaluate.",
    )
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def parse_feature_list(raw: str) -> tuple[str, ...]:
    columns = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not columns:
        raise ValueError("feature list cannot be empty")
    return columns


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _parse_float(value: str | None, *, column: str, sample_id: str, object_id: str) -> float:
    if value is None:
        raise ValueError(f"Missing {column!r} at sample_id={sample_id} object_id={object_id}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty {column!r} at sample_id={sample_id} object_id={object_id}")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric {column!r}={value!r} at sample_id={sample_id} object_id={object_id}"
        ) from exc


def build_augmented_csv(input_csv: Path, output_csv: Path) -> list[str]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    derived_columns = list(DERIVED_GEOMETRY_FEATURES.keys())
    for column in derived_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            object_id = str(row.get("object_id", "")).strip()
            numeric = {
                key: _parse_float(value, column=key, sample_id=sample_id, object_id=object_id)
                for key, value in row.items()
                if key
                and key
                in {
                    "root_horizontal_alignment",
                    "face_normal_alignment",
                    "root_line_distance_xz",
                    "face_normal_horizontal_alignment",
                    "face_normal_line_distance_xz",
                    "torso_horizontal_alignment",
                }
            }
            missing_inputs = [
                key
                for key in {
                    "root_horizontal_alignment",
                    "face_normal_alignment",
                    "root_line_distance_xz",
                    "face_normal_horizontal_alignment",
                    "face_normal_line_distance_xz",
                    "torso_horizontal_alignment",
                }
                if key not in numeric
            ]
            if missing_inputs:
                raise ValueError(
                    f"Cannot derive features for sample_id={sample_id} object_id={object_id}; "
                    f"missing source columns {missing_inputs}"
                )
            for name, fn in DERIVED_GEOMETRY_FEATURES.items():
                row[name] = fn(numeric)
            writer.writerow(row)
    return derived_columns


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
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        {**row, "feature_columns": ",".join(row["feature_columns"])}
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)


def sort_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row["best_gt_exact_count"]),
        int(row["best_teacher_top1_count"]),
        -float(row["best_kl"]),
    )


def main() -> None:
    args = parse_args()
    base_features = parse_feature_list(args.base_features)
    derived_features = parse_feature_list(args.derived_features)

    unknown = [name for name in derived_features if name not in DERIVED_GEOMETRY_FEATURES]
    if unknown:
        raise ValueError(f"Unknown derived feature(s): {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented_csv = args.output_dir / "candidates_with_derived.csv"
    built_columns = build_augmented_csv(args.input_csv, augmented_csv)
    print(f"[info] wrote augmented CSV with derived columns: {built_columns}")

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
    baseline["derived_feature"] = ""
    results.append(baseline)

    for derived in derived_features:
        feature_columns = tuple(base_features) + (derived,)
        run_dir = args.output_dir / "runs" / f"plus_{sanitize_tag(derived)}"
        result = run_student(
            input_csv=augmented_csv,
            output_dir=run_dir,
            feature_columns=feature_columns,
            teacher_temperature=args.teacher_temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["derived_feature"] = derived
        results.append(result)

    ranked = sorted(results, key=sort_key, reverse=True)
    write_json(args.output_dir / "results.json", ranked)
    write_csv(args.output_dir / "results.csv", ranked)
    write_json(
        args.output_dir / "summary.json",
        {
            "input_csv": str(args.input_csv),
            "augmented_csv": str(augmented_csv),
            "base_features": list(base_features),
            "derived_features": list(derived_features),
            "epochs": args.epochs,
            "teacher_temperature": args.teacher_temperature,
            "best_result": ranked[0] if ranked else None,
            "baseline_result": baseline,
        },
    )
    print("[done] ranked results")
    for index, row in enumerate(ranked, start=1):
        label = row["derived_feature"] or "BASE"
        print(
            f"{index}. {label}: gt={row['best_gt_exact_count']}/{row['num_test_samples']}, "
            f"teacher={row['best_teacher_top1_count']}/{row['num_test_samples']}, "
            f"kl={row['best_kl']:.6f}"
        )


if __name__ == "__main__":
    main()
