#!/usr/bin/env python3
"""Evaluate selected student feature combinations over a base feature set."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selected feature combinations on top of a base set.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-features",
        type=str,
        default="root_horizontal_alignment,face_normal_alignment",
    )
    parser.add_argument(
        "--combo-features",
        type=str,
        required=True,
        help="Semicolon-separated combos, where each combo is a comma-separated feature list.",
    )
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def parse_feature_list(raw: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError("feature list cannot be empty")
    return parts


def parse_combo_sets(raw: str) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for block in raw.split(";"):
        block = block.strip()
        if not block:
            continue
        combo = parse_feature_list(block)
        if combo in seen:
            continue
        seen.add(combo)
        combos.append(combo)
    if not combos:
        raise ValueError("at least one combo must be provided")
    return combos


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def score_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row["best_gt_exact_count"]),
        int(row["best_teacher_top1_count"]),
        -float(row["best_kl"]),
    )


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
    summary_path = output_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    total = int(summary["num_test_samples"])
    return {
        "run_dir": str(output_dir),
        "summary_path": str(summary_path),
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


def main() -> None:
    args = parse_args()
    base_features = parse_feature_list(args.base_features)
    combos = parse_combo_sets(args.combo_features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    baseline = run_student(
        input_csv=args.input_csv,
        output_dir=args.output_dir / "runs" / "base_pair",
        feature_columns=base_features,
        teacher_temperature=args.teacher_temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    baseline["combo_features"] = []
    results.append(baseline)

    for combo in combos:
        cols = tuple(base_features) + combo
        tag = "__".join(sanitize_tag(item) for item in combo)
        result = run_student(
            input_csv=args.input_csv,
            output_dir=args.output_dir / "runs" / f"plus_{tag}",
            feature_columns=cols,
            teacher_temperature=args.teacher_temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["combo_features"] = list(combo)
        results.append(result)

    ranked = sorted(results, key=score_key, reverse=True)
    write_json(args.output_dir / "results.json", ranked)
    write_csv(args.output_dir / "results.csv", ranked)
    write_json(
        args.output_dir / "summary.json",
        {
            "input_csv": str(args.input_csv),
            "base_features": list(base_features),
            "combo_features": [list(combo) for combo in combos],
            "epochs": args.epochs,
            "teacher_temperature": args.teacher_temperature,
            "baseline_result": baseline,
            "best_result": ranked[0] if ranked else None,
        },
    )

    print("[done] ranked results")
    for index, row in enumerate(ranked, start=1):
        label = ",".join(row["combo_features"]) if row["combo_features"] else "BASE"
        print(
            f"{index}. {label}: gt={row['best_gt_exact_count']}/{row['num_test_samples']}, "
            f"teacher={row['best_teacher_top1_count']}/{row['num_test_samples']}, "
            f"kl={row['best_kl']:.6f}"
        )


if __name__ == "__main__":
    main()
