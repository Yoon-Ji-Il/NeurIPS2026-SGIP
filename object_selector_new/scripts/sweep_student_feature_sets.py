#!/usr/bin/env python3
"""Run student feature ablations and small feature-set searches."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.student.model import DEFAULT_FEATURE_COLUMNS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep student feature sets.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-topk", type=int, default=6)
    parser.add_argument("--greedy-max-size", type=int, default=6)
    parser.add_argument("--stop-greedy-on-no-improve", action="store_true")
    return parser.parse_args()


def score_key(result: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(result["best_gt_exact_count"]),
        int(result["best_teacher_top1_count"]),
        -float(result["best_kl"]),
    )


def sanitize_tag(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def run_student(
    *,
    input_csv: Path,
    output_root: Path,
    feature_columns: tuple[str, ...],
    teacher_temperature: float,
    device: str,
    epochs: int,
    batch_size: int,
    run_name: str,
) -> dict[str, Any]:
    run_dir = output_root / run_name
    command = [
        sys.executable,
        "-m",
        "object_selector_new.scripts.train_student",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(run_dir),
        "--feature-columns",
        ",".join(feature_columns),
        "--teacher-temperature",
        str(teacher_temperature),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    print(f"[run] {run_name}: {', '.join(feature_columns)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    with (run_dir / "summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    total = int(summary["num_test_samples"])
    return {
        "run_name": run_name,
        "feature_columns": list(feature_columns),
        "num_features": len(feature_columns),
        "best_epoch": int(summary["best_epoch"]),
        "best_teacher_top1_count": int(summary["best_teacher_top1_count"]),
        "best_teacher_top1_pct": float(summary["best_teacher_top1_count"]) / max(total, 1),
        "best_gt_exact_count": int(summary["best_gt_exact_count"]),
        "best_gt_exact_pct": float(summary["best_gt_exact_count"]) / max(total, 1),
        "best_kl": float(summary["best_kl"]),
        "num_test_samples": total,
        "summary_path": str(run_dir / "summary.json"),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    single_results: list[dict[str, Any]] = []

    for feature in DEFAULT_FEATURE_COLUMNS:
        cols = (feature,)
        run_name = f"single_{sanitize_tag(feature)}"
        result = run_student(
            input_csv=args.input_csv,
            output_root=output_dir / "runs",
            feature_columns=cols,
            teacher_temperature=args.teacher_temperature,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            run_name=run_name,
        )
        cache[cols] = result
        single_results.append(result)

    single_results.sort(key=score_key, reverse=True)
    write_json(output_dir / "single_feature_results.json", single_results)
    write_csv(
        output_dir / "single_feature_results.csv",
        [
            {
                **row,
                "feature_columns": ",".join(row["feature_columns"]),
            }
            for row in single_results
        ],
    )

    top_single_features = [
        tuple(row["feature_columns"])[0]
        for row in single_results[: max(int(args.pair_topk), 0)]
    ]
    print(f"[info] top singles for pair sweep: {top_single_features}")

    pair_results: list[dict[str, Any]] = []
    for feature_a, feature_b in itertools.combinations(top_single_features, 2):
        cols = tuple(sorted((feature_a, feature_b)))
        run_name = f"pair_{sanitize_tag(feature_a)}__{sanitize_tag(feature_b)}"
        result = run_student(
            input_csv=args.input_csv,
            output_root=output_dir / "runs",
            feature_columns=cols,
            teacher_temperature=args.teacher_temperature,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            run_name=run_name,
        )
        cache[cols] = result
        pair_results.append(result)

    pair_results.sort(key=score_key, reverse=True)
    write_json(output_dir / "pair_feature_results.json", pair_results)
    write_csv(
        output_dir / "pair_feature_results.csv",
        [
            {
                **row,
                "feature_columns": ",".join(row["feature_columns"]),
            }
            for row in pair_results
        ],
    )

    greedy_path: list[dict[str, Any]] = []
    current = (top_single_features[0],) if top_single_features else ()
    best_current = cache.get(current)
    if best_current is not None:
        greedy_path.append(best_current)
    while current and len(current) < max(int(args.greedy_max_size), 1):
        candidates: list[dict[str, Any]] = []
        for feature in DEFAULT_FEATURE_COLUMNS:
            if feature in current:
                continue
            cols = tuple(sorted((*current, feature)))
            cached = cache.get(cols)
            if cached is None:
                run_name = f"greedy_{len(cols)}f__{'__'.join(sanitize_tag(c) for c in cols)}"
                cached = run_student(
                    input_csv=args.input_csv,
                    output_root=output_dir / "runs",
                    feature_columns=cols,
                    teacher_temperature=args.teacher_temperature,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    run_name=run_name,
                )
                cache[cols] = cached
            candidates.append(cached)
        if not candidates:
            break
        candidates.sort(key=score_key, reverse=True)
        best_next = candidates[0]
        if args.stop_greedy_on_no_improve and best_current is not None and score_key(best_next) <= score_key(best_current):
            break
        greedy_path.append(best_next)
        current = tuple(best_next["feature_columns"])
        best_current = best_next

    write_json(output_dir / "greedy_feature_path.json", greedy_path)
    write_csv(
        output_dir / "greedy_feature_path.csv",
        [
            {
                **row,
                "feature_columns": ",".join(row["feature_columns"]),
            }
            for row in greedy_path
        ],
    )

    summary = {
        "input_csv": str(args.input_csv),
        "teacher_temperature": args.teacher_temperature,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "default_feature_columns": list(DEFAULT_FEATURE_COLUMNS),
        "top_single_feature": single_results[0] if single_results else None,
        "top_pair_feature_set": pair_results[0] if pair_results else None,
        "greedy_best_feature_set": max(greedy_path, key=score_key) if greedy_path else None,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[done] wrote sweep outputs to {output_dir}")


if __name__ == "__main__":
    main()
