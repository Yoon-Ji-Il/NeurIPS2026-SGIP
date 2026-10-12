#!/usr/bin/env python3
"""Collect nine eval logs and report mean/std over the three seeds."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = REPO_ROOT / "new_runs"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVAL_ROOT / "eval_student_testonly_mask_logit_gaussian_summary"
)
SEVERITIES = ("mild", "moderate", "severe")
SEEDS = (0, 1, 2)
METRICS = (
    "loss_total",
    "loss_trans",
    "loss_des_trans",
    "mpjpe",
    "des_mpjpe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_metric_dict(path: Path) -> dict[str, float]:
    candidates: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and all(key in value for key in METRICS):
            candidates.append(value)
    if not candidates:
        raise ValueError(f"No eval metric dictionary found in {path}")
    return {key: float(candidates[-1][key]) for key in METRICS}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    output_root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for severity in SEVERITIES:
        for seed in SEEDS:
            condition = f"mask_logit_gaussian_{severity}_seed{seed}"
            run_dir = (
                eval_root / f"eval_student_testonly_{condition}"
            )
            log_path = run_dir / "eval.log"
            if not log_path.is_file():
                raise FileNotFoundError(log_path)
            metrics = read_metric_dict(log_path)
            rows.append(
                {
                    "condition": condition,
                    "severity": severity,
                    "sigma_logit": {
                        "mild": 0.5,
                        "moderate": 1.0,
                        "severe": 2.0,
                    }[severity],
                    "seed": seed,
                    **metrics,
                }
            )

    aggregate: list[dict[str, Any]] = []
    for severity in SEVERITIES:
        severity_rows = [
            row for row in rows if row["severity"] == severity
        ]
        output: dict[str, Any] = {
            "severity": severity,
            "sigma_logit": severity_rows[0]["sigma_logit"],
            "num_seeds": len(severity_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in severity_rows]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregate.append(output)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "per_condition_metrics.csv", rows)
    write_csv(output_root / "sigma_mean_std.csv", aggregate)
    with (output_root / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"per_condition": rows, "sigma_mean_std": aggregate},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
