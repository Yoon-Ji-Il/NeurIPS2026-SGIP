#!/usr/bin/env python3
"""Merge and compare fixed-category GIMO intent classifier runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple complete GIMO intent classification runs."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def clean_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key).lstrip("\ufeff").strip(): (
            "" if value is None else str(value).strip()
        )
        for key, value in row.items()
        if key is not None
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(
    path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not text:
        raise ValueError(f"cannot slug model name {value!r}")
    return text


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    payloads: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for run_dir in args.run_dirs:
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "intent_predictions.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "complete":
            raise ValueError(f"run is not complete: {run_dir}")
        model_name = str(metrics.get("model_name", "")).strip()
        if not model_name:
            raise ValueError(f"missing model_name: {metrics_path}")
        if model_name in seen_models:
            raise ValueError(f"duplicate model run: {model_name}")
        seen_models.add(model_name)
        payloads.append(
            {
                "run_dir": run_dir,
                "metrics": metrics,
                "model_name": model_name,
                "slug": slug(model_name),
                "rows": read_csv(predictions_path),
            }
        )
    if not payloads:
        raise ValueError("no run directories provided")

    base_rows = payloads[0]["rows"]
    identity_fields = [
        "row_index",
        "sample_uid",
        "action",
        "object",
        "gt_intent",
    ]
    base_identity = [
        tuple(row[field] for field in identity_fields) for row in base_rows
    ]
    for payload in payloads[1:]:
        identity = [
            tuple(row[field] for field in identity_fields)
            for row in payload["rows"]
        ]
        if identity != base_identity:
            raise ValueError(f"row identity/order mismatch: {payload['run_dir']}")

    base_fields = [
        "row_index",
        "row_id",
        "split",
        "sample_id",
        "sample_uid",
        "video_name",
        "action",
        "object",
        "action_object_input",
        "gt_intent",
    ]
    model_fields: list[str] = []
    comparison_rows: list[dict[str, Any]] = []
    for row_index, base_row in enumerate(base_rows):
        output = {field: base_row[field] for field in base_fields}
        predictions: list[str] = []
        correct_models: list[str] = []
        for payload in payloads:
            model_row = payload["rows"][row_index]
            model_slug = payload["slug"]
            prediction = model_row["selected_intent"]
            predictions.append(prediction)
            if as_bool(model_row["is_correct"]):
                correct_models.append(payload["model_name"])
            output[f"{model_slug}_selected_intent"] = prediction
            output[f"{model_slug}_is_correct"] = model_row["is_correct"]
            output[f"{model_slug}_confidence"] = model_row["confidence"]
            output[f"{model_slug}_reason"] = model_row["reason"]
        output["all_models_agree"] = len(set(predictions)) == 1
        output["correct_model_count"] = len(correct_models)
        output["correct_models"] = " | ".join(correct_models)
        output["all_models_correct"] = len(correct_models) == len(payloads)
        output["no_model_correct"] = len(correct_models) == 0
        comparison_rows.append(output)

    for payload in payloads:
        model_slug = payload["slug"]
        model_fields.extend(
            [
                f"{model_slug}_selected_intent",
                f"{model_slug}_is_correct",
                f"{model_slug}_confidence",
                f"{model_slug}_reason",
            ]
        )
    comparison_fields = base_fields + model_fields + [
        "all_models_agree",
        "correct_model_count",
        "correct_models",
        "all_models_correct",
        "no_model_correct",
    ]
    write_csv(
        args.output_dir / "intent_category_model_comparison.csv",
        comparison_rows,
        comparison_fields,
    )
    disagreement_rows = [
        row for row in comparison_rows if not bool(row["all_models_agree"])
    ]
    any_error_rows = [
        row for row in comparison_rows if not bool(row["all_models_correct"])
    ]
    write_csv(
        args.output_dir / "model_disagreements.csv",
        disagreement_rows,
        comparison_fields,
    )
    write_csv(
        args.output_dir / "any_model_errors.csv",
        any_error_rows,
        comparison_fields,
    )

    summary_rows: list[dict[str, Any]] = []
    for payload in payloads:
        metrics = payload["metrics"]
        summary_rows.append(
            {
                "model_name": payload["model_name"],
                "rows": metrics["rows"],
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "macro_f1_all_16": metrics["macro_f1_all_16"],
                "macro_f1_supported_intents": metrics[
                    "macro_f1_supported_intents"
                ],
                "train_rows": metrics["split_metrics"]["train"]["rows"],
                "train_correct": metrics["split_metrics"]["train"]["correct"],
                "train_accuracy": metrics["split_metrics"]["train"]["accuracy"],
                "test_rows": metrics["split_metrics"]["test"]["rows"],
                "test_correct": metrics["split_metrics"]["test"]["correct"],
                "test_accuracy": metrics["split_metrics"]["test"]["accuracy"],
                "run_dir": str(payload["run_dir"]),
            }
        )
    summary_fields = [
        "model_name",
        "rows",
        "correct",
        "accuracy",
        "macro_f1_all_16",
        "macro_f1_supported_intents",
        "train_rows",
        "train_correct",
        "train_accuracy",
        "test_rows",
        "test_correct",
        "test_accuracy",
        "run_dir",
    ]
    write_csv(
        args.output_dir / "intent_category_model_summary.csv",
        summary_rows,
        summary_fields,
    )

    model_names = [payload["model_name"] for payload in payloads]
    pairwise_rows: list[dict[str, Any]] = []
    for left in payloads:
        output: dict[str, Any] = {"model_name": left["model_name"]}
        for right in payloads:
            agreements = [
                left["rows"][index]["selected_intent"]
                == right["rows"][index]["selected_intent"]
                for index in range(len(base_rows))
            ]
            output[slug(right["model_name"])] = safe_div(
                sum(agreements), len(agreements)
            )
        pairwise_rows.append(output)
    write_csv(
        args.output_dir / "pairwise_model_agreement.csv",
        pairwise_rows,
        ["model_name"] + [slug(name) for name in model_names],
    )

    joint_outcomes: dict[str, int] = {}
    if len(payloads) == 2:
        left, right = payloads
        counts = {
            "both_correct": 0,
            f"{left['slug']}_only_correct": 0,
            f"{right['slug']}_only_correct": 0,
            "both_wrong_same_prediction": 0,
            "both_wrong_different_prediction": 0,
        }
        for index in range(len(base_rows)):
            left_correct = as_bool(left["rows"][index]["is_correct"])
            right_correct = as_bool(right["rows"][index]["is_correct"])
            if left_correct and right_correct:
                counts["both_correct"] += 1
            elif left_correct:
                counts[f"{left['slug']}_only_correct"] += 1
            elif right_correct:
                counts[f"{right['slug']}_only_correct"] += 1
            elif (
                left["rows"][index]["selected_intent"]
                == right["rows"][index]["selected_intent"]
            ):
                counts["both_wrong_same_prediction"] += 1
            else:
                counts["both_wrong_different_prediction"] += 1
        joint_outcomes = counts

    correct_count_distribution = Counter(
        int(row["correct_model_count"]) for row in comparison_rows
    )
    all_agree_count = sum(
        bool(row["all_models_agree"]) for row in comparison_rows
    )
    best_accuracy = max(
        (float(payload["metrics"]["accuracy"]) for payload in payloads),
        default=0.0,
    )
    best_models = [
        payload["model_name"]
        for payload in payloads
        if float(payload["metrics"]["accuracy"]) == best_accuracy
    ]

    write_json(
        args.output_dir / "summary.json",
        {
            "models": model_names,
            "rows": len(base_rows),
            "gt_column": "want",
            "joint_outcomes": joint_outcomes,
            "correct_model_count_distribution": {
                str(count): correct_count_distribution.get(count, 0)
                for count in range(len(payloads) + 1)
            },
            "all_models_agree_count": all_agree_count,
            "all_models_agree_rate": safe_div(
                all_agree_count, len(comparison_rows)
            ),
            "model_disagreement_count": len(disagreement_rows),
            "any_model_error_count": len(any_error_rows),
            "best_accuracy": best_accuracy,
            "best_models": best_models,
            "files": {
                "row_comparison_csv": str(
                    args.output_dir / "intent_category_model_comparison.csv"
                ),
                "model_summary_csv": str(
                    args.output_dir / "intent_category_model_summary.csv"
                ),
                "pairwise_agreement_csv": str(
                    args.output_dir / "pairwise_model_agreement.csv"
                ),
                "model_disagreements_csv": str(
                    args.output_dir / "model_disagreements.csv"
                ),
                "any_model_errors_csv": str(
                    args.output_dir / "any_model_errors.csv"
                ),
            },
        },
    )
    print(f"Wrote {args.output_dir / 'intent_category_model_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'intent_category_model_summary.csv'}")
    print(f"Wrote {args.output_dir / 'pairwise_model_agreement.csv'}")
    print(f"Wrote {args.output_dir / 'model_disagreements.csv'}")
    print(f"Wrote {args.output_dir / 'any_model_errors.csv'}")
    print("\nMODEL RESULTS")
    print(
        f"{'model':<18} {'overall':>10} {'train':>10} "
        f"{'test':>10} {'macro-F1':>10}"
    )
    for row in summary_rows:
        print(
            f"{row['model_name']:<18} "
            f"{100 * float(row['accuracy']):>9.2f}% "
            f"{100 * float(row['train_accuracy']):>9.2f}% "
            f"{100 * float(row['test_accuracy']):>9.2f}% "
            f"{float(row['macro_f1_all_16']):>10.4f}"
        )
    print(
        f"\nRows where model predictions differ: "
        f"{len(disagreement_rows)}/{len(comparison_rows)}"
    )


if __name__ == "__main__":
    main()
