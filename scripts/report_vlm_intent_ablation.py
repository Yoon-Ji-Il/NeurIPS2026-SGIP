#!/usr/bin/env python3
"""Report categorical GIMO intent agreement for same-prompt VLM runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


MODEL_METADATA = {
    "qwen3-vl:8b-instruct-q4_k_m": {
        "display_name": "Qwen3-VL-8B-Instruct",
        "parameter_scale": "8B",
        "quantization": "Q4_K_M",
        "family": "Qwen3-VL",
    },
    "qwen/qwen3-vl-8b-instruct": {
        "display_name": "Qwen3-VL-8B-Instruct",
        "parameter_scale": "8B",
        "quantization": "BF16",
        "family": "Qwen3-VL",
    },
    "qwen3-vl:4b-instruct-q4_k_m": {
        "display_name": "Qwen3-VL-4B-Instruct",
        "parameter_scale": "4B",
        "quantization": "Q4_K_M",
        "family": "Qwen3-VL",
    },
    "gemma3:4b-it-q4_k_m": {
        "display_name": "Gemma-3-4B-IT",
        "parameter_scale": "4B",
        "quantization": "Q4_K_M",
        "family": "Gemma 3",
    },
    "huggingfacem4/idefics3-8b-llama3": {
        "display_name": "Idefics3-8B-Llama3",
        "parameter_scale": "8B",
        "quantization": "BF16",
        "family": "Idefics3 / Llama 3.1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed-category GIMO intent predictions against a "
            "Qwen3-VL-8B same-prompt baseline."
        )
    )
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-run-dirs", type=Path, nargs="+", required=True
    )
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
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
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
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise ValueError(f"cannot slug model name {value!r}")
    return result


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def metadata_for(model_name: str) -> dict[str, str]:
    known = MODEL_METADATA.get(model_name.lower())
    if known:
        return dict(known)
    return {
        "display_name": model_name,
        "parameter_scale": "",
        "quantization": "",
        "family": "",
    }


def cohen_kappa(
    baseline_labels: list[str],
    candidate_labels: list[str],
    labels: list[str],
) -> float:
    if len(baseline_labels) != len(candidate_labels):
        raise ValueError("Cohen kappa inputs have different lengths")
    if not baseline_labels:
        return 0.0
    observed = safe_div(
        sum(left == right for left, right in zip(baseline_labels, candidate_labels)),
        len(baseline_labels),
    )
    left_counts = Counter(baseline_labels)
    right_counts = Counter(candidate_labels)
    expected = sum(
        safe_div(left_counts[label], len(baseline_labels))
        * safe_div(right_counts[label], len(candidate_labels))
        for label in labels
    )
    return safe_div(observed - expected, 1.0 - expected)


def total_variation(
    baseline_counts: Counter[str],
    candidate_counts: Counter[str],
    labels: list[str],
    total: int,
) -> float:
    if total <= 0:
        return 0.0
    return 0.5 * sum(
        abs(baseline_counts[label] / total - candidate_counts[label] / total)
        for label in labels
    )


def load_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise ValueError(f"run is not complete: {run_dir}")
    model_name = str(metrics.get("model_name", "")).strip()
    if not model_name:
        raise ValueError(f"missing model_name: {metrics_path}")
    rows = read_csv(run_dir / "intent_predictions.csv")
    unique_rows = read_csv(run_dir / "unique_pair_predictions.csv")
    if len(rows) != int(metrics.get("rows", -1)):
        raise ValueError(f"row count disagrees with metrics: {run_dir}")
    if len(unique_rows) != int(metrics.get("completed_unique_pairs", -1)):
        raise ValueError(f"unique-pair count disagrees with metrics: {run_dir}")
    return {
        "run_dir": run_dir.resolve(),
        "metrics": metrics,
        "manifest": manifest,
        "model_name": model_name,
        "slug": slug(model_name),
        "metadata": metadata_for(model_name),
        "rows": rows,
        "unique_rows": unique_rows,
    }


def identity(rows: list[dict[str, str]], fields: list[str]) -> list[tuple[str, ...]]:
    return [tuple(row[field] for field in fields) for row in rows]


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline_run_dir)
    candidates = [load_run(path) for path in args.candidate_run_dirs]
    runs = [baseline, *candidates]
    if len({run["model_name"] for run in runs}) != len(runs):
        raise ValueError("duplicate model runs were supplied")

    prompt_versions = {run["metrics"].get("prompt_version") for run in runs}
    if len(prompt_versions) != 1:
        raise ValueError(f"prompt-version mismatch: {sorted(prompt_versions)}")
    canonical_lists = {
        tuple(run["metrics"].get("canonical_intents", [])) for run in runs
    }
    if len(canonical_lists) != 1:
        raise ValueError("canonical intent lists differ between runs")
    labels = list(next(iter(canonical_lists)))
    if not labels:
        raise ValueError("canonical intent list is empty")

    row_identity_fields = [
        "row_index",
        "sample_uid",
        "action",
        "object",
        "gt_intent",
    ]
    unique_identity_fields = ["action", "object", "gt_intent"]
    baseline_row_identity = identity(baseline["rows"], row_identity_fields)
    baseline_unique_identity = identity(
        baseline["unique_rows"], unique_identity_fields
    )
    for run in candidates:
        if identity(run["rows"], row_identity_fields) != baseline_row_identity:
            raise ValueError(f"row identity/order mismatch: {run['run_dir']}")
        if (
            identity(run["unique_rows"], unique_identity_fields)
            != baseline_unique_identity
        ):
            raise ValueError(
                f"unique-pair identity/order mismatch: {run['run_dir']}"
            )

    baseline_row_labels = [
        row["selected_intent"] for row in baseline["rows"]
    ]
    baseline_unique_labels = [
        row["selected_intent"] for row in baseline["unique_rows"]
    ]
    baseline_row_counts = Counter(baseline_row_labels)
    baseline_unique_counts = Counter(baseline_unique_labels)

    summary_rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        row_labels = [row["selected_intent"] for row in run["rows"]]
        unique_labels = [
            row["selected_intent"] for row in run["unique_rows"]
        ]
        row_agreement_count = sum(
            left == right
            for left, right in zip(baseline_row_labels, row_labels)
        )
        unique_agreement_count = sum(
            left == right
            for left, right in zip(baseline_unique_labels, unique_labels)
        )
        row_counts = Counter(row_labels)
        unique_counts = Counter(unique_labels)
        baseline_correct_candidate_wrong = sum(
            as_bool(baseline_row["is_correct"])
            and not as_bool(candidate_row["is_correct"])
            for baseline_row, candidate_row in zip(
                baseline["rows"], run["rows"]
            )
        )
        baseline_wrong_candidate_correct = sum(
            not as_bool(baseline_row["is_correct"])
            and as_bool(candidate_row["is_correct"])
            for baseline_row, candidate_row in zip(
                baseline["rows"], run["rows"]
            )
        )
        both_wrong_but_disagree = sum(
            not as_bool(baseline_row["is_correct"])
            and not as_bool(candidate_row["is_correct"])
            and baseline_row["selected_intent"]
            != candidate_row["selected_intent"]
            for baseline_row, candidate_row in zip(
                baseline["rows"], run["rows"]
            )
        )
        metrics = run["metrics"]
        split_metrics = metrics["split_metrics"]
        summary_rows.append(
            {
                "model_name": run["model_name"],
                "display_name": run["metadata"]["display_name"],
                "role": "baseline" if run_index == 0 else "candidate",
                "family": run["metadata"]["family"],
                "parameter_scale": run["metadata"]["parameter_scale"],
                "quantization": run["metadata"]["quantization"],
                "rows": len(row_labels),
                "accuracy_vs_dataset_want": metrics["accuracy"],
                "train_accuracy_vs_dataset_want": split_metrics["train"][
                    "accuracy"
                ],
                "test_accuracy_vs_dataset_want": split_metrics["test"][
                    "accuracy"
                ],
                "macro_f1_vs_dataset_want": metrics["macro_f1_all_16"],
                "row_agreement_count_with_qwen3vl_8b": row_agreement_count,
                "row_agreement_rate_with_qwen3vl_8b": safe_div(
                    row_agreement_count, len(row_labels)
                ),
                "unique_pair_agreement_count_with_qwen3vl_8b": (
                    unique_agreement_count
                ),
                "unique_pair_agreement_rate_with_qwen3vl_8b": safe_div(
                    unique_agreement_count, len(unique_labels)
                ),
                "row_cohen_kappa_with_qwen3vl_8b": cohen_kappa(
                    baseline_row_labels, row_labels, labels
                ),
                "row_intent_distribution_tv_distance_from_qwen3vl_8b": (
                    total_variation(
                        baseline_row_counts, row_counts, labels, len(row_labels)
                    )
                ),
                "unique_intent_distribution_tv_distance_from_qwen3vl_8b": (
                    total_variation(
                        baseline_unique_counts,
                        unique_counts,
                        labels,
                        len(unique_labels),
                    )
                ),
                "qwen3vl_8b_correct_candidate_wrong": (
                    baseline_correct_candidate_wrong
                ),
                "qwen3vl_8b_wrong_candidate_correct": (
                    baseline_wrong_candidate_correct
                ),
                "both_wrong_but_intents_differ": both_wrong_but_disagree,
                "run_dir": str(run["run_dir"]),
            }
        )

    summary_fields = list(summary_rows[0])
    write_csv(
        args.output_dir / "vlm_intent_agreement_summary.csv",
        summary_rows,
        summary_fields,
    )

    composition_rows: list[dict[str, Any]] = []
    for intent in labels:
        output: dict[str, Any] = {"intent": intent}
        baseline_row_count = baseline_row_counts[intent]
        baseline_unique_count = baseline_unique_counts[intent]
        output["qwen3vl_8b_row_count"] = baseline_row_count
        output["qwen3vl_8b_row_rate"] = safe_div(
            baseline_row_count, len(baseline["rows"])
        )
        output["qwen3vl_8b_unique_pair_count"] = baseline_unique_count
        output["qwen3vl_8b_unique_pair_rate"] = safe_div(
            baseline_unique_count, len(baseline["unique_rows"])
        )
        for run in candidates:
            model_slug = run["slug"]
            row_counts = Counter(
                row["selected_intent"] for row in run["rows"]
            )
            unique_counts = Counter(
                row["selected_intent"] for row in run["unique_rows"]
            )
            row_count = row_counts[intent]
            unique_count = unique_counts[intent]
            row_rate = safe_div(row_count, len(run["rows"]))
            unique_rate = safe_div(unique_count, len(run["unique_rows"]))
            output[f"{model_slug}_row_count"] = row_count
            output[f"{model_slug}_row_rate"] = row_rate
            output[f"{model_slug}_row_delta_count_vs_8b"] = (
                row_count - baseline_row_count
            )
            output[f"{model_slug}_row_delta_percentage_points_vs_8b"] = (
                100.0
                * (
                    row_rate
                    - safe_div(baseline_row_count, len(baseline["rows"]))
                )
            )
            output[f"{model_slug}_unique_pair_count"] = unique_count
            output[f"{model_slug}_unique_pair_rate"] = unique_rate
            output[f"{model_slug}_unique_delta_count_vs_8b"] = (
                unique_count - baseline_unique_count
            )
            output[
                f"{model_slug}_unique_delta_percentage_points_vs_8b"
            ] = 100.0 * (
                unique_rate
                - safe_div(
                    baseline_unique_count, len(baseline["unique_rows"])
                )
            )
        composition_rows.append(output)
    composition_fields = list(composition_rows[0])
    write_csv(
        args.output_dir / "vlm_intent_composition.csv",
        composition_rows,
        composition_fields,
    )

    per_intent_rows: list[dict[str, Any]] = []
    for run in candidates:
        for intent in labels:
            row_indices = [
                index
                for index, label in enumerate(baseline_row_labels)
                if label == intent
            ]
            unique_indices = [
                index
                for index, label in enumerate(baseline_unique_labels)
                if label == intent
            ]
            row_predictions = [
                run["rows"][index]["selected_intent"] for index in row_indices
            ]
            unique_predictions = [
                run["unique_rows"][index]["selected_intent"]
                for index in unique_indices
            ]
            per_intent_rows.append(
                {
                    "model_name": run["model_name"],
                    "baseline_intent": intent,
                    "baseline_row_support": len(row_indices),
                    "row_agreement_count": sum(
                        prediction == intent for prediction in row_predictions
                    ),
                    "row_agreement_rate": safe_div(
                        sum(
                            prediction == intent
                            for prediction in row_predictions
                        ),
                        len(row_indices),
                    ),
                    "baseline_unique_pair_support": len(unique_indices),
                    "unique_pair_agreement_count": sum(
                        prediction == intent
                        for prediction in unique_predictions
                    ),
                    "unique_pair_agreement_rate": safe_div(
                        sum(
                            prediction == intent
                            for prediction in unique_predictions
                        ),
                        len(unique_indices),
                    ),
                    "row_candidate_prediction_counts": json.dumps(
                        dict(sorted(Counter(row_predictions).items())),
                        sort_keys=True,
                    ),
                    "unique_candidate_prediction_counts": json.dumps(
                        dict(sorted(Counter(unique_predictions).items())),
                        sort_keys=True,
                    ),
                }
            )
    per_intent_fields = list(per_intent_rows[0])
    write_csv(
        args.output_dir / "vlm_per_intent_agreement.csv",
        per_intent_rows,
        per_intent_fields,
    )

    combined_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
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
    for run in runs:
        model_fields.extend(
            [
                f"{run['slug']}_selected_intent",
                f"{run['slug']}_is_correct_vs_want",
            ]
        )
    for index, baseline_row in enumerate(baseline["rows"]):
        output = {field: baseline_row[field] for field in base_fields}
        predictions: list[str] = []
        for run in runs:
            run_row = run["rows"][index]
            prediction = run_row["selected_intent"]
            predictions.append(prediction)
            output[f"{run['slug']}_selected_intent"] = prediction
            output[f"{run['slug']}_is_correct_vs_want"] = run_row[
                "is_correct"
            ]
        output["all_models_agree"] = len(set(predictions)) == 1
        output["candidate_disagrees_with_qwen3vl_8b"] = " | ".join(
            run["model_name"]
            for run, prediction in zip(candidates, predictions[1:])
            if prediction != predictions[0]
        )
        combined_rows.append(output)
        if not output["all_models_agree"]:
            disagreement_rows.append(output)
    combined_fields = base_fields + model_fields + [
        "all_models_agree",
        "candidate_disagrees_with_qwen3vl_8b",
    ]
    write_csv(
        args.output_dir / "vlm_intent_predictions_comparison.csv",
        combined_rows,
        combined_fields,
    )
    write_csv(
        args.output_dir / "vlm_intent_disagreements.csv",
        disagreement_rows,
        combined_fields,
    )

    pairwise_rows: list[dict[str, Any]] = []
    for left in runs:
        for right in runs:
            row_agree = sum(
                left_row["selected_intent"] == right_row["selected_intent"]
                for left_row, right_row in zip(left["rows"], right["rows"])
            )
            unique_agree = sum(
                left_row["selected_intent"] == right_row["selected_intent"]
                for left_row, right_row in zip(
                    left["unique_rows"], right["unique_rows"]
                )
            )
            pairwise_rows.append(
                {
                    "left_model": left["model_name"],
                    "right_model": right["model_name"],
                    "row_agreement_count": row_agree,
                    "row_agreement_rate": safe_div(
                        row_agree, len(left["rows"])
                    ),
                    "unique_pair_agreement_count": unique_agree,
                    "unique_pair_agreement_rate": safe_div(
                        unique_agree, len(left["unique_rows"])
                    ),
                }
            )
    write_csv(
        args.output_dir / "vlm_pairwise_intent_agreement.csv",
        pairwise_rows,
        list(pairwise_rows[0]),
    )

    all_models_agree_count = sum(
        bool(row["all_models_agree"]) for row in combined_rows
    )
    report_lines = [
        "# GIMO VLM intent agreement ablation",
        "",
        "## Protocol",
        "",
        (
            f"- Input: {len(baseline['rows'])} GIMO samples, "
            f"{len(baseline['unique_rows'])} unique Action+Object pairs."
        ),
        (
            f"- Fixed taxonomy: {', '.join(labels)}."
        ),
        (
            f"- Prompt version: `{baseline['metrics']['prompt_version']}`; "
            "temperature 0, seed 0, thinking disabled."
        ),
        (
            "- All variants use the same semantic prompt and deterministic "
            "decoding. Runtime and quantization are recorded per model."
        ),
        (
            "- Qwen3-VL-8B-Instruct is the same-prompt reference. Agreement "
            "means exact equality of the selected categorical intent, not "
            "lexical/token overlap."
        ),
        (
            "- This is a text-only Action+Object intent ablation. Although all "
            "backbones are VLMs, no image is supplied, so the result "
            "must not be presented as visual grounding performance."
        ),
        "",
        "## Main results",
        "",
        (
            "| Model | Want accuracy | Agreement with Qwen3-VL-8B "
            "(rows) | Agreement (unique pairs) | Cohen's κ | "
            "Intent-distribution TV |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| {display_name} | {accuracy} | {row_agreement} | "
            "{unique_agreement} | {kappa:.4f} | {tv:.4f} |".format(
                display_name=row["display_name"],
                accuracy=percentage(float(row["accuracy_vs_dataset_want"])),
                row_agreement=(
                    f"{row['row_agreement_count_with_qwen3vl_8b']}/"
                    f"{row['rows']} "
                    f"({percentage(float(row['row_agreement_rate_with_qwen3vl_8b']))})"
                ),
                unique_agreement=(
                    f"{row['unique_pair_agreement_count_with_qwen3vl_8b']}/"
                    f"{len(baseline['unique_rows'])} "
                    f"({percentage(float(row['unique_pair_agreement_rate_with_qwen3vl_8b']))})"
                ),
                kappa=float(row["row_cohen_kappa_with_qwen3vl_8b"]),
                tv=float(
                    row[
                        "row_intent_distribution_tv_distance_from_qwen3vl_8b"
                    ]
                ),
            )
        )
    report_lines.extend(
        [
            "",
            (
                f"All {len(runs)} models selected the same intent for "
                f"{all_models_agree_count}/{len(combined_rows)} rows "
                f"({percentage(safe_div(all_models_agree_count, len(combined_rows)))})."
            ),
            "",
            "## Agreement and correctness transitions",
            "",
            (
                "| Model | Agrees with 8B | 8B correct → candidate wrong | "
                "8B wrong → candidate correct | Both wrong, different labels |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows[1:]:
        report_lines.append(
            f"| {row['display_name']} | "
            f"{row['row_agreement_count_with_qwen3vl_8b']}/{row['rows']} | "
            f"{row['qwen3vl_8b_correct_candidate_wrong']} | "
            f"{row['qwen3vl_8b_wrong_candidate_correct']} | "
            f"{row['both_wrong_but_intents_differ']} |"
        )
    report_lines.extend(
        [
            "",
            "## Largest composition shifts from Qwen3-VL-8B",
            "",
        ]
    )
    for run in candidates:
        model_slug = run["slug"]
        delta_field = (
            f"{model_slug}_row_delta_percentage_points_vs_8b"
        )
        largest = sorted(
            composition_rows,
            key=lambda row: abs(float(row[delta_field])),
            reverse=True,
        )[:5]
        report_lines.append(f"### {run['metadata']['display_name']}")
        report_lines.append("")
        report_lines.append("| Intent | Δ count | Δ percentage points |")
        report_lines.append("|---|---:|---:|")
        for row in largest:
            report_lines.append(
                f"| {row['intent']} | "
                f"{int(row[f'{model_slug}_row_delta_count_vs_8b']):+d} | "
                f"{float(row[delta_field]):+.2f} |"
            )
        report_lines.append("")
    report_lines.extend(
        [
            "## Interpretation rule",
            "",
            (
                "Use the agreement columns as the primary size-ablation "
                "result. Dataset-`want` accuracy is a secondary external "
                "reference; it answers a different question from agreement "
                "with the 8B teacher."
            ),
            "",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vlm_intent_ablation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    write_json(
        args.output_dir / "summary.json",
        {
            "baseline_model": baseline["model_name"],
            "candidate_models": [run["model_name"] for run in candidates],
            "prompt_version": baseline["metrics"]["prompt_version"],
            "rows": len(baseline["rows"]),
            "unique_pairs": len(baseline["unique_rows"]),
            "canonical_intents": labels,
            "all_models_agree_count": all_models_agree_count,
            "all_models_agree_rate": safe_div(
                all_models_agree_count, len(combined_rows)
            ),
            "primary_metric": (
                "exact categorical intent agreement with the same-prompt "
                "Qwen3-VL-8B-Instruct baseline"
            ),
            "text_only_ablation": True,
            "visual_grounding_metric": False,
            "files": {
                "report_markdown": str(
                    args.output_dir / "vlm_intent_ablation_report.md"
                ),
                "agreement_summary_csv": str(
                    args.output_dir / "vlm_intent_agreement_summary.csv"
                ),
                "composition_csv": str(
                    args.output_dir / "vlm_intent_composition.csv"
                ),
                "per_intent_agreement_csv": str(
                    args.output_dir / "vlm_per_intent_agreement.csv"
                ),
                "comparison_csv": str(
                    args.output_dir / "vlm_intent_predictions_comparison.csv"
                ),
                "disagreements_csv": str(
                    args.output_dir / "vlm_intent_disagreements.csv"
                ),
                "pairwise_agreement_csv": str(
                    args.output_dir / "vlm_pairwise_intent_agreement.csv"
                ),
            },
        },
    )
    print(f"Wrote {args.output_dir / 'vlm_intent_ablation_report.md'}")
    print(f"Wrote {args.output_dir / 'vlm_intent_agreement_summary.csv'}")
    print(f"Wrote {args.output_dir / 'vlm_intent_composition.csv'}")
    print(f"Wrote {args.output_dir / 'vlm_intent_disagreements.csv'}")


if __name__ == "__main__":
    main()
