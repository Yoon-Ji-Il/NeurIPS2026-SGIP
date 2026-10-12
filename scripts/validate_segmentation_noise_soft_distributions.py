#!/usr/bin/env python3
"""Validate candidate alignment and probability invariants for noisy Qwen posteriors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "random_instance_deletion_10pct",
    "random_instance_deletion_20pct",
    "target_neighbor_merge",
    "target_two_way_split",
    "partial_noisy_target_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--soft-root",
        type=Path,
        default=(
            REPO_ROOT
            / "object_selector"
            / "teacher_targets"
            / "segmentation_noise_soft"
        ),
    )
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "object_selector" / "scene_graphs" / "segmentation_noise",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument(
        "--hard-consistent",
        action="store_true",
        help="Validate the posterior anchored to the existing Qwen hard Top-1.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_graphs(root: Path, condition: str) -> dict[tuple[str, int], dict]:
    output = {}
    for split in ("train", "test"):
        path = root / condition / split / "samples.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                key = (split, int(graph["sample_index"]))
                if key in output:
                    raise ValueError(f"Duplicate scene graph key: {key}")
                output[key] = graph
    return output


def validate_condition(args: argparse.Namespace, condition: str) -> dict:
    output_dir = args.soft_root / condition
    candidate_name = (
        "teacher_candidate_scores_hard_consistent.csv"
        if args.hard_consistent
        else "teacher_candidate_scores.csv"
    )
    posterior_field = (
        "posterior_hard_consistent" if args.hard_consistent else "posterior"
    )
    final_field = (
        "is_hard_teacher_selected" if args.hard_consistent else "is_final"
    )
    candidates = read_csv(output_dir / candidate_name)
    failures = read_csv(output_dir / "teacher_selection_failures.csv")
    graphs = load_graphs(args.scene_graph_root, condition)

    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        groups[(row["split"], int(row["split_index"]))].append(row)

    errors: list[str] = []
    if len(groups) != len(graphs):
        errors.append(f"sample_count:{len(groups)}!={len(graphs)}")
    if failures:
        errors.append(f"failure_count:{len(failures)}")

    entropy_sum = 0.0
    top1_sum = 0.0
    missing_gt_samples = 0
    for key, graph in graphs.items():
        rows = groups.get(key, [])
        expected_ids = {
            str(obj["object_id"])
            for obj in graph.get("objects", [])
            if str(obj.get("object_id", "")).strip()
        }
        actual_ids = {row["object_id"] for row in rows}
        if actual_ids != expected_ids:
            errors.append(f"{key}:candidate_set_mismatch")
            continue
        probabilities = [float(row[posterior_field]) for row in rows]
        if not probabilities or not all(
            math.isfinite(value) and value >= 0.0 for value in probabilities
        ):
            errors.append(f"{key}:invalid_probability")
            continue
        probability_sum = sum(probabilities)
        if abs(probability_sum - 1.0) > 1.0e-5:
            errors.append(f"{key}:posterior_sum={probability_sum}")
        final_rows = [row for row in rows if row[final_field] == "1"]
        if len(final_rows) != 1:
            errors.append(f"{key}:final_count={len(final_rows)}")
        elif float(final_rows[0][posterior_field]) != max(probabilities):
            errors.append(f"{key}:final_not_argmax")
        if len({row["choice_token"] for row in rows}) != len(rows):
            errors.append(f"{key}:duplicate_choice_token")
        if not any(row["is_gt"] == "1" for row in rows):
            missing_gt_samples += 1
        entropy_sum += -sum(
            value * math.log(max(value, 1.0e-12)) for value in probabilities
        )
        top1_sum += max(probabilities)

    summary = {
        "condition": condition,
        "distribution": "hard_consistent" if args.hard_consistent else "raw",
        "num_samples": len(groups),
        "num_train_samples": sum(key[0] == "train" for key in groups),
        "num_test_samples": sum(key[0] == "test" for key in groups),
        "num_candidate_rows": len(candidates),
        "missing_gt_candidate_samples": missing_gt_samples,
        "mean_entropy": entropy_sum / max(len(groups), 1),
        "mean_top1_probability": top1_sum / max(len(groups), 1),
        "error_count": len(errors),
        "errors": errors[:100],
    }
    validation_name = (
        "hard_consistent_distribution_validation.json"
        if args.hard_consistent
        else "soft_distribution_validation.json"
    )
    (output_dir / validation_name).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summaries = [
        validate_condition(args, condition) for condition in args.conditions
    ]
    print(json.dumps(summaries, indent=2))
    if any(summary["error_count"] for summary in summaries):
        raise SystemExit(1)
    root_name = (
        "hard_consistent_validation_summary.json"
        if args.hard_consistent
        else "validation_summary.json"
    )
    (args.soft_root / root_name).write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
