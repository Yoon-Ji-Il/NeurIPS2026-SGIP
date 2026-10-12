#!/usr/bin/env python3
"""Anchor raw Qwen choice logits to the existing Qwen hard Top-1 without using GT."""

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
        "--hard-root",
        type=Path,
        default=(
            REPO_ROOT
            / "object_selector"
            / "teacher_targets"
            / "segmentation_noise"
        ),
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument(
        "--minimum-logit-margin",
        type=float,
        default=1.0,
        help="Minimum raw-logit lead assigned to the existing hard Top-1.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def process_condition(args: argparse.Namespace, condition: str) -> dict:
    soft_dir = args.soft_root / condition
    hard_dir = args.hard_root / condition
    candidate_rows = read_csv(soft_dir / "teacher_candidate_scores.csv")
    raw_summary_rows = read_csv(soft_dir / "teacher_selection_summary.csv")
    hard_summary_rows = read_csv(hard_dir / "teacher_selection_summary.csv")

    hard_lookup = {
        (row["split"], int(row["split_index"])): row["selected_object_id"]
        for row in hard_summary_rows
        if row.get("status") == "ok"
    }
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["split"], int(row["split_index"]))].append(row)

    output_rows: list[dict[str, object]] = []
    anchored_samples = 0
    for key, rows in groups.items():
        hard_object_id = hard_lookup.get(key, "")
        id_to_index = {row["object_id"]: index for index, row in enumerate(rows)}
        if hard_object_id not in id_to_index:
            raise ValueError(
                f"{condition} {key}: hard Qwen object {hard_object_id!r} "
                "is absent from the noisy candidate set"
            )
        raw_logits = [float(row["teacher_logit"]) for row in rows]
        calibrated_logits = list(raw_logits)
        hard_index = id_to_index[hard_object_id]
        max_other = max(
            value for index, value in enumerate(raw_logits) if index != hard_index
        ) if len(raw_logits) > 1 else raw_logits[hard_index]
        anchor_applied = len(raw_logits) > 1 and raw_logits[hard_index] <= max_other
        if anchor_applied:
            calibrated_logits[hard_index] = (
                max_other + float(args.minimum_logit_margin)
            )
            anchored_samples += 1
        posterior = softmax(calibrated_logits, float(args.temperature))

        for index, row in enumerate(rows):
            output = dict(row)
            output.update(
                {
                    "hard_teacher_object_id": hard_object_id,
                    "teacher_logit_hard_consistent": calibrated_logits[index],
                    "posterior_hard_consistent": posterior[index],
                    "is_hard_teacher_selected": int(index == hard_index),
                    "hard_anchor_applied": int(anchor_applied),
                }
            )
            output_rows.append(output)

    summary_lookup = {
        (row["split"], int(row["split_index"])): row for row in raw_summary_rows
    }
    output_summary_rows: list[dict[str, object]] = []
    grouped_output: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in output_rows:
        grouped_output[(str(row["split"]), int(row["split_index"]))].append(row)
    for key in sorted(summary_lookup, key=lambda item: (item[0], item[1])):
        raw_row = summary_lookup[key]
        rows = grouped_output.get(key, [])
        if raw_row.get("status") != "ok" or not rows:
            output_summary_rows.append(dict(raw_row))
            continue
        selected = max(
            rows, key=lambda row: float(row["posterior_hard_consistent"])
        )
        output = dict(raw_row)
        output.update(
            {
                "raw_soft_selected_object_id": raw_row["selected_object_id"],
                "selected_object_id": selected["object_id"],
                "selected_label": selected["name"],
                "selected_total": selected["teacher_logit_hard_consistent"],
                "selected_posterior": selected["posterior_hard_consistent"],
                "hard_anchor_applied": selected["hard_anchor_applied"],
            }
        )
        output_summary_rows.append(output)

    candidate_path = soft_dir / "teacher_candidate_scores_hard_consistent.csv"
    summary_path = soft_dir / "teacher_selection_summary_hard_consistent.csv"
    write_csv(candidate_path, output_rows)
    write_csv(summary_path, output_summary_rows)
    summary = {
        "condition": condition,
        "temperature": args.temperature,
        "minimum_logit_margin": args.minimum_logit_margin,
        "num_samples": len(groups),
        "anchored_samples": anchored_samples,
        "already_consistent_samples": len(groups) - anchored_samples,
        "candidate_csv": str(candidate_path),
        "summary_csv": str(summary_path),
        "hard_top1_match_count": sum(
            max(rows, key=lambda row: float(row["posterior_hard_consistent"]))[
                "object_id"
            ]
            == hard_lookup[key]
            for key, rows in grouped_output.items()
        ),
    }
    (soft_dir / "hard_consistent_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be > 0")
    if args.minimum_logit_margin <= 0.0:
        raise ValueError("--minimum-logit-margin must be > 0")
    summaries = [
        process_condition(args, condition) for condition in args.conditions
    ]
    print(json.dumps(summaries, indent=2))
    (args.soft_root / "hard_consistent_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
