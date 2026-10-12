#!/usr/bin/env python3
"""Summarize sample-wise noisy segmentation selection and motion experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CONDITIONS = (
    "random_instance_deletion_10pct",
    "random_instance_deletion_20pct",
    "target_neighbor_merge",
    "target_two_way_split",
    "partial_noisy_target_mask",
)
SPLIT_SIZES = {"train": 178, "test": 37}
METRICS = ("loss_total", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe")
TEST_RE = re.compile(
    r"Test epoch:\s*(?P<epoch>\d+)\s+"
    r"loss_total:\s*(?P<loss_total>[-+.\deE]+)\s+"
    r"loss_trans:\s*(?P<loss_trans>[-+.\deE]+)\s+"
    r"loss_des_trans:\s*(?P<loss_des_trans>[-+.\deE]+)\s+"
    r"mpjpe:\s*(?P<mpjpe>[-+.\deE]+)\s+"
    r"des_mpjpe:\s*(?P<des_mpjpe>[-+.\deE]+)"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--slice-root",
        type=Path,
        default=Path("data/GIMO/SLICES_888s"),
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=Path(
            "data/OpenIns3D/output/results_segmentation_noise_samplewise"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected-epochs", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "new_runs" / "segmentation_noise",
    )
    return parser.parse_args()


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value}")
    return result


def summarize_selector(args: argparse.Namespace, condition: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    inventory_path = args.noise_root / condition / "inventory_manifest.json"
    inventory = json.loads(inventory_path.read_text())
    corruptions = {
        str(record["sample_uid"]): record for record in inventory["samples"]
    }
    for split, expected_size in SPLIT_SIZES.items():
        manifest_path = (
            args.slice_root
            / "segmentation_noise"
            / condition
            / split
            / "final_object_qwen3_14b_segmentation_noise_points_1024_manifest.json"
        )
        if not manifest_path.is_file():
            result[f"qwen_{split}_manifest_exists"] = False
            result[f"qwen_{split}_count"] = 0
            result[f"qwen_{split}_correct"] = 0
            result[f"qwen_{split}_top1_accuracy"] = None
            continue

        records = json.loads(manifest_path.read_text())
        ok_records = [record for record in records if record.get("status") == "ok"]
        raw_id_correct = sum(
            bool(record.get("selected_matches_gt")) for record in ok_records
        )
        lineage_correct = 0
        for record in ok_records:
            accepted_ids = {str(record["gt_object_id"])}
            corruption = corruptions[str(record["sample_uid"])]
            accepted_ids.update(
                str(value)
                for value in corruption.get("target_descendant_instance_ids", [])
            )
            if corruption.get("target_instance_id_after"):
                accepted_ids.add(str(corruption["target_instance_id_after"]))
            lineage_correct += str(record["selected_object_id"]) in accepted_ids
        result[f"qwen_{split}_manifest_exists"] = True
        result[f"qwen_{split}_count"] = len(ok_records)
        result[f"qwen_{split}_expected_count"] = expected_size
        result[f"qwen_{split}_raw_id_correct"] = raw_id_correct
        result[f"qwen_{split}_correct"] = lineage_correct
        result[f"qwen_{split}_top1_accuracy"] = (
            lineage_correct / len(ok_records) if ok_records else None
        )
        result[f"qwen_{split}_all_ok"] = (
            len(records) == expected_size and len(ok_records) == expected_size
        )
    return result


def parse_test_rows(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for match in TEST_RE.finditer(log_path.read_text(errors="replace")):
        row: dict[str, Any] = {"epoch": int(match.group("epoch"))}
        for metric in METRICS:
            row[metric] = finite_float(match.group(metric))
        rows.append(row)
    return rows


def summarize_run(
    args: argparse.Namespace, stage: str, condition: str
) -> dict[str, Any]:
    run_dir = (
        args.root
        / "new_runs"
        / "segmentation_noise"
        / stage
        / f"{condition}_seed{args.seed}"
    )
    log_path = run_dir / "log_rank0.txt"
    rows = parse_test_rows(log_path)
    result: dict[str, Any] = {
        f"{stage}_run_dir": str(run_dir),
        f"{stage}_log_exists": log_path.is_file(),
        f"{stage}_test_epoch_count": len(rows),
        f"{stage}_last_checkpoint_exists": (run_dir / "last.pth").is_file(),
        f"{stage}_best_loss_des_checkpoint_exists": (
            run_dir / "best_loss_des_trans.pth"
        ).is_file(),
    }
    if not rows:
        result[f"{stage}_completed"] = False
        return result

    epochs = [row["epoch"] for row in rows]
    result[f"{stage}_first_epoch"] = epochs[0]
    result[f"{stage}_last_epoch"] = epochs[-1]
    result[f"{stage}_epochs_contiguous"] = epochs == list(range(len(rows)))
    result[f"{stage}_completed"] = (
        len(rows) == args.expected_epochs
        and epochs == list(range(args.expected_epochs))
        and result[f"{stage}_last_checkpoint_exists"]
        and result[f"{stage}_best_loss_des_checkpoint_exists"]
    )

    final_row = rows[-1]
    for metric in METRICS:
        result[f"{stage}_final_{metric}"] = final_row[metric]
        best_row = min(rows, key=lambda row: row[metric])
        result[f"{stage}_best_{metric}"] = best_row[metric]
        result[f"{stage}_best_{metric}_epoch"] = best_row["epoch"]

    checkpoint_row = min(rows, key=lambda row: row["loss_des_trans"])
    result[f"{stage}_checkpoint_epoch"] = checkpoint_row["epoch"]
    for metric in METRICS:
        result[f"{stage}_checkpoint_{metric}"] = checkpoint_row[metric]
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        row: dict[str, Any] = {"condition": condition, "seed": args.seed}
        row.update(summarize_selector(args, condition))
        row.update(summarize_run(args, "teacher", condition))
        row.update(summarize_run(args, "student", condition))
        rows.append(row)

    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete_teachers = sum(bool(row.get("teacher_completed")) for row in rows)
    complete_students = sum(bool(row.get("student_completed")) for row in rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(
        f"completed teacher={complete_teachers}/{len(CONDITIONS)} "
        f"student={complete_students}/{len(CONDITIONS)}"
    )


if __name__ == "__main__":
    main()
