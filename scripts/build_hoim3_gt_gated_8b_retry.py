#!/usr/bin/env python3
"""Prepare and merge a GT-gated Qwen3-VL-8B retry for HOI-M3.

GT is used only to decide which existing Qwen3-VL-4B selections are retried.
The retry inference manifest intentionally contains no GT interaction fields,
so Qwen3-VL-8B still sees only the target marker, privileged 3-second video,
scene graph, and observed object candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_ROOT = Path("data/HOIm3")
PROTOCOL = "hoim3-entity-interaction-hikscale-v1"
BASE_ROOT = REPO_ROOT / "runs_hoim3" / "qwen3vl4b_teacher"
INPUT_ROOT = REPO_ROOT / "runs_hoim3" / "qwen3vl8b_teacher"
OUTPUT_ROOT = (
    REPO_ROOT / "runs_hoim3" / "qwen3vl8b_gt_gated_retry"
)
EXPECTED_COUNTS = {"train": 3332, "test": 689}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "merge"))
    parser.add_argument("--preprocess-root", type=Path, default=PREPROCESS_ROOT)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def sample_id(row: dict[str, Any]) -> str:
    return (
        f"{row['sequence']}|p{int(row['target_person_index'])}|"
        f"s{int(row['start_frame'])}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_motion_manifest(
    root: Path,
    protocol: str,
    split: str,
) -> list[dict[str, Any]]:
    path = root / "manifests" / protocol / f"{split}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
    os.replace(temporary, path)


def gt_target(row: dict[str, Any]) -> dict[str, Any]:
    observed_objects = {}
    future_objects = {}
    observed_human = False
    future_human = False
    for entity in row["entities"]:
        if entity["type"] == "object":
            object_id = f"o{int(entity['index'])}"
            object_name = str(entity["name"])
            if bool(entity.get("observed_interaction")):
                observed_objects[object_id] = object_name
            if bool(entity.get("future_interaction")):
                future_objects[object_id] = object_name
        elif entity["type"] == "human":
            observed_human |= bool(entity.get("observed_interaction"))
            future_human |= bool(entity.get("future_interaction"))
    all_objects = {**observed_objects, **future_objects}
    any_human = observed_human or future_human
    if all_objects:
        accepted_ids = sorted(all_objects)
    elif any_human:
        accepted_ids = ["__none__"]
    else:
        raise ValueError(
            f"{sample_id(row)} has no observed/future GT interaction entity"
        )
    return {
        "interaction_category": str(row["interaction_category"]),
        "observed_object_ids": sorted(observed_objects),
        "observed_object_names": [
            observed_objects[key] for key in sorted(observed_objects)
        ],
        "future_object_ids": sorted(future_objects),
        "future_object_names": [
            future_objects[key] for key in sorted(future_objects)
        ],
        "accepted_object_ids": accepted_ids,
        "accepted_object_names": (
            [all_objects[key] for key in accepted_ids]
            if all_objects
            else ["none / human-human interaction"]
        ),
        "observed_human_interaction": observed_human,
        "future_human_interaction": future_human,
    }


def aligned_rows(args: argparse.Namespace, split: str):
    motion = read_motion_manifest(
        args.preprocess_root,
        args.protocol,
        split,
    )
    selections = read_jsonl(args.base_root / split / "selections.jsonl")
    inference = read_jsonl(
        args.input_root / split / "inference_manifest_materialized.jsonl"
    )
    expected = EXPECTED_COUNTS[split]
    if not (len(motion) == len(selections) == len(inference) == expected):
        raise ValueError(
            f"{split} alignment lengths: motion={len(motion)}, "
            f"selection={len(selections)}, inference={len(inference)}, "
            f"expected={expected}"
        )
    for index, triple in enumerate(zip(motion, selections, inference, strict=True)):
        gt_row, selection, input_row = triple
        key = sample_id(gt_row)
        keys = (
            key,
            str(selection["sample_id"]),
            str(input_row["sample_id"]),
        )
        indices = (
            index,
            int(selection["sample_index"]),
            int(input_row["sample_index"]),
        )
        if len(set(keys)) != 1 or len(set(indices)) != 1:
            raise ValueError(
                f"{split}[{index}] alignment mismatch: keys={keys}, "
                f"indices={indices}"
            )
    return motion, selections, inference


def prepare_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    motion, selections, inference = aligned_rows(args, split)
    retry_inputs = []
    gates = []
    category_counts = Counter()
    wrong_pairs = Counter()
    correct = 0
    multi_gt = 0
    for index, (gt_row, selection, input_row) in enumerate(
        zip(motion, selections, inference, strict=True)
    ):
        target = gt_target(gt_row)
        accepted = set(target["accepted_object_ids"])
        selected = str(selection["selected_object_id"])
        if selected in accepted:
            correct += 1
            continue
        candidate_ids = {
            str(candidate["object_id"])
            for candidate in input_row["objects"]
        }
        selectable = sorted(accepted & candidate_ids)
        if not selectable:
            raise ValueError(
                f"{selection['sample_id']}: GT target {sorted(accepted)} "
                f"is absent from candidates {sorted(candidate_ids)}"
            )
        retry_inputs.append(input_row)
        gate = {
            "sample_index": index,
            "sample_id": str(selection["sample_id"]),
            "split": split,
            "qwen4b_selected_object_id": selected,
            "qwen4b_selected_object_name": str(
                selection["selected_object_name"]
            ),
            "selectable_accepted_object_ids": selectable,
            **target,
        }
        gates.append(gate)
        category_counts[target["interaction_category"]] += 1
        if len(accepted) > 1:
            multi_gt += 1
        wrong_pairs[
            (
                str(selection["selected_object_name"]),
                ", ".join(target["accepted_object_names"]),
            )
        ] += 1
    split_dir = args.output_root / split
    atomic_jsonl(
        split_dir / "mismatch_inference_manifest.jsonl",
        retry_inputs,
    )
    atomic_jsonl(split_dir / "gt_gate_manifest.jsonl", gates)
    summary = {
        "split": split,
        "samples": len(motion),
        "qwen4b_correct": correct,
        "qwen4b_mismatch": len(gates),
        "qwen4b_set_accuracy": correct / len(motion),
        "multi_gt_mismatches": multi_gt,
        "mismatch_categories": dict(sorted(category_counts.items())),
        "top_mismatch_pairs": [
            {
                "qwen4b_selected": selected,
                "gt_accepted": accepted,
                "count": count,
            }
            for (selected, accepted), count in wrong_pairs.most_common(50)
        ],
        "gt_definition": (
            "Union of observed-1s and future-2s interaction objects; "
            "__none__ only when there is no interacting object and at least "
            "one observed/future human interaction."
        ),
        "gt_exposed_to_qwen8b": False,
        "retry_input": str(
            (split_dir / "mismatch_inference_manifest.jsonl").resolve()
        ),
        "gate_manifest": str(
            (split_dir / "gt_gate_manifest.jsonl").resolve()
        ),
    }
    atomic_json(split_dir / "prepare_summary.json", summary)
    return summary


def merge_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    _, base_selections, _ = aligned_rows(args, split)
    gates = read_jsonl(args.output_root / split / "gt_gate_manifest.jsonl")
    retries = read_jsonl(args.output_root / split / "retry_selections.jsonl")
    gate_by_id = {str(row["sample_id"]): row for row in gates}
    retry_by_id = {str(row["sample_id"]): row for row in retries}
    if len(gate_by_id) != len(gates) or len(retry_by_id) != len(retries):
        raise ValueError(f"{split}: duplicate gate or retry sample IDs")
    if set(gate_by_id) != set(retry_by_id):
        missing = sorted(set(gate_by_id) - set(retry_by_id))
        extra = sorted(set(retry_by_id) - set(gate_by_id))
        raise ValueError(
            f"{split}: retry set mismatch; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    hybrid = []
    retry_correct = 0
    retry_statuses = Counter()
    changed = 0
    for base in base_selections:
        key = str(base["sample_id"])
        if key not in gate_by_id:
            output = dict(base)
            output["selection_provenance"] = "qwen3vl4b_gt_set_match"
            output["gt_gated_retry"] = False
        else:
            gate = gate_by_id[key]
            output = dict(retry_by_id[key])
            output["selection_provenance"] = "qwen3vl8b_gt_gated_retry"
            output["gt_gated_retry"] = True
            output["qwen4b_original_selection"] = {
                "selected_object_id": gate["qwen4b_selected_object_id"],
                "selected_object_name": gate["qwen4b_selected_object_name"],
            }
            output["gt_gate_metadata"] = {
                field: gate[field]
                for field in (
                    "interaction_category",
                    "observed_object_ids",
                    "future_object_ids",
                    "accepted_object_ids",
                    "accepted_object_names",
                    "observed_human_interaction",
                    "future_human_interaction",
                )
            }
            retry_statuses[str(output["status"])] += 1
            accepted = set(gate["accepted_object_ids"])
            retry_correct += int(
                str(output["selected_object_id"]) in accepted
            )
            changed += int(
                str(output["selected_object_id"])
                != str(gate["qwen4b_selected_object_id"])
            )
        hybrid.append(output)
    hybrid.sort(key=lambda row: int(row["sample_index"]))
    if len(hybrid) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split}: hybrid count {len(hybrid)}")
    atomic_jsonl(args.output_root / split / "selections.jsonl", hybrid)
    base_correct = len(base_selections) - len(gates)
    summary = {
        "split": split,
        "samples": len(base_selections),
        "qwen4b_kept_gt_matches": base_correct,
        "qwen8b_retried": len(gates),
        "qwen8b_retry_changed_top1": changed,
        "qwen8b_retry_gt_set_matches": retry_correct,
        "qwen8b_retry_gt_set_accuracy": retry_correct / len(gates),
        "hybrid_gt_set_matches": base_correct + retry_correct,
        "hybrid_gt_set_accuracy": (
            (base_correct + retry_correct) / len(base_selections)
        ),
        "retry_status_counts": dict(sorted(retry_statuses.items())),
        "gt_gated": True,
        "gt_exposed_to_qwen8b": False,
        "deployment_valid": False,
        "interpretation": (
            "Oracle/privileged analysis: GT decides whether 4B or 8B is run. "
            "Do not report as a standard test-time selector."
        ),
        "output": str(
            (args.output_root / split / "selections.jsonl").resolve()
        ),
    }
    atomic_json(args.output_root / split / "hybrid_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        summaries = {
            split: prepare_split(args, split)
            for split in ("train", "test")
        }
        atomic_json(args.output_root / "prepare_summary.json", summaries)
    else:
        summaries = {
            split: merge_split(args, split)
            for split in ("train", "test")
        }
        atomic_json(args.output_root / "hybrid_summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
