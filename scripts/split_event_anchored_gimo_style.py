#!/usr/bin/env python
"""Create a GIMO-sized train/val/test split from EVENT_ANCHORED tensors."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch


TENSOR_FILES = (
    "poses_input.pth",
    "poses_label.pth",
    "joints_input.pth",
    "joints_label.pth",
    "scene_points_32768.pth",
    "gazes.pth",
    "final_object_gt_points_1024.pth",
    "final_object_gt_points_1024_mask.pth",
)

VAL_FROM_TRAIN_SEQUENCES = {"N0Sofa_00034_02"}
VAL_FROM_TEST_SEQUENCES = {
    "MPH16_00157_01",
    "N3OpenArea_03301_01",
    "N0SittingBooth_03301_01",
    "N0SittingBooth_03403_01",
    "N3OpenArea_00158_01",
}


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def target_split(original_split: str, source_sequence: str) -> str:
    if original_split == "train":
        return "val" if source_sequence in VAL_FROM_TRAIN_SEQUENCES else "train"
    if original_split == "test":
        return "val" if source_sequence in VAL_FROM_TEST_SEQUENCES else "test"
    raise ValueError(f"Unexpected split: {original_split}")


def load_manifest(root: Path, split: str) -> list[dict[str, Any]]:
    return json.loads((root / "SLICES_888s" / split / "segment_manifest.json").read_text())


def sample_id_for(split: str, index: int, row: dict[str, Any]) -> str:
    return (
        f"{split}_{index:05d}_{row['sequence']}"
        f"_event_f{row['segment_start_frame_idx']}_f{row['segment_end_frame_idx']}"
    )


def write_dataset_csv(path: Path, rows: list[dict[str, Any]], include_val: bool) -> None:
    fieldnames = ["sequence_path", "start_frame", "end_frame", "scene", "transformation", "training"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["split"] == "val" and not include_val:
                continue
            training = "1" if row["split"] == "train" else ("val" if row["split"] == "val" else "0")
            writer.writerow(
                {
                    "sequence_path": row["sample_id"],
                    "start_frame": row["segment_start_frame_idx"],
                    "end_frame": row["segment_end_frame_idx"],
                    "scene": row["scene_name"],
                    "transformation": "identity",
                    "training": training,
                }
            )


def write_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "sample_id",
        "source_sequence",
        "scene_name",
        "segment_start_frame_idx",
        "segment_end_frame_idx",
        "segment_num_frames",
        "anchor_start_frame_idx",
        "anchor_end_frame_idx",
        "anchor_num_frames",
        "context_num_frames",
        "num_action_segments",
        "target_action_label",
        "target_action_prefix",
        "target_object_idx",
        "target_object_name",
        "target_semantic_label_name",
        "target_priority_rank",
        "target_priority_name",
        "ambiguous_object_candidates",
        "object_candidate_count",
        "included_segment_roles",
        "included_action_labels",
        "sampled_frame_indices",
        "target_candidates_json",
        "original_split",
        "original_sample_index",
        "original_sample_id",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            target = row["target_object"]
            writer.writerow(
                {
                    "split": row["split"],
                    "sample_index": row["sample_index"],
                    "sample_id": row["sample_id"],
                    "source_sequence": row["sequence"],
                    "scene_name": row["scene_name"],
                    "segment_start_frame_idx": row["segment_start_frame_idx"],
                    "segment_end_frame_idx": row["segment_end_frame_idx"],
                    "segment_num_frames": row["segment_num_frames"],
                    "anchor_start_frame_idx": row["anchor_start_frame_idx"],
                    "anchor_end_frame_idx": row["anchor_end_frame_idx"],
                    "anchor_num_frames": row["anchor_num_frames"],
                    "context_num_frames": row["context_num_frames"],
                    "num_action_segments": row["num_action_segments"],
                    "target_action_label": target["interaction_label"],
                    "target_action_prefix": target["action_prefix"],
                    "target_object_idx": target["object_idx"],
                    "target_object_name": target["label_object_name"],
                    "target_semantic_label_name": target.get("semantic_label_name"),
                    "target_priority_rank": target["target_priority_rank"],
                    "target_priority_name": target["target_priority_name"],
                    "ambiguous_object_candidates": row["ambiguous_object_candidates"],
                    "object_candidate_count": row["object_candidate_count"],
                    "included_segment_roles": json_compact([item["role"] for item in row["included_segments"]]),
                    "included_action_labels": json_compact([item["interaction_labels"] for item in row["included_segments"]]),
                    "sampled_frame_indices": json_compact(row["sampled_frame_indices"]),
                    "target_candidates_json": json_compact(row["target_candidates"]),
                    "original_split": row["original_split"],
                    "original_sample_index": row["original_sample_index"],
                    "original_sample_id": row["original_sample_id"],
                }
            )


def write_source_sequences_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["split", "source_sequence", "scene_name", "source_frame_count", "produced_sample_count"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_root_metadata(input_root: Path, output_root: Path) -> None:
    for file_name in ("scene_objects.csv", "sequence_scene_objects.json"):
        src = input_root / file_name
        if src.exists():
            shutil.copy2(src, output_root / file_name)
    if (input_root / "scene_objects").exists():
        shutil.copytree(input_root / "scene_objects", output_root / "scene_objects")


def write_sequence_scene_objects_csv(input_root: Path, output_root: Path, source_rows: list[dict[str, Any]]) -> None:
    scene_objects_path = input_root / "scene_objects.csv"
    if not scene_objects_path.exists():
        return

    with scene_objects_path.open() as file:
        objects = list(csv.DictReader(file))

    objects_by_scene: dict[str, list[dict[str, str]]] = {}
    for row in objects:
        objects_by_scene.setdefault(row["scene_name"], []).append(row)

    fieldnames = [
        "split",
        "source_sequence",
        "scene_name",
        "source_frame_count",
        "produced_sample_count",
        *[name for name in objects[0].keys() if name != "scene_name"],
    ]
    with (output_root / "sequence_scene_objects.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for source in source_rows:
            for obj in objects_by_scene.get(source["scene_name"], []):
                writer.writerow(
                    {
                        **source,
                        **{key: value for key, value in obj.items() if key != "scene_name"},
                    }
                )

    payload = []
    for source in source_rows:
        payload.append({**source, "scene_objects": objects_by_scene.get(source["scene_name"], [])})
    (output_root / "sequence_scene_objects.json").write_text(json.dumps(payload, indent=2))


def subset_tensors(input_root: Path, output_root: Path, assignments: dict[str, list[dict[str, Any]]]) -> None:
    source_by_split = {
        split: load_manifest(input_root, split)
        for split in ("train", "test")
    }
    indices_by_output_and_source: dict[str, dict[str, list[int]]] = {
        split: {"train": [], "test": []}
        for split in ("train", "val", "test")
    }
    for output_split, rows in assignments.items():
        for row in rows:
            indices_by_output_and_source[output_split][row["original_split"]].append(int(row["original_sample_index"]))

    for output_split, rows in assignments.items():
        split_root = output_root / "SLICES_888s" / output_split
        split_root.mkdir(parents=True, exist_ok=True)
        for tensor_name in TENSOR_FILES:
            chunks = []
            for source_split in ("train", "test"):
                indices = indices_by_output_and_source[output_split][source_split]
                if not indices:
                    continue
                tensor = torch.load(input_root / "SLICES_888s" / source_split / tensor_name, map_location="cpu", weights_only=False)
                chunks.append(tensor[torch.tensor(indices, dtype=torch.long)])
            if not chunks:
                raise ValueError(f"No tensor chunks for {output_split}/{tensor_name}")
            torch.save(torch.cat(chunks, dim=0), split_root / tensor_name)
        (split_root / "segment_manifest.json").write_text(json.dumps(rows, indent=2))
        (split_root / "final_object_gt_points_1024_manifest.json").write_text(json.dumps(rows, indent=2))
        write_event_csv(split_root / "event_segments.csv", rows)
        (split_root / "event_segments.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/PROX/PROX_COINS_GIMO_EVENT_ANCHORED"))
    parser.add_argument("--output-root", type=Path, default=Path("data/PROX/PROX_COINS_GIMO_EVENT_ANCHORED_GIMO_SPLIT"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} already exists. Pass --overwrite.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    copy_root_metadata(args.input_root, args.output_root)

    assignments: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for original_split in ("train", "test"):
        for row in load_manifest(args.input_root, original_split):
            output_split = target_split(original_split, str(row["sequence"]))
            updated = dict(row)
            updated["original_split"] = original_split
            updated["original_sample_index"] = int(row["sample_index"])
            updated["original_sample_id"] = row["sample_id"]
            updated["split"] = output_split
            updated["sample_index"] = len(assignments[output_split])
            updated["sample_id"] = sample_id_for(output_split, updated["sample_index"], updated)
            assignments[output_split].append(updated)

    subset_tensors(args.input_root, args.output_root, assignments)

    all_rows = [row for split in ("train", "val", "test") for row in assignments[split]]
    write_dataset_csv(args.output_root / "dataset.csv", all_rows, include_val=False)
    write_dataset_csv(args.output_root / "dataset_all_splits.csv", all_rows, include_val=True)
    write_event_csv(args.output_root / "event_segments.csv", all_rows)
    (args.output_root / "event_segments.json").write_text(json.dumps(all_rows, indent=2))

    source_counts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in all_rows:
        key = (row["split"], row["sequence"], row["scene_name"])
        payload = source_counts.setdefault(
            key,
            {
                "split": row["split"],
                "source_sequence": row["sequence"],
                "scene_name": row["scene_name"],
                "source_frame_count": 0,
                "produced_sample_count": 0,
            },
        )
        payload["source_frame_count"] = max(payload["source_frame_count"], int(row["segment_end_frame_idx"]) + 1)
        payload["produced_sample_count"] += 1
    source_rows = sorted(source_counts.values(), key=lambda row: (row["split"], row["scene_name"], row["source_sequence"]))
    write_source_sequences_csv(args.output_root / "source_sequences.csv", source_rows)
    write_sequence_scene_objects_csv(args.input_root, args.output_root, source_rows)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "split_rule": {
            "val_from_train_sequences": sorted(VAL_FROM_TRAIN_SEQUENCES),
            "val_from_test_sequences": sorted(VAL_FROM_TEST_SEQUENCES),
            "note": "Splits are reassigned by complete source_sequence where possible to match GIMO-sized train/test counts.",
        },
        "counts": {split: len(rows) for split, rows in assignments.items()},
        "dataset_csv": "train/test rows only for current SIF3D loader",
        "dataset_all_splits_csv": "train/val/test rows for audit and scene-graph generation",
    }
    (args.output_root / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
