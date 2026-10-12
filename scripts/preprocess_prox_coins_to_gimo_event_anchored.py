#!/usr/bin/env python
"""Build PROX/COINS tensors with object-event anchored sequence cuts.

Each sample is anchored by a clear non-floor/wall interaction at the final
frame. Walking/standing floor-only context and immediate same-target object
context may be attached before the anchor, but a different object interaction
starts a new sample. The script also writes CSV/JSON audit tables describing
which action/object was used to cut each sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preprocess_prox_coins_to_gimo as base
import preprocess_prox_coins_to_gimo_long as long_base


INPUT_LEN = base.INPUT_LEN
OUTPUT_LEN = base.OUTPUT_LEN
TOTAL_LEN = base.TOTAL_LEN
FLOOR_WALL = base.FLOOR_WALL

TARGET_PRIORITY_PREFIXES = (
    "sit down",
    "sit on",
    "lie down",
    "lie on",
    "stand up",
    "sit up",
    "step down",
    "step up",
    "restfoot",
    "touch",
    "move leg",
    "move",
)


def clean_prefix(label: str) -> str:
    prefix = long_base.label_prefix(label)
    if prefix.startswith("wrong "):
        prefix = prefix.removeprefix("wrong ").strip()
    return prefix


def target_priority_rank(label: str) -> int:
    prefix = clean_prefix(label)
    for idx, candidate in enumerate(TARGET_PRIORITY_PREFIXES):
        if prefix.startswith(candidate):
            return idx
    return len(TARGET_PRIORITY_PREFIXES)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def primary_target_from_record(
    record: dict[str, Any],
    scene_instance_info: dict[int, base.InstanceInfo],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for order, (label, object_idx) in enumerate(zip(record["interaction_labels"], record["interaction_obj_idx"])):
        object_idx = int(object_idx)
        label = str(label)
        label_object_name = base.object_name_from_label(label)
        catalog_info = scene_instance_info.get(object_idx)
        semantic_name = (catalog_info.semantic_label_name if catalog_info else label_object_name).lower()
        is_floor_wall = label_object_name in FLOOR_WALL or semantic_name in FLOOR_WALL
        rank = target_priority_rank(label)
        candidates.append(
            {
                "interaction_label": label,
                "action_prefix": clean_prefix(label),
                "object_idx": object_idx,
                "label_object_name": label_object_name,
                "semantic_label_name": catalog_info.semantic_label_name if catalog_info else None,
                "is_floor_wall": is_floor_wall,
                "target_priority_rank": rank,
                "annotation_order": order,
            }
        )

    object_candidates = [row for row in candidates if not row["is_floor_wall"]]
    if not object_candidates:
        return {
            "target": None,
            "candidates": candidates,
            "object_candidate_count": 0,
            "ambiguous_object_candidates": False,
            "target_selection_rule": "no non-floor/wall object candidate",
        }

    ranked = sorted(
        object_candidates,
        key=lambda row: (row["target_priority_rank"], row["annotation_order"], row["object_idx"]),
    )
    target = dict(ranked[0])
    target["target_priority_name"] = TARGET_PRIORITY_PREFIXES[target["target_priority_rank"]] if target["target_priority_rank"] < len(TARGET_PRIORITY_PREFIXES) else "other"
    ambiguous = len(object_candidates) > 1
    return {
        "target": target,
        "candidates": candidates,
        "object_candidate_count": len(object_candidates),
        "ambiguous_object_candidates": ambiguous,
        "target_selection_rule": "first non-floor/wall candidate by action priority, then annotation order",
    }


def segment_gap(prev_segment: list[dict[str, Any]], next_segment: list[dict[str, Any]]) -> int:
    return int(next_segment[0]["frame_idx"]) - int(prev_segment[-1]["frame_idx"])


def sample_segment(segment: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    sampled_idx = np.rint(np.linspace(0, len(segment) - 1, TOTAL_LEN)).astype(np.int64)
    sampled = [segment[int(idx)] for idx in sampled_idx]
    return sampled, [int(record["frame_idx"]) for record in sampled]


def action_label_key(segment: list[dict[str, Any]]) -> str:
    return " + ".join(str(label) for label in segment[0]["interaction_labels"])


def build_event_samples(
    records: list[dict[str, Any]],
    scene_instance_info: dict[str, dict[int, base.InstanceInfo]],
    max_context_frames: int,
    max_gap_frames: int,
    include_same_target_context: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments_by_sequence = long_base.action_segments_by_sequence(records)
    samples: list[dict[str, Any]] = []
    target_label_counter: Counter[str] = Counter()
    target_object_counter: Counter[str] = Counter()
    skipped_too_short = 0
    anchor_segment_count = 0
    raw_segment_count = sum(len(segments) for segments in segments_by_sequence.values())

    for sequence, segments in segments_by_sequence.items():
        for anchor_idx, anchor_segment in enumerate(segments):
            scene = anchor_segment[-1]["scene_name"]
            target_info = primary_target_from_record(anchor_segment[-1], scene_instance_info[scene])
            target = target_info["target"]
            if target is None:
                continue
            anchor_segment_count += 1

            target_object_idx = int(target["object_idx"])
            first_segment_idx = anchor_idx
            context_frames = 0
            include_roles = {anchor_idx: "anchor"}
            allow_older_same_target = True

            scan_idx = anchor_idx - 1
            while scan_idx >= 0:
                candidate = segments[scan_idx]
                next_segment = segments[first_segment_idx]
                if segment_gap(candidate, next_segment) > max_gap_frames:
                    break

                is_floor_context = long_base.is_floor_context_segment(candidate)
                is_same_target = include_same_target_context and long_base.has_same_object_idx(candidate, target_object_idx)
                if is_same_target and not allow_older_same_target:
                    break
                if not (is_floor_context or is_same_target):
                    break
                if context_frames + len(candidate) > max_context_frames:
                    break

                include_roles[scan_idx] = "floor_context" if is_floor_context else "same_target_context"
                if is_floor_context:
                    allow_older_same_target = False
                first_segment_idx = scan_idx
                context_frames += len(candidate)
                scan_idx -= 1

            long_segment: list[dict[str, Any]] = []
            included_segments: list[dict[str, Any]] = []
            for include_idx in range(first_segment_idx, anchor_idx + 1):
                segment = segments[include_idx]
                long_segment.extend(segment)
                included_segments.append(
                    {
                        "segment_idx": include_idx,
                        "role": include_roles.get(include_idx, "included"),
                        "start_frame_idx": int(segment[0]["frame_idx"]),
                        "end_frame_idx": int(segment[-1]["frame_idx"]),
                        "num_frames": len(segment),
                        "interaction_labels": [str(label) for label in segment[0]["interaction_labels"]],
                        "interaction_obj_idx": [int(value) for value in segment[0]["interaction_obj_idx"]],
                    }
                )

            if len(long_segment) < TOTAL_LEN:
                skipped_too_short += 1
                continue

            target_name = str(target.get("semantic_label_name") or target["label_object_name"]).lower()
            target_label_counter[str(target["interaction_label"])] += 1
            target_object_counter[target_name] += 1
            samples.append(
                {
                    "sequence": sequence,
                    "scene": scene,
                    "first_segment_idx": first_segment_idx,
                    "anchor_segment_idx": anchor_idx,
                    "segment": long_segment,
                    "anchor_segment": anchor_segment,
                    "included_segments": included_segments,
                    "target_info": target_info,
                    "context_frames": int(len(long_segment) - len(anchor_segment)),
                    "num_action_segments": int(anchor_idx - first_segment_idx + 1),
                }
            )

    summary = {
        "records": len(records),
        "raw_action_segments": raw_segment_count,
        "object_anchor_segments": anchor_segment_count,
        "event_segments_len_at_least_16": len(samples),
        "skipped_too_short_after_context": skipped_too_short,
        "target_action_label_counts": dict(target_label_counter.most_common()),
        "target_object_counts": dict(target_object_counter.most_common()),
        "target_priority_prefixes": list(TARGET_PRIORITY_PREFIXES),
        "include_same_target_context": include_same_target_context,
        "max_context_frames": max_context_frames,
        "max_gap_frames": max_gap_frames,
        "window_rule": "floor-only context plus immediate same-target context before a non-floor/wall anchor event",
    }
    return samples, summary


def percentile_payload(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(arr.max()),
    }


def write_event_csv(path: Path, manifests: list[dict[str, Any]]) -> None:
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifests:
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
                }
            )


def build_split_event(
    split: str,
    records_path: Path,
    output_split_root: Path,
    inventories: dict[str, dict[str, Any]],
    segmentation_dir: Path,
    sample_points: int,
    object_points: int,
    max_context_frames: int,
    max_gap_frames: int,
    include_same_target_context: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = pickle.loads(records_path.read_bytes())
    records, canonical_summary = long_base.canonicalize_records(raw_records)
    source_sequence_counts = Counter(str(row["sequence"]) for row in raw_records)
    source_sequence_scene = {str(row["sequence"]): str(row["scene_name"]) for row in raw_records}

    output_split_root.mkdir(parents=True, exist_ok=True)
    scene_instance_info = {
        scene: base.parse_instance_txt(segmentation_dir / f"{scene}.txt") for scene in inventories
    }
    event_samples, event_summary = build_event_samples(
        records,
        scene_instance_info=scene_instance_info,
        max_context_frames=max_context_frames,
        max_gap_frames=max_gap_frames,
        include_same_target_context=include_same_target_context,
    )

    poses_all: list[np.ndarray] = []
    scene_points_all: list[np.ndarray] = []
    object_points_all: list[np.ndarray] = []
    object_mask_all: list[bool] = []
    manifests: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    segment_lengths: list[float] = []
    context_lengths: list[float] = []
    anchor_lengths: list[float] = []
    produced_by_sequence: Counter[str] = Counter()

    for sample in event_samples:
        segment = sample["segment"]
        sampled, sampled_frame_indices = sample_segment(segment)
        scene = sample["scene"]
        sequence = sample["sequence"]
        target_info = sample["target_info"]
        target = target_info["target"]
        assert target is not None

        sample_index = len(poses_all)
        sample_id = (
            f"{split}_{sample_index:05d}_{sequence}"
            f"_event_f{segment[0]['frame_idx']}_f{segment[-1]['frame_idx']}"
        )
        poses = np.stack([base.frame_pose(record) for record in sampled], axis=0)
        poses_all.append(poses)
        segment_lengths.append(float(len(segment)))
        context_lengths.append(float(sample["context_frames"]))
        anchor_lengths.append(float(len(sample["anchor_segment"])))
        produced_by_sequence[sequence] += 1

        scene_xyz = inventories[scene]["xyz"]
        scene_points_all.append(base.sample_points_fn(scene_xyz, sample_points, base.stable_seed("scene", scene)))

        object_idx = int(target["object_idx"])
        obj_xyz = inventories[scene]["points_by_object_idx"].get(object_idx, np.zeros((0, 3), dtype=np.float32))
        has_points = obj_xyz.shape[0] > 0
        object_points_all.append(base.sample_points_fn(obj_xyz, object_points, base.stable_seed("object", scene, object_idx)))
        object_mask_all.append(has_points)

        manifest_row = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "split": split,
            "scene_name": scene,
            "sequence": sequence,
            "segment_start_frame_idx": int(segment[0]["frame_idx"]),
            "segment_end_frame_idx": int(segment[-1]["frame_idx"]),
            "segment_num_frames": len(segment),
            "anchor_start_frame_idx": int(sample["anchor_segment"][0]["frame_idx"]),
            "anchor_end_frame_idx": int(sample["anchor_segment"][-1]["frame_idx"]),
            "anchor_num_frames": len(sample["anchor_segment"]),
            "context_num_frames": int(sample["context_frames"]),
            "num_action_segments": int(sample["num_action_segments"]),
            "included_segments": sample["included_segments"],
            "sampled_frame_indices": sampled_frame_indices,
            "sampled_interaction_labels": [list(record["interaction_labels"]) for record in sampled],
            "sampled_interaction_obj_idx": [
                [int(value) for value in record["interaction_obj_idx"]] for record in sampled
            ],
            "sampled_raw_interaction_labels": [
                list(record.get("raw_interaction_labels", record["interaction_labels"])) for record in sampled
            ],
            "sampled_raw_interaction_obj_idx": [
                [int(value) for value in record.get("raw_interaction_obj_idx", record["interaction_obj_idx"])]
                for record in sampled
            ],
            "target_candidates": target_info["candidates"],
            "target_object": target,
            "target_object_mask": bool(object_mask_all[-1]),
            "object_candidate_count": int(target_info["object_candidate_count"]),
            "ambiguous_object_candidates": bool(target_info["ambiguous_object_candidates"]),
            "target_rule": target_info["target_selection_rule"],
            "window_rule": "floor-only context plus immediate same-target context before a non-floor/wall anchor event",
        }
        manifests.append(manifest_row)
        dataset_rows.append(
            {
                "sequence_path": sample_id,
                "start_frame": int(segment[0]["frame_idx"]),
                "end_frame": int(segment[-1]["frame_idx"]),
                "scene": scene,
                "transformation": "identity",
                "training": 1 if split == "train" else 0,
            }
        )

    if not poses_all:
        raise ValueError(f"{split}: no samples produced")

    poses_tensor = torch.from_numpy(np.stack(poses_all, axis=0)).float()
    torch.save(poses_tensor[:, :INPUT_LEN], output_split_root / "poses_input.pth")
    torch.save(poses_tensor[:, INPUT_LEN:], output_split_root / "poses_label.pth")
    torch.save(torch.from_numpy(np.stack(scene_points_all, axis=0)).float(), output_split_root / f"scene_points_{sample_points}.pth")
    torch.save(torch.from_numpy(np.stack(object_points_all, axis=0)).float(), output_split_root / f"final_object_gt_points_{object_points}.pth")
    torch.save(torch.tensor(object_mask_all, dtype=torch.bool), output_split_root / f"final_object_gt_points_{object_points}_mask.pth")
    (output_split_root / "segment_manifest.json").write_text(json.dumps(manifests, indent=2))
    (output_split_root / f"final_object_gt_points_{object_points}_manifest.json").write_text(json.dumps(manifests, indent=2))
    write_event_csv(output_split_root / "event_segments.csv", manifests)
    (output_split_root / "event_segments.json").write_text(json.dumps(manifests, indent=2))

    source_sequence_rows = []
    for sequence in sorted(source_sequence_counts):
        source_sequence_rows.append(
            {
                "split": split,
                "source_sequence": sequence,
                "scene_name": source_sequence_scene[sequence],
                "source_frame_count": int(source_sequence_counts[sequence]),
                "produced_sample_count": int(produced_by_sequence[sequence]),
            }
        )

    summary = {
        "split": split,
        "raw_records": len(raw_records),
        "canonicalization": canonical_summary,
        **event_summary,
        "samples": len(poses_all),
        "object_valid_samples": int(sum(object_mask_all)),
        "object_missing_samples": int(len(object_mask_all) - sum(object_mask_all)),
        "source_unique_sequences": len(source_sequence_counts),
        "produced_unique_sequences": len([row for row in source_sequence_rows if row["produced_sample_count"] > 0]),
        "segment_num_frames_stats": percentile_payload(segment_lengths),
        "context_num_frames_stats": percentile_payload(context_lengths),
        "anchor_num_frames_stats": percentile_payload(anchor_lengths),
    }
    (output_split_root / "summary_no_joints_yet.json").write_text(json.dumps(summary, indent=2))
    return dataset_rows, summary, manifests, source_sequence_rows


def write_dataset_csv(output_root: Path, rows: list[dict[str, Any]]) -> None:
    with (output_root / "dataset.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sequence_path", "start_frame", "end_frame", "scene", "transformation", "training"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_source_sequence_csv(output_root: Path, rows: list[dict[str, Any]]) -> None:
    with (output_root / "source_sequences.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "source_sequence", "scene_name", "source_frame_count", "produced_sample_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_scene_object_tables(
    output_root: Path,
    inventories: dict[str, dict[str, Any]],
    manifests: list[dict[str, Any]],
    source_sequence_rows: list[dict[str, Any]],
) -> None:
    target_counts = Counter((row["scene_name"], int(row["target_object"]["object_idx"])) for row in manifests)
    scene_object_rows = []
    for scene, inventory in sorted(inventories.items()):
        for item in inventory["catalog"]["instances"]:
            object_idx = int(item["object_idx"])
            centroid = item["centroid"]
            bbox_min = item["bbox_min"]
            bbox_max = item["bbox_max"]
            scene_object_rows.append(
                {
                    "scene_name": scene,
                    "instance_id": item["instance_id"],
                    "object_idx": object_idx,
                    "semantic_label_name": item["semantic_label_name"],
                    "num_points": item["num_points"],
                    "centroid_x": centroid[0],
                    "centroid_y": centroid[1],
                    "centroid_z": centroid[2],
                    "bbox_min_x": bbox_min[0],
                    "bbox_min_y": bbox_min[1],
                    "bbox_min_z": bbox_min[2],
                    "bbox_max_x": bbox_max[0],
                    "bbox_max_y": bbox_max[1],
                    "bbox_max_z": bbox_max[2],
                    "target_sample_count": int(target_counts[(scene, object_idx)]),
                }
            )

    object_fieldnames = [
        "scene_name",
        "instance_id",
        "object_idx",
        "semantic_label_name",
        "num_points",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_min_z",
        "bbox_max_x",
        "bbox_max_y",
        "bbox_max_z",
        "target_sample_count",
    ]
    with (output_root / "scene_objects.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=object_fieldnames)
        writer.writeheader()
        writer.writerows(scene_object_rows)

    objects_by_scene = defaultdict(list)
    for row in scene_object_rows:
        objects_by_scene[row["scene_name"]].append(row)

    sequence_object_rows = []
    for source in source_sequence_rows:
        for obj in objects_by_scene[source["scene_name"]]:
            sequence_object_rows.append(
                {
                    "split": source["split"],
                    "source_sequence": source["source_sequence"],
                    "scene_name": source["scene_name"],
                    "source_frame_count": source["source_frame_count"],
                    "produced_sample_count": source["produced_sample_count"],
                    **obj,
                }
            )

    sequence_object_fieldnames = [
        "split",
        "source_sequence",
        "scene_name",
        "source_frame_count",
        "produced_sample_count",
        *object_fieldnames[1:],
    ]
    with (output_root / "sequence_scene_objects.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sequence_object_fieldnames)
        writer.writeheader()
        writer.writerows(sequence_object_rows)

    sequence_payload = []
    for source in source_sequence_rows:
        sequence_payload.append(
            {
                **source,
                "scene_objects": objects_by_scene[source["scene_name"]],
            }
        )
    (output_root / "sequence_scene_objects.json").write_text(json.dumps(sequence_payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-root", type=Path, default=Path("data/PROX"))
    parser.add_argument("--coins-root", type=Path, default=Path("data/PROX/COINS"))
    parser.add_argument("--scenes-semantics-root", type=Path, default=Path("data/PROX/scenes_semantics"))
    parser.add_argument("--output-root", type=Path, default=Path("data/PROX/PROX_COINS_GIMO_EVENT_ANCHORED"))
    parser.add_argument("--smplx-path", type=Path, default=Path("data/body_models"))
    parser.add_argument("--vposer-path", type=Path, default=Path("./vposer_v1_0"))
    parser.add_argument("--sample-points", type=int, default=32768)
    parser.add_argument("--object-points", type=int, default=1024)
    parser.add_argument("--smplx-batch-size", type=int, default=256)
    parser.add_argument("--max-context-frames", type=int, default=240)
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--no-same-target-context", action="store_true")
    parser.add_argument("--no-export-instance-ply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    slices_root = output_root / "SLICES_888s"
    scene_objects_root = output_root / "scene_objects"
    scene_dir = args.prox_root / "scenes"
    scene_semantics_dir = args.scenes_semantics_root
    segmentation_dir = args.coins_root / "scene_segmentation"
    semantics_dir = args.coins_root / "interaction_semantics"

    scene_names = sorted(path.name.removesuffix("_segment.json") for path in segmentation_dir.glob("*_segment.json"))
    inventories = {
        scene: base.load_scene_inventory(
            scene,
            [
                scene_dir / f"{scene}.ply",
                scene_semantics_dir / f"{scene}.ply",
                scene_semantics_dir / f"{scene}_withlabels.ply",
                scene_semantics_dir / f"{scene}_withlabels_OpenAWall.ply",
                scene_semantics_dir / f"{scene}_withlabels_old.ply",
            ],
            segmentation_dir,
            scene_objects_root,
            export_instances=not args.no_export_instance_ply,
        )
        for scene in scene_names
    }

    dataset_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    all_manifests: list[dict[str, Any]] = []
    all_source_sequence_rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        rows, summary, manifests, source_sequence_rows = build_split_event(
            split=split,
            records_path=semantics_dir / f"{split}.pkl",
            output_split_root=slices_root / split,
            inventories=inventories,
            segmentation_dir=segmentation_dir,
            sample_points=args.sample_points,
            object_points=args.object_points,
            max_context_frames=args.max_context_frames,
            max_gap_frames=args.max_gap_frames,
            include_same_target_context=not args.no_same_target_context,
        )
        dataset_rows.extend(rows)
        split_summaries[split] = summary
        all_manifests.extend(manifests)
        all_source_sequence_rows.extend(source_sequence_rows)

    write_dataset_csv(output_root, dataset_rows)
    write_event_csv(output_root / "event_segments.csv", all_manifests)
    (output_root / "event_segments.json").write_text(json.dumps(all_manifests, indent=2))
    write_source_sequence_csv(output_root, all_source_sequence_rows)
    write_scene_object_tables(output_root, inventories, all_manifests, all_source_sequence_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vposer, _ = load_vposer(str(args.vposer_path), vp_model="snapshot")
    vposer = vposer.to(device).eval()
    body_model = smplx.create(
        str(args.smplx_path),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        num_pca_comps=12,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=args.smplx_batch_size,
    ).to(device).eval()

    shape_summary = {}
    for split in ("train", "test"):
        shape_summary[split] = base.add_joints_to_split(
            slices_root / split,
            vposer=vposer,
            body_model=body_model,
            batch_size=args.smplx_batch_size,
            device=device,
        )

    final_summary = {
        "output_root": str(output_root),
        "coordinate_frame": "PROX_scene",
        "input_seq_len": INPUT_LEN,
        "output_seq_len": OUTPUT_LEN,
        "sample_points": args.sample_points,
        "object_points": args.object_points,
        "target_rule": "non-floor/wall object selected by action priority",
        "window_rule": "floor-only context plus immediate same-target context before a non-floor/wall anchor event",
        "target_priority_prefixes": list(TARGET_PRIORITY_PREFIXES),
        "max_context_frames": args.max_context_frames,
        "max_gap_frames": args.max_gap_frames,
        "include_same_target_context": not args.no_same_target_context,
        "splits": split_summaries,
        "tensor_shapes": shape_summary,
        "dataset_csv_rows": len(dataset_rows),
        "audit_files": {
            "event_segments_csv": str(output_root / "event_segments.csv"),
            "event_segments_json": str(output_root / "event_segments.json"),
            "source_sequences_csv": str(output_root / "source_sequences.csv"),
            "scene_objects_csv": str(output_root / "scene_objects.csv"),
            "sequence_scene_objects_csv": str(output_root / "sequence_scene_objects.csv"),
            "sequence_scene_objects_json": str(output_root / "sequence_scene_objects.json"),
        },
    }
    (output_root / "preprocess_summary.json").write_text(json.dumps(final_summary, indent=2))
    for split, summary in split_summaries.items():
        summary_path = slices_root / split / "summary.json"
        payload = dict(summary)
        payload["tensor_shapes"] = shape_summary[split]
        summary_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
