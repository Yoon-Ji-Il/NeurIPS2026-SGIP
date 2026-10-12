#!/usr/bin/env python
"""Build a stricter PROX/COINS cache with context-only input and action future.

This variant keeps the same tensor contract as the other PROX/GIMO preprocessing
scripts, but changes how each sample is cut:

  - input (6 frames) is sampled only from contiguous floor-context motion that
    happens immediately before a target action onset;
  - future (10 frames) starts at the first target-action segment and may extend
    forward only through immediate same-target continuation segments;
  - same-target action frames are never allowed in the input context.

The goal is to remove the large "future hint" that the previous PROX caches
contained when they pulled same-target context into the observed window.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from collections import Counter
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
import preprocess_prox_coins_to_gimo_event_anchored as event_base
import preprocess_prox_coins_to_gimo_long as long_base


INPUT_LEN = base.INPUT_LEN
OUTPUT_LEN = base.OUTPUT_LEN
TOTAL_LEN = base.TOTAL_LEN


def segment_gap(prev_segment: list[dict[str, Any]], next_segment: list[dict[str, Any]]) -> int:
    return int(next_segment[0]["frame_idx"]) - int(prev_segment[-1]["frame_idx"])


def segment_target_info(
    segment: list[dict[str, Any]],
    scene_instance_info: dict[int, base.InstanceInfo],
) -> dict[str, Any]:
    return event_base.primary_target_from_record(segment[-1], scene_instance_info)


def trim_tail(records: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if max_frames <= 0 or len(records) <= max_frames:
        return records
    return records[-max_frames:]


def trim_head(records: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if max_frames <= 0 or len(records) <= max_frames:
        return records
    return records[:max_frames]


def sample_fixed_count(records: list[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[int]]:
    if len(records) < count:
        raise ValueError(f"Need at least {count} records, got {len(records)}")
    sampled_idx = np.rint(np.linspace(0, len(records) - 1, count)).astype(np.int64)
    sampled = [records[int(idx)] for idx in sampled_idx]
    frame_indices = [int(record["frame_idx"]) for record in sampled]
    return sampled, frame_indices


def collect_floor_context(
    segments: list[list[dict[str, Any]]],
    anchor_idx: int,
    max_context_frames: int,
    max_gap_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    floor_segments: list[list[dict[str, Any]]] = []
    scan_idx = anchor_idx - 1
    next_segment = segments[anchor_idx]
    while scan_idx >= 0:
        candidate = segments[scan_idx]
        if segment_gap(candidate, next_segment) > max_gap_frames:
            break
        if not long_base.is_floor_context_segment(candidate):
            break
        floor_segments.append(candidate)
        next_segment = candidate
        scan_idx -= 1

    floor_segments.reverse()
    included_segments = []
    flat_records: list[dict[str, Any]] = []
    for segment in floor_segments:
        included_segments.append(
            {
                "role": "floor_context",
                "start_frame_idx": int(segment[0]["frame_idx"]),
                "end_frame_idx": int(segment[-1]["frame_idx"]),
                "num_frames": len(segment),
                "interaction_labels": [str(label) for label in segment[0]["interaction_labels"]],
                "interaction_obj_idx": [int(value) for value in segment[0]["interaction_obj_idx"]],
            }
        )
        flat_records.extend(segment)

    trimmed = trim_tail(flat_records, max_context_frames)
    return trimmed, included_segments


def collect_future_action(
    segments: list[list[dict[str, Any]]],
    anchor_idx: int,
    target_object_idx: int,
    max_future_frames: int,
    max_gap_frames: int,
    include_following_same_target: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    action_segments = [segments[anchor_idx]]
    if include_following_same_target:
        prev_segment = segments[anchor_idx]
        scan_idx = anchor_idx + 1
        while scan_idx < len(segments):
            candidate = segments[scan_idx]
            if segment_gap(prev_segment, candidate) > max_gap_frames:
                break
            if not long_base.has_same_object_idx(candidate, target_object_idx):
                break
            action_segments.append(candidate)
            prev_segment = candidate
            scan_idx += 1

    included_segments = []
    flat_records: list[dict[str, Any]] = []
    for seg_idx, segment in enumerate(action_segments):
        included_segments.append(
            {
                "role": "anchor" if seg_idx == 0 else "same_target_future",
                "start_frame_idx": int(segment[0]["frame_idx"]),
                "end_frame_idx": int(segment[-1]["frame_idx"]),
                "num_frames": len(segment),
                "interaction_labels": [str(label) for label in segment[0]["interaction_labels"]],
                "interaction_obj_idx": [int(value) for value in segment[0]["interaction_obj_idx"]],
            }
        )
        flat_records.extend(segment)

    trimmed = trim_head(flat_records, max_future_frames)
    return trimmed, included_segments


def build_strict_samples(
    records: list[dict[str, Any]],
    scene_instance_info: dict[str, dict[int, base.InstanceInfo]],
    max_context_frames: int,
    max_future_frames: int,
    max_gap_frames: int,
    min_context_frames: int,
    min_future_frames: int,
    include_following_same_target: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments_by_sequence = long_base.action_segments_by_sequence(records)

    raw_segment_count = sum(len(segments) for segments in segments_by_sequence.values())
    anchor_candidates = 0
    samples: list[dict[str, Any]] = []
    target_label_counter: Counter[str] = Counter()
    target_object_counter: Counter[str] = Counter()
    skipped_not_onset = 0
    skipped_no_floor_context = 0
    skipped_short_context = 0
    skipped_short_future = 0

    for sequence, segments in segments_by_sequence.items():
        for anchor_idx, anchor_segment in enumerate(segments):
            scene = str(anchor_segment[-1]["scene_name"])
            target_info = segment_target_info(anchor_segment, scene_instance_info[scene])
            target = target_info["target"]
            if target is None:
                continue
            anchor_candidates += 1

            target_object_idx = int(target["object_idx"])

            if anchor_idx == 0:
                skipped_no_floor_context += 1
                continue

            prev_segment = segments[anchor_idx - 1]
            if segment_gap(prev_segment, anchor_segment) > max_gap_frames:
                skipped_no_floor_context += 1
                continue
            if not long_base.is_floor_context_segment(prev_segment):
                skipped_not_onset += 1
                continue

            context_records, context_segments = collect_floor_context(
                segments=segments,
                anchor_idx=anchor_idx,
                max_context_frames=max_context_frames,
                max_gap_frames=max_gap_frames,
            )
            if len(context_records) < min_context_frames:
                skipped_short_context += 1
                continue

            future_records, future_segments = collect_future_action(
                segments=segments,
                anchor_idx=anchor_idx,
                target_object_idx=target_object_idx,
                max_future_frames=max_future_frames,
                max_gap_frames=max_gap_frames,
                include_following_same_target=include_following_same_target,
            )
            if len(future_records) < min_future_frames:
                skipped_short_future += 1
                continue

            sampled_context, sampled_context_idx = sample_fixed_count(context_records, INPUT_LEN)
            sampled_future, sampled_future_idx = sample_fixed_count(future_records, OUTPUT_LEN)
            sampled = sampled_context + sampled_future
            sampled_frame_indices = sampled_context_idx + sampled_future_idx

            target_name = str(target.get("semantic_label_name") or target["label_object_name"]).lower()
            target_label_counter[str(target["interaction_label"])] += 1
            target_object_counter[target_name] += 1

            samples.append(
                {
                    "sequence": sequence,
                    "scene": scene,
                    "target_info": target_info,
                    "sampled": sampled,
                    "sampled_frame_indices": sampled_frame_indices,
                    "context_frame_indices": sampled_context_idx,
                    "future_frame_indices": sampled_future_idx,
                    "context_records_total": len(context_records),
                    "future_records_total": len(future_records),
                    "context_segments": context_segments,
                    "future_segments": future_segments,
                    "anchor_segment": anchor_segment,
                    "anchor_segment_idx": anchor_idx,
                    "context_start_frame_idx": int(context_records[0]["frame_idx"]),
                    "context_end_frame_idx": int(context_records[-1]["frame_idx"]),
                    "future_start_frame_idx": int(future_records[0]["frame_idx"]),
                    "future_end_frame_idx": int(future_records[-1]["frame_idx"]),
                }
            )

    summary = {
        "records": len(records),
        "raw_action_segments": raw_segment_count,
        "anchor_candidates_with_non_floor_target": anchor_candidates,
        "strict_samples": len(samples),
        "skipped_not_onset_or_non_floor_prev": skipped_not_onset,
        "skipped_missing_floor_context": skipped_no_floor_context,
        "skipped_short_context": skipped_short_context,
        "skipped_short_future": skipped_short_future,
        "target_action_label_counts": dict(target_label_counter.most_common()),
        "target_object_counts": dict(target_object_counter.most_common()),
        "target_priority_prefixes": list(event_base.TARGET_PRIORITY_PREFIXES),
        "window_rule": "input sampled only from contiguous floor-context before target onset; future sampled from anchor action onset plus same-target continuation",
        "max_context_frames": max_context_frames,
        "max_future_frames": max_future_frames,
        "max_gap_frames": max_gap_frames,
        "min_context_frames": min_context_frames,
        "min_future_frames": min_future_frames,
        "include_following_same_target": include_following_same_target,
    }
    return samples, summary


def build_split_strict(
    split: str,
    records_path: Path,
    output_split_root: Path,
    inventories: dict[str, dict[str, Any]],
    segmentation_dir: Path,
    sample_points: int,
    object_points: int,
    max_context_frames: int,
    max_future_frames: int,
    max_gap_frames: int,
    min_context_frames: int,
    min_future_frames: int,
    include_following_same_target: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = pickle.loads(records_path.read_bytes())
    records, canonical_summary = long_base.canonicalize_records(raw_records)
    source_sequence_counts = Counter(str(row["sequence"]) for row in raw_records)
    source_sequence_scene = {str(row["sequence"]): str(row["scene_name"]) for row in raw_records}

    output_split_root.mkdir(parents=True, exist_ok=True)
    scene_instance_info = {
        scene: base.parse_instance_txt(segmentation_dir / f"{scene}.txt") for scene in inventories
    }
    strict_samples, strict_summary = build_strict_samples(
        records=records,
        scene_instance_info=scene_instance_info,
        max_context_frames=max_context_frames,
        max_future_frames=max_future_frames,
        max_gap_frames=max_gap_frames,
        min_context_frames=min_context_frames,
        min_future_frames=min_future_frames,
        include_following_same_target=include_following_same_target,
    )

    poses_all: list[np.ndarray] = []
    scene_points_all: list[np.ndarray] = []
    object_points_all: list[np.ndarray] = []
    object_mask_all: list[bool] = []
    manifests: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    context_lengths: list[float] = []
    future_lengths: list[float] = []
    produced_by_sequence: Counter[str] = Counter()

    for sample in strict_samples:
        scene = sample["scene"]
        sequence = sample["sequence"]
        target_info = sample["target_info"]
        target = target_info["target"]
        assert target is not None

        sampled = sample["sampled"]
        sample_index = len(poses_all)
        sample_id = (
            f"{split}_{sample_index:05d}_{sequence}"
            f"_strict_ctx{sample['context_start_frame_idx']}-{sample['context_end_frame_idx']}"
            f"_fut{sample['future_start_frame_idx']}-{sample['future_end_frame_idx']}"
        )
        poses = np.stack([base.frame_pose(record) for record in sampled], axis=0)
        poses_all.append(poses)
        context_lengths.append(float(sample["context_records_total"]))
        future_lengths.append(float(sample["future_records_total"]))
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
            "context_start_frame_idx": sample["context_start_frame_idx"],
            "context_end_frame_idx": sample["context_end_frame_idx"],
            "future_start_frame_idx": sample["future_start_frame_idx"],
            "future_end_frame_idx": sample["future_end_frame_idx"],
            "context_num_frames_raw": int(sample["context_records_total"]),
            "future_num_frames_raw": int(sample["future_records_total"]),
            "anchor_start_frame_idx": int(sample["anchor_segment"][0]["frame_idx"]),
            "anchor_end_frame_idx": int(sample["anchor_segment"][-1]["frame_idx"]),
            "anchor_num_frames": len(sample["anchor_segment"]),
            "context_segments": sample["context_segments"],
            "future_segments": sample["future_segments"],
            "sampled_frame_indices": sample["sampled_frame_indices"],
            "sampled_context_frame_indices": sample["context_frame_indices"],
            "sampled_future_frame_indices": sample["future_frame_indices"],
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
            "target_rule": "first non-floor/wall target in the first action-onset segment, chosen by action priority then annotation order",
            "window_rule": "input sampled only from floor-context before onset; future sampled from onset and same-target continuation only",
        }
        manifests.append(manifest_row)
        dataset_rows.append(
            {
                "sequence_path": sample_id,
                "start_frame": int(sample["context_start_frame_idx"]),
                "end_frame": int(sample["future_end_frame_idx"]),
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
    write_strict_csv(output_split_root / "event_segments.csv", manifests)
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
        **strict_summary,
        "samples": len(poses_all),
        "object_valid_samples": int(sum(object_mask_all)),
        "object_missing_samples": int(len(object_mask_all) - sum(object_mask_all)),
        "source_unique_sequences": len(source_sequence_counts),
        "produced_unique_sequences": len([row for row in source_sequence_rows if row["produced_sample_count"] > 0]),
        "context_num_frames_stats": event_base.percentile_payload(context_lengths),
        "future_num_frames_stats": event_base.percentile_payload(future_lengths),
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


def write_strict_csv(path: Path, manifests: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "sample_id",
        "source_sequence",
        "scene_name",
        "context_start_frame_idx",
        "context_end_frame_idx",
        "context_num_frames_raw",
        "future_start_frame_idx",
        "future_end_frame_idx",
        "future_num_frames_raw",
        "anchor_start_frame_idx",
        "anchor_end_frame_idx",
        "anchor_num_frames",
        "target_action_label",
        "target_action_prefix",
        "target_object_idx",
        "target_object_name",
        "target_semantic_label_name",
        "target_priority_rank",
        "target_priority_name",
        "ambiguous_object_candidates",
        "object_candidate_count",
        "context_segment_labels",
        "future_segment_labels",
        "sampled_context_frame_indices",
        "sampled_future_frame_indices",
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
                    "context_start_frame_idx": row["context_start_frame_idx"],
                    "context_end_frame_idx": row["context_end_frame_idx"],
                    "context_num_frames_raw": row["context_num_frames_raw"],
                    "future_start_frame_idx": row["future_start_frame_idx"],
                    "future_end_frame_idx": row["future_end_frame_idx"],
                    "future_num_frames_raw": row["future_num_frames_raw"],
                    "anchor_start_frame_idx": row["anchor_start_frame_idx"],
                    "anchor_end_frame_idx": row["anchor_end_frame_idx"],
                    "anchor_num_frames": row["anchor_num_frames"],
                    "target_action_label": target["interaction_label"],
                    "target_action_prefix": target["action_prefix"],
                    "target_object_idx": target["object_idx"],
                    "target_object_name": target["label_object_name"],
                    "target_semantic_label_name": target.get("semantic_label_name"),
                    "target_priority_rank": target["target_priority_rank"],
                    "target_priority_name": target["target_priority_name"],
                    "ambiguous_object_candidates": row["ambiguous_object_candidates"],
                    "object_candidate_count": row["object_candidate_count"],
                    "context_segment_labels": json.dumps(
                        [item["interaction_labels"] for item in row["context_segments"]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "future_segment_labels": json.dumps(
                        [item["interaction_labels"] for item in row["future_segments"]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "sampled_context_frame_indices": json.dumps(
                        row["sampled_context_frame_indices"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "sampled_future_frame_indices": json.dumps(
                        row["sampled_future_frame_indices"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "target_candidates_json": json.dumps(
                        row["target_candidates"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-root", type=Path, default=Path("data/PROX"))
    parser.add_argument("--coins-root", type=Path, default=Path("data/PROX/COINS"))
    parser.add_argument("--scenes-semantics-root", type=Path, default=Path("data/PROX/scenes_semantics"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/PROX/PROX_COINS_GIMO_STRICT_CONTEXT_TO_ACTION"),
    )
    parser.add_argument("--smplx-path", type=Path, default=Path("data/body_models"))
    parser.add_argument("--vposer-path", type=Path, default=Path("./vposer_v1_0"))
    parser.add_argument("--sample-points", type=int, default=32768)
    parser.add_argument("--object-points", type=int, default=1024)
    parser.add_argument("--smplx-batch-size", type=int, default=256)
    parser.add_argument("--max-context-frames", type=int, default=90)
    parser.add_argument("--max-future-frames", type=int, default=120)
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--min-context-frames", type=int, default=INPUT_LEN)
    parser.add_argument("--min-future-frames", type=int, default=OUTPUT_LEN)
    parser.add_argument("--include-following-same-target", action="store_true", default=True)
    parser.add_argument("--no-include-following-same-target", dest="include_following_same_target", action="store_false")
    parser.add_argument("--skip-joints", action="store_true")
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
        rows, summary, manifests, source_rows = build_split_strict(
            split=split,
            records_path=semantics_dir / f"{split}.pkl",
            output_split_root=slices_root / split,
            inventories=inventories,
            segmentation_dir=segmentation_dir,
            sample_points=args.sample_points,
            object_points=args.object_points,
            max_context_frames=args.max_context_frames,
            max_future_frames=args.max_future_frames,
            max_gap_frames=args.max_gap_frames,
            min_context_frames=args.min_context_frames,
            min_future_frames=args.min_future_frames,
            include_following_same_target=args.include_following_same_target,
        )
        dataset_rows.extend(rows)
        split_summaries[split] = summary
        all_manifests.extend(manifests)
        all_source_sequence_rows.extend(source_rows)

    write_dataset_csv(output_root, dataset_rows)
    event_base.write_source_sequence_csv(output_root, all_source_sequence_rows)
    event_base.write_scene_object_tables(output_root, inventories, all_manifests, all_source_sequence_rows)

    shape_summary = {}
    if args.skip_joints:
        for split in ("train", "test"):
            poses_input = torch.load(slices_root / split / "poses_input.pth", map_location="cpu", weights_only=False)
            poses_label = torch.load(slices_root / split / "poses_label.pth", map_location="cpu", weights_only=False)
            shape_summary[split] = {
                "poses_input": list(poses_input.shape),
                "poses_label": list(poses_label.shape),
                "joints_input": None,
                "joints_label": None,
            }
    else:
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
        "window_rule": "input sampled only from floor-context before target onset; future sampled from onset and same-target continuation only",
        "target_rule": "first non-floor/wall target in the first action-onset segment, chosen by action priority then annotation order",
        "max_context_frames": args.max_context_frames,
        "max_future_frames": args.max_future_frames,
        "max_gap_frames": args.max_gap_frames,
        "min_context_frames": args.min_context_frames,
        "min_future_frames": args.min_future_frames,
        "include_following_same_target": args.include_following_same_target,
        "skip_joints": bool(args.skip_joints),
        "splits": split_summaries,
        "tensor_shapes": shape_summary,
        "dataset_csv_rows": len(dataset_rows),
    }
    (output_root / "preprocess_summary.json").write_text(json.dumps(final_summary, indent=2))
    for split, summary in split_summaries.items():
        payload = dict(summary)
        payload["tensor_shapes"] = shape_summary[split]
        (slices_root / split / "summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
