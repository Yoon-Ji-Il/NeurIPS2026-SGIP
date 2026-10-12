#!/usr/bin/env python
"""Build PROX/COINS SIF3D tensors with approach/walking context.

This variant keeps the same tensor contract as ``preprocess_prox_coins_to_gimo.py``
but changes the sample definition:

  - choose a target interaction segment whose final frame has a non-floor/wall
    target object;
  - extend backward within the same original sequence through walking/standing
    floor context followed by immediate same-target transition/action segments;
  - uniformly downsample the extended segment to 6 input + 10 output frames.

The original short-action cache is left untouched by default.
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


INPUT_LEN = base.INPUT_LEN
OUTPUT_LEN = base.OUTPUT_LEN
TOTAL_LEN = base.TOTAL_LEN
FLOOR_WALL = base.FLOOR_WALL

CONTEXT_OBJECT_NAMES = {"floor"}
CONTEXT_LABEL_PREFIXES = ("walk", "side walk", "stand", "a pose", "step")
TRANSITION_PRIORITY_PREFIXES = (
    "sit down",
    "lie down",
    "stand up",
    "sit up",
    "step down",
    "step up",
    "step back",
    "move",
    "restfoot",
    "turnover",
    "touch",
    "jump",
)


def action_segments_by_sequence(records: list[dict[str, Any]]) -> dict[str, list[list[dict[str, Any]]]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sequence[record["sequence"]].append(record)

    segments_by_sequence: dict[str, list[list[dict[str, Any]]]] = {}
    for sequence in sorted(by_sequence):
        rows = sorted(by_sequence[sequence], key=lambda row: int(row["frame_idx"]))
        segments: list[list[dict[str, Any]]] = []
        start = 0
        for idx in range(1, len(rows) + 1):
            if idx == len(rows) or base.action_key(rows[idx]) != base.action_key(rows[start]):
                segments.append(rows[start:idx])
                start = idx
        segments_by_sequence[sequence] = segments
    return segments_by_sequence


def label_object_name(label: str) -> str:
    return base.object_name_from_label(label).strip().lower()


def label_prefix(label: str) -> str:
    return str(label).split("-", 1)[0].strip().lower()


def label_is_floor_wall(label: str) -> bool:
    return label_object_name(label) in FLOOR_WALL


def transition_priority(label: str) -> tuple[int, int, str]:
    prefix = label_prefix(label)
    for idx, candidate in enumerate(TRANSITION_PRIORITY_PREFIXES):
        if prefix.startswith(candidate):
            return (0, idx, str(label))
    return (1, len(TRANSITION_PRIORITY_PREFIXES), str(label))


def canonical_interactions(record: dict[str, Any]) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    raw_pairs = [
        (str(label), int(object_idx))
        for label, object_idx in zip(record["interaction_labels"], record["interaction_obj_idx"])
    ]
    non_floor_pairs = [(label, object_idx) for label, object_idx in raw_pairs if not label_is_floor_wall(label)]
    pairs = non_floor_pairs if non_floor_pairs else raw_pairs

    by_object: dict[int, list[str]] = defaultdict(list)
    for label, object_idx in pairs:
        by_object[object_idx].append(label)

    canonical_pairs: list[tuple[str, int]] = []
    dropped: list[dict[str, Any]] = []
    for object_idx, labels in by_object.items():
        selected = sorted(labels, key=transition_priority)[0]
        canonical_pairs.append((selected, object_idx))
        for label in labels:
            if label != selected:
                dropped.append(
                    {
                        "label": label,
                        "object_idx": object_idx,
                        "reason": "lower_priority_same_object_label",
                    }
                )

    for label, object_idx in raw_pairs:
        if (label, object_idx) not in pairs:
            dropped.append(
                {
                    "label": label,
                    "object_idx": object_idx,
                    "reason": "context_label_removed_from_mixed_frame",
                }
            )

    canonical_pairs.sort(key=lambda pair: pair[1])
    labels = [label for label, _ in canonical_pairs]
    object_indices = [object_idx for _, object_idx in canonical_pairs]
    return labels, object_indices, dropped


def canonicalize_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_records: list[dict[str, Any]] = []
    dropped_counter: Counter[str] = Counter()
    mixed_frame_count = 0
    changed_frame_count = 0

    for record in records:
        labels, object_indices, dropped = canonical_interactions(record)
        row = dict(record)
        raw_labels = list(record["interaction_labels"])
        raw_object_indices = [int(value) for value in record["interaction_obj_idx"]]
        row["raw_interaction_labels"] = raw_labels
        row["raw_interaction_obj_idx"] = raw_object_indices
        row["interaction_labels"] = labels
        row["interaction_obj_idx"] = object_indices
        row["canonical_dropped_interactions"] = dropped
        canonical_records.append(row)

        if len(raw_labels) > 1:
            mixed_frame_count += 1
        if labels != raw_labels or object_indices != raw_object_indices:
            changed_frame_count += 1
        for item in dropped:
            dropped_counter[str(item["label"])] += 1

    summary = {
        "canonicalization_rule": "drop floor/wall context labels from mixed target frames; prefer transition labels over steady labels for the same object",
        "transition_priority_prefixes": list(TRANSITION_PRIORITY_PREFIXES),
        "mixed_frame_count": mixed_frame_count,
        "changed_frame_count": changed_frame_count,
        "dropped_label_counts": dict(dropped_counter.most_common()),
    }
    return canonical_records, summary


def valid_target_from_record(
    record: dict[str, Any],
    scene_instance_info: dict[int, base.InstanceInfo],
) -> dict[str, Any] | None:
    target_info = base.choose_target_object(record, scene_instance_info)
    return target_info["target"]


def has_same_object_idx(segment: list[dict[str, Any]], object_idx: int) -> bool:
    for record in (segment[0], segment[-1]):
        for label, row_object_idx in zip(record["interaction_labels"], record["interaction_obj_idx"]):
            if label_is_floor_wall(label):
                continue
            if int(row_object_idx) == int(object_idx):
                return True
    return False


def is_floor_context_segment(segment: list[dict[str, Any]]) -> bool:
    labels = [str(label).strip().lower() for label in segment[0]["interaction_labels"]]
    if not labels:
        return False
    for label in labels:
        if label_object_name(label) not in CONTEXT_OBJECT_NAMES:
            return False
        if not label.startswith(CONTEXT_LABEL_PREFIXES):
            return False
    return True


def segment_gap(prev_segment: list[dict[str, Any]], next_segment: list[dict[str, Any]]) -> int:
    return int(next_segment[0]["frame_idx"]) - int(prev_segment[-1]["frame_idx"])


def build_long_segments(
    records: list[dict[str, Any]],
    scene_instance_info: dict[str, dict[int, base.InstanceInfo]],
    max_context_frames: int,
    max_gap_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments_by_sequence = action_segments_by_sequence(records)
    samples: list[dict[str, Any]] = []
    target_counter: Counter[str] = Counter()
    context_label_counter: Counter[str] = Counter()
    raw_segment_count = sum(len(segments) for segments in segments_by_sequence.values())
    target_segment_count = 0

    for sequence, segments in segments_by_sequence.items():
        for segment_idx, target_segment in enumerate(segments):
            scene = target_segment[-1]["scene_name"]
            target = valid_target_from_record(target_segment[-1], scene_instance_info[scene])
            if target is None:
                continue
            target_segment_count += 1
            target_object_idx = int(target["object_idx"])
            target_label_name = str(target.get("semantic_label_name") or target["label_object_name"]).lower()

            first_segment_idx = segment_idx
            context_frames = 0
            allow_older_same_target = True
            scan_idx = segment_idx - 1
            while scan_idx >= 0:
                candidate = segments[scan_idx]
                next_segment = segments[first_segment_idx]
                if segment_gap(candidate, next_segment) > max_gap_frames:
                    break

                is_context = is_floor_context_segment(candidate)
                is_same_target = has_same_object_idx(candidate, target_object_idx)
                if is_same_target and not allow_older_same_target:
                    break
                if not (is_context or is_same_target):
                    break

                if context_frames + len(candidate) > max_context_frames:
                    break

                if is_context:
                    allow_older_same_target = False
                    for label in candidate[0]["interaction_labels"]:
                        context_label_counter[str(label)] += len(candidate)
                first_segment_idx = scan_idx
                context_frames += len(candidate)
                scan_idx -= 1

            long_segment: list[dict[str, Any]] = []
            for include_idx in range(first_segment_idx, segment_idx + 1):
                long_segment.extend(segments[include_idx])
            if len(long_segment) < TOTAL_LEN:
                continue

            target_counter[target_label_name] += 1
            samples.append(
                {
                    "sequence": sequence,
                    "scene": scene,
                    "first_segment_idx": first_segment_idx,
                    "target_segment_idx": segment_idx,
                    "segment": long_segment,
                    "target_segment": target_segment,
                    "target_info": base.choose_target_object(target_segment[-1], scene_instance_info[scene]),
                    "context_frames": int(len(long_segment) - len(target_segment)),
                    "num_action_segments": int(segment_idx - first_segment_idx + 1),
                }
            )

    summary = {
        "records": len(records),
        "raw_action_segments": raw_segment_count,
        "target_action_segments": target_segment_count,
        "long_segments_len_at_least_16": len(samples),
        "target_label_counts": dict(target_counter.most_common()),
        "context_label_frame_counts": dict(context_label_counter.most_common()),
        "max_context_frames": max_context_frames,
        "max_gap_frames": max_gap_frames,
        "context_label_prefixes": list(CONTEXT_LABEL_PREFIXES),
    }
    return samples, summary


def sample_segment(segment: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    sampled_idx = np.rint(np.linspace(0, len(segment) - 1, TOTAL_LEN)).astype(np.int64)
    return [segment[int(idx)] for idx in sampled_idx], [int(segment[int(idx)]["frame_idx"]) for idx in sampled_idx]


def displacement_stats(sampled_poses: list[np.ndarray]) -> dict[str, float]:
    poses = np.stack(sampled_poses, axis=0)
    input_last = poses[INPUT_LEN - 1, 3:6]
    future = poses[INPUT_LEN:, 3:6]
    return {
        "future_root_displacement_mean": float(np.linalg.norm(future - input_last[None, :], axis=1).mean()),
        "future_root_displacement_final": float(np.linalg.norm(future[-1] - input_last)),
    }


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


def build_split_long(
    split: str,
    records_path: Path,
    output_split_root: Path,
    inventories: dict[str, dict[str, Any]],
    segmentation_dir: Path,
    sample_points: int,
    object_points: int,
    max_context_frames: int,
    max_gap_frames: int,
    drop_no_target: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_records = pickle.loads(records_path.read_bytes())
    records, canonicalization_summary = canonicalize_records(raw_records)
    output_split_root.mkdir(parents=True, exist_ok=True)
    scene_instance_info = {
        scene: base.parse_instance_txt(segmentation_dir / f"{scene}.txt") for scene in inventories
    }
    long_samples, segment_summary = build_long_segments(
        records,
        scene_instance_info=scene_instance_info,
        max_context_frames=max_context_frames,
        max_gap_frames=max_gap_frames,
    )

    poses_all: list[np.ndarray] = []
    scene_points_all: list[np.ndarray] = []
    object_points_all: list[np.ndarray] = []
    object_mask_all: list[bool] = []
    manifest: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    skipped_no_target = 0
    segment_lengths: list[float] = []
    context_lengths: list[float] = []
    final_displacements: list[float] = []
    mean_displacements: list[float] = []

    for sample in long_samples:
        segment = sample["segment"]
        sampled, sampled_frame_indices = sample_segment(segment)
        last_record = sampled[-1]
        scene = sample["scene"]
        sequence = sample["sequence"]
        target_info = base.choose_target_object(last_record, scene_instance_info[scene])
        target = target_info["target"]
        has_target = target is not None
        if drop_no_target and not has_target:
            skipped_no_target += 1
            continue

        sample_index = len(poses_all)
        sample_id = (
            f"{split}_{sample_index:05d}_{sequence}"
            f"_long_f{segment[0]['frame_idx']}_f{segment[-1]['frame_idx']}"
        )
        poses = np.stack([base.frame_pose(record) for record in sampled], axis=0)
        poses_all.append(poses)

        disp = displacement_stats([base.frame_pose(record) for record in sampled])
        final_displacements.append(disp["future_root_displacement_final"])
        mean_displacements.append(disp["future_root_displacement_mean"])
        segment_lengths.append(float(len(segment)))
        context_lengths.append(float(sample["context_frames"]))

        scene_xyz = inventories[scene]["xyz"]
        scene_points_all.append(base.sample_points_fn(scene_xyz, sample_points, base.stable_seed("scene", scene)))

        object_idx = int(target["object_idx"]) if has_target else None
        if object_idx is None:
            object_points_all.append(np.zeros((object_points, 3), dtype=np.float32))
            object_mask_all.append(False)
        else:
            obj_xyz = inventories[scene]["points_by_object_idx"].get(object_idx, np.zeros((0, 3), dtype=np.float32))
            has_points = obj_xyz.shape[0] > 0
            object_points_all.append(base.sample_points_fn(obj_xyz, object_points, base.stable_seed("object", scene, object_idx)))
            object_mask_all.append(has_points)
            if not has_points:
                has_target = False

        manifest.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "split": split,
                "scene_name": scene,
                "sequence": sequence,
                "segment_start_frame_idx": int(segment[0]["frame_idx"]),
                "segment_end_frame_idx": int(segment[-1]["frame_idx"]),
                "target_segment_start_frame_idx": int(sample["target_segment"][0]["frame_idx"]),
                "target_segment_end_frame_idx": int(sample["target_segment"][-1]["frame_idx"]),
                "segment_num_frames": len(segment),
                "target_segment_num_frames": len(sample["target_segment"]),
                "context_num_frames": int(sample["context_frames"]),
                "num_action_segments": int(sample["num_action_segments"]),
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
                "final_frame_interactions": target_info["candidates"],
                "target_object": target,
                "target_object_mask": bool(object_mask_all[-1]),
                "target_rule": "last sampled frame, last non-floor/wall interaction",
                "window_rule": "walking/standing floor context followed by immediate same-target action segments, then uniform 16-frame downsampling",
                "future_root_displacement_final": disp["future_root_displacement_final"],
                "future_root_displacement_mean": disp["future_root_displacement_mean"],
            }
        )
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
    (output_split_root / "segment_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_split_root / f"final_object_gt_points_{object_points}_manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "split": split,
        "raw_records": len(raw_records),
        "canonicalization": canonicalization_summary,
        **segment_summary,
        "samples": len(poses_all),
        "object_valid_samples": int(sum(object_mask_all)),
        "object_missing_or_floor_wall_samples": int(len(object_mask_all) - sum(object_mask_all)),
        "skipped_no_target": skipped_no_target,
        "segment_num_frames_stats": percentile_payload(segment_lengths),
        "context_num_frames_stats": percentile_payload(context_lengths),
        "future_root_displacement_final_stats": percentile_payload(final_displacements),
        "future_root_displacement_mean_stats": percentile_payload(mean_displacements),
    }
    (output_split_root / "summary_no_joints_yet.json").write_text(json.dumps(summary, indent=2))
    return dataset_rows, summary


def write_dataset_csv(output_root: Path, rows: list[dict[str, Any]]) -> None:
    with (output_root / "dataset.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sequence_path", "start_frame", "end_frame", "scene", "transformation", "training"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-root", type=Path, default=Path("data/PROX"))
    parser.add_argument("--coins-root", type=Path, default=Path("data/PROX/COINS"))
    parser.add_argument("--scenes-semantics-root", type=Path, default=Path("data/PROX/scenes_semantics"))
    parser.add_argument("--output-root", type=Path, default=Path("data/PROX/PROX_COINS_GIMO_LONG_V2"))
    parser.add_argument("--smplx-path", type=Path, default=Path("data/body_models"))
    parser.add_argument("--vposer-path", type=Path, default=Path("./vposer_v1_0"))
    parser.add_argument("--sample-points", type=int, default=32768)
    parser.add_argument("--object-points", type=int, default=1024)
    parser.add_argument("--smplx-batch-size", type=int, default=256)
    parser.add_argument("--max-context-frames", type=int, default=240)
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--drop-no-target", action="store_true")
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
    for split in ("train", "test"):
        rows, summary = build_split_long(
            split=split,
            records_path=semantics_dir / f"{split}.pkl",
            output_split_root=slices_root / split,
            inventories=inventories,
            segmentation_dir=segmentation_dir,
            sample_points=args.sample_points,
            object_points=args.object_points,
            max_context_frames=args.max_context_frames,
            max_gap_frames=args.max_gap_frames,
            drop_no_target=args.drop_no_target,
        )
        dataset_rows.extend(rows)
        split_summaries[split] = summary

    write_dataset_csv(output_root, dataset_rows)

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
        "window_rule": "walking/standing floor context followed by immediate same-target action segments, then uniform 16-frame downsampling",
        "target_rule": "last sampled frame, last non-floor/wall interaction; mask false if no valid object",
        "max_context_frames": args.max_context_frames,
        "max_gap_frames": args.max_gap_frames,
        "splits": split_summaries,
        "tensor_shapes": shape_summary,
        "dataset_csv_rows": len(dataset_rows),
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
