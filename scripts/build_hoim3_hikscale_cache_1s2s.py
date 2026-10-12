#!/usr/bin/env python3
"""Build HIK-style 30/60 direct-24 caches for HOI-M3 HIK-scale."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.hoim3_hikscale import (  # noqa: E402
    FPS,
    FUTURE_FRAMES,
    HOIM3Store,
    INPUT_FRAMES,
    N_JOINTS,
    OBJECT_POINTS,
    PROTOCOL,
    SCENE_POINTS,
    build_observed_sample,
    read_manifest,
    sample_key_sha256,
)


DEFAULT_PREPROCESS_ROOT = Path("data/HOIm3")
DEFAULT_OUTPUT_ROOT = Path(
    "data/HOIm3/"
    "ours_hikscale_direct24_h30f60"
)
DEFAULT_STUDENT_OBJECT_NAME = (
    "final_object_student_hoim3_all_objects_no_cone_points_1024.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=DEFAULT_PREPROCESS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
    )
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--scene-points", type=int, default=SCENE_POINTS)
    parser.add_argument("--student-object-name", default=DEFAULT_STUDENT_OBJECT_NAME)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Smoke-only per-split cap; -1 builds the complete manifest.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def tensorize_cues(cues: dict) -> dict:
    output = {}
    for key, value in cues.items():
        if key in {"torso_directions", "face_directions", "root_velocities", "hand_positions"}:
            continue
        if key == "valid":
            output[key] = value
        else:
            output[key] = [
                round(float(component), 6)
                for component in value.tolist()
            ]
    return output


def build_split(
    split: str,
    args: argparse.Namespace,
    store: HOIM3Store,
) -> dict:
    source_rows = read_manifest(
        args.preprocess_root,
        split,
        protocol=args.protocol,
    )
    full_count = len(source_rows)
    if args.max_samples >= 0:
        source_rows = source_rows[: args.max_samples]
    count = len(source_rows)
    joints_input = torch.empty(
        (count, INPUT_FRAMES, N_JOINTS, 3),
        dtype=torch.float32,
    )
    joints_label = torch.empty(
        (count, FUTURE_FRAMES, N_JOINTS, 3),
        dtype=torch.float32,
    )
    orient_input = torch.empty((count, INPUT_FRAMES, 3), dtype=torch.float32)
    orient_label = torch.empty((count, FUTURE_FRAMES, 3), dtype=torch.float32)
    scenes = torch.empty(
        (count, int(args.scene_points), 3),
        dtype=torch.float32,
    )
    student_objects = torch.zeros(
        (count, OBJECT_POINTS, 3),
        dtype=torch.float32,
    )
    student_mask = torch.zeros((count,), dtype=torch.bool)
    cache_manifest = []
    scene_graphs = []

    for index, record in enumerate(source_rows):
        sample = build_observed_sample(
            store,
            record,
            scene_points=args.scene_points,
        )
        joints_input[index] = torch.from_numpy(sample["joints_input"])
        joints_label[index] = torch.from_numpy(sample["joints_label"])
        orient_input[index] = torch.from_numpy(sample["orient_input"])
        orient_label[index] = torch.from_numpy(sample["orient_label"])
        scenes[index] = torch.from_numpy(sample["scene"])
        selected = sample["candidates"][0] if sample["candidates"] else None
        if selected is not None:
            selected_index = int(selected["object_index"])
            student_objects[index] = torch.from_numpy(
                sample["object_points_by_index"][selected_index]
            )
            student_mask[index] = True

        cache_manifest.append(
            {
                "index": index,
                "split": split,
                "sample_id": sample["sample_id"],
                "sequence": sample["sequence"],
                "target_person_index": sample["target_person_index"],
                "start_frame": sample["start_frame"],
                "last_observed_frame": sample["last_observed_frame"],
                "end_frame_exclusive": sample["end_frame_exclusive"],
                "scene_origin_world": [
                    round(float(value), 6)
                    for value in sample["origin"]
                ],
                "scene": sample["scene_meta"],
                "other_people": sample["other_people"],
                "student_geometry_object": (
                    {
                        "object_id": selected["object_id"],
                        "object_index": selected["object_index"],
                        "name": selected["name"],
                        "geometry_score_no_cone": selected["metrics"][
                            "geometry_score_no_cone"
                        ],
                    }
                    if selected is not None
                    else None
                ),
                "future_person_data_used": False,
                "future_entity_annotations_used": False,
            }
        )
        filtered_by_id = {
            str(candidate["object_id"]): candidate
            for candidate in sample["candidates"]
        }
        graph_candidates = []
        for scene_object in sample["all_objects"]:
            object_id = str(scene_object["object_id"])
            if object_id in filtered_by_id:
                graph_candidates.append(filtered_by_id[object_id])
            else:
                copied = dict(scene_object)
                copied["candidate_reason"] = (
                    "scene_object_outside_cone_or_radius"
                )
                graph_candidates.append(copied)
        graph_candidates.sort(
            key=lambda row: (
                -float(row["metrics"]["geometry_score_no_cone"]),
                float(row["metrics"]["root_bbox_distance_3d"]),
                str(row["object_id"]),
            )
        )
        graph_candidates.append(
            {
                "object_id": "__none__",
                "object_index": -1,
                "name": "none / human-human interaction",
                "label": "NONE",
                "shape": "none",
                "centroid": [0.0, 0.0, 0.0],
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [0.0, 0.0, 0.0],
                "candidate_reason": "no_physical_object_or_human_human_interaction",
                "metrics": {
                    "geometry_score": 0.0,
                    "geometry_score_no_cone": 0.0,
                    "root_bbox_distance_3d": 0.0,
                    "hand_bbox_distance_3d": 0.0,
                    "torso_cone_fraction": 0.0,
                    "face_cone_fraction": 0.0,
                    "root_heading_cone_fraction": 0.0,
                },
            }
        )
        scene_graphs.append(
            {
                "schema_version": "hoim3_hik_action_geometry_scene_graph_v2",
                "sample_index": index,
                "sample_id": sample["sample_id"],
                "split": split,
                "sequence": sample["sequence"],
                "target_person_index": sample["target_person_index"],
                "start_frame": sample["start_frame"],
                "input_frames": list(
                    range(
                        sample["start_frame"],
                        sample["last_observed_frame"] + 1,
                    )
                ),
                "future_frames": list(
                    range(
                        sample["last_observed_frame"] + 1,
                        sample["end_frame_exclusive"],
                    )
                ),
                "last_observed_frame": sample["last_observed_frame"],
                "coordinate_frame": "target_root_centered_hoim3_y_up",
                "scene_origin_world": [
                    round(float(value), 6)
                    for value in sample["origin"]
                ],
                "human_final_frame": tensorize_cues(sample["human_cues"]),
                "other_people": sample["other_people"],
                "candidate_policy": {
                    "scene_radius_m": 3.5,
                    "other_person_radius_m": 2.5,
                    "cone_filtering_enabled": False,
                    "max_candidates": 24,
                    "llm_candidate_coverage": (
                        "all_objects_present_at_last_observed"
                    ),
                    "geometry_rank_uses_cone": False,
                    "future_annotations_used": False,
                },
                "scene_object_count": len(sample["all_objects"]),
                "candidate_count": len(graph_candidates),
                "objects": graph_candidates,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"[{split}] {index + 1}/{count}", flush=True)

    split_dir = args.output_root / "SLICES_888s" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "joints24_input.pth": joints_input.contiguous(),
        "joints24_label.pth": joints_label.contiguous(),
        "orient_input.pth": orient_input.contiguous(),
        "orient_label.pth": orient_label.contiguous(),
        f"scene_points_{args.scene_points}.pth": scenes.contiguous(),
        args.student_object_name: student_objects.contiguous(),
        args.student_object_name.replace(".pth", "_mask.pth"): student_mask,
    }
    for name, tensor in outputs.items():
        atomic_torch(split_dir / name, tensor)
    atomic_json(split_dir / "hoim3_hikscale_cache_manifest.json", cache_manifest)
    atomic_json(
        split_dir
        / args.student_object_name.replace(".pth", "_manifest.json"),
        [
            {
                "index": row["index"],
                "sample_id": row["sample_id"],
                "selection": row["student_geometry_object"],
            }
            for row in cache_manifest
        ],
    )

    graph_dir = args.output_root / "scene_graphs" / split
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "samples.jsonl"
    temporary_graph = graph_path.with_suffix(".jsonl.tmp")
    with temporary_graph.open("w", encoding="utf-8") as handle:
        for graph in scene_graphs:
            handle.write(
                json.dumps(graph, ensure_ascii=False, allow_nan=False) + "\n"
            )
    os.replace(temporary_graph, graph_path)

    people_counts = [
        int(row["other_people"]["num_people"])
        for row in cache_manifest
    ]
    candidate_counts = [int(row["candidate_count"]) for row in scene_graphs]
    summary = {
        "split": split,
        "protocol": args.protocol,
        "samples": count,
        "full_manifest_samples": full_count,
        "sample_key_sha256": sample_key_sha256(source_rows),
        "fps": FPS,
        "input_frames": INPUT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "joint_count": N_JOINTS,
        "representation": "native_hoim3_direct24_no_smplx",
        "coordinate_frame": "target_last_observed_root_centered_y_up",
        "scene_rule": (
            "HIK-compatible observed object crop plus bounded last-frame "
            "non-target 24-joint point sets"
        ),
        "scene_points": int(args.scene_points),
        "object_points": OBJECT_POINTS,
        "student_object_rule": "all-object observed no-cone geometry top1",
        "llm_scene_graph_object_rule": (
            "all last-observed present objects; no cone filtering"
        ),
        "student_valid_objects": int(student_mask.sum().item()),
        "people_histogram": dict(sorted(Counter(people_counts).items())),
        "candidate_count_histogram": dict(
            sorted(Counter(candidate_counts).items())
        ),
        "future_person_data_used": False,
        "future_entity_annotations_used": False,
        "scene_graph_path": str(graph_path),
        "shapes": {
            name: list(tensor.shape)
            for name, tensor in outputs.items()
        },
    }
    atomic_json(split_dir / "hoim3_hikscale_cache_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.scene_points != SCENE_POINTS:
        raise ValueError(
            "The HIK/HOI-M3 comparison requires exactly 1000 scene points"
        )
    if not args.student_object_name.endswith(".pth"):
        raise ValueError("--student-object-name must end with .pth")
    store = HOIM3Store(args.preprocess_root)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    summaries = {
        split: build_split(split, args, store)
        for split in splits
    }
    atomic_json(args.output_root / "cache_summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
