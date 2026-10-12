#!/usr/bin/env python3
"""Build split-aligned HIK H25F50 caches from native poses3d (29 joints)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.hik_raw_1s2s import (  # noqa: E402
    RawHIKIndex,
    hik_orientations_to_model,
    hik_points_to_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/HIK/SAST/data"
        ),
    )
    parser.add_argument(
        "--context-root",
        type=Path,
        default=Path(
            "data/HIK/"
            "hik_preprocessed/H25F50_nonwalking"
        ),
    )
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik/scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/HIK/"
            "hik_preprocessed/H25F50_nonwalking_direct29"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_by_id(path: Path):
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[row["sample_id"]] = row
    return output


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_write_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_split(split, args, index):
    context_dir = args.context_root / "SLICES_888s" / split
    context_manifest = read_json(
        context_dir / "hik_h25f50_cache_manifest.json"
    )
    if args.limit >= 0:
        context_manifest = context_manifest[: args.limit]
    graphs = read_jsonl_by_id(args.scene_graph_root / split / "samples.jsonl")
    count = len(context_manifest)
    joints = torch.empty((count, 75, 29, 3), dtype=torch.float32)
    orientations = torch.empty((count, 75, 3), dtype=torch.float32)
    output_manifest = []

    for output_index, source_row in enumerate(context_manifest):
        sample_id = source_row["sample_id"]
        if sample_id not in graphs:
            raise KeyError(f"Scene graph missing sample {sample_id}")
        graph = graphs[sample_id]
        frames = tuple(graph["input_frames"]) + tuple(graph["future_frames"])
        if len(frames) != 75:
            raise ValueError(f"{sample_id} has {len(frames)} frames")
        ref = index.find_complete_track(graph["kitchen"], graph["pid"], frames)
        if ref is None:
            raise KeyError(f"No complete raw track for {sample_id}")
        rows = index.load_rows(ref, frames)

        raw_joints = np.asarray(rows["poses3d"], dtype=np.float32)
        converted = hik_points_to_model(raw_joints)
        origin = converted[24, 0].copy()
        converted = converted - origin[None, None, :]
        transforms = np.asarray(rows["transforms"], dtype=np.float32)
        orientation = hik_orientations_to_model(transforms[:, 3:6])
        if converted.shape != (75, 29, 3):
            raise ValueError(
                f"{sample_id} direct joints have shape {converted.shape}"
            )
        if orientation.shape != (75, 3):
            raise ValueError(
                f"{sample_id} orientation has shape {orientation.shape}"
            )
        if not np.isfinite(converted).all() or not np.isfinite(
            orientation
        ).all():
            raise ValueError(f"{sample_id} contains non-finite values")
        if not np.allclose(converted[24, 0], 0.0, atol=1e-6):
            raise ValueError(f"{sample_id} last observed root is not zero")

        joints[output_index] = torch.from_numpy(converted)
        orientations[output_index] = torch.from_numpy(orientation)
        output_manifest.append(
            {
                "index": output_index,
                "sample_id": sample_id,
                "split": split,
                "kitchen": graph["kitchen"],
                "pid": graph["pid"],
                "event_action": graph["event_action"],
                "future_anchor": graph["future_anchor"],
                "track": ref.path.name,
                "source_context_index": source_row["index"],
                "representation": "native_hik_poses3d_29x3",
                "origin": "target_joint0_at_last_observed_frame",
            }
        )
        if (output_index + 1) % 100 == 0 or output_index + 1 == count:
            print(f"[{split}] {output_index + 1}/{count}", flush=True)

    output_dir = args.output_root / "SLICES_888s" / split
    atomic_torch_save(joints[:, :25].contiguous(), output_dir / "joints29_input.pth")
    atomic_torch_save(joints[:, 25:].contiguous(), output_dir / "joints29_label.pth")
    atomic_torch_save(
        orientations[:, :25].contiguous(),
        output_dir / "orient_input.pth",
    )
    atomic_torch_save(
        orientations[:, 25:].contiguous(),
        output_dir / "orient_label.pth",
    )
    atomic_write_json(
        output_manifest,
        output_dir / "hik_direct29_cache_manifest.json",
    )
    summary = {
        "split": split,
        "samples": count,
        "representation": "native_hik_poses3d_29x3",
        "fps": 25,
        "input_frames": 25,
        "future_frames": 50,
        "joint_count": 29,
        "coordinate_system": "model_y_up_scene_centered",
        "action_counts": dict(
            sorted(Counter(row["event_action"] for row in output_manifest).items())
        ),
        "shapes": {
            "joints29_input": [count, 25, 29, 3],
            "joints29_label": [count, 50, 29, 3],
            "orient_input": [count, 25, 3],
            "orient_label": [count, 50, 3],
        },
    }
    atomic_write_json(summary, output_dir / "hik_direct29_cache_summary.json")
    return summary


def main():
    args = parse_args()
    index = RawHIKIndex(args.raw_root)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    summaries = {
        split: build_split(split, args, index)
        for split in splits
    }
    atomic_write_json(summaries, args.output_root / "direct29_summary.json")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
