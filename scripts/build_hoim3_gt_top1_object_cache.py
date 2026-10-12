#!/usr/bin/env python3
"""Build a split-aligned oracle GT Top-1 object cache for HOI-M3.

The HOI-M3 interaction manifest may mark multiple objects as interacting in
one 3-second forecasting sample.  SIF3D consumes one 1024-point object, so
this exporter uses a deterministic privileged Top-1 rule:

1. prefer future-2s interaction objects;
2. if none exist, use observed-1s interaction objects;
3. among last-observed visible GT objects, rank contact/manipulation strength;
4. use a false mask only for human-human-only interaction samples.

Only the object identity is privileged.  Geometry is always taken at the
last observed frame and centered at the target's last-observed root.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.hoim3_hikscale import (  # noqa: E402
    HOIM3Store,
    OBJECT_POINTS,
    PROTOCOL,
    read_manifest,
    sample_key,
)


PREPROCESS_ROOT = Path("data/HOIm3")
CACHE_ROOT = (
    PREPROCESS_ROOT
    / "ours_hikscale_direct24_h30f60"
    / "SLICES_888s"
)
OUTPUT_NAME = (
    "final_object_gt_top1_interaction_strength_hoim3_points_1024.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-root", type=Path, default=PREPROCESS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def finite_float(value: Any, default: float) -> float:
    if value is None:
        return default
    result = float(value)
    return result if np.isfinite(result) else default


def strength(entity: dict[str, Any], segment: str) -> tuple:
    stats = entity[segment]
    body_contacts = int(stats["body_contact_frames"])
    hand_contacts = int(stats["hand_contact_frames"])
    translation = finite_float(
        stats["translation_displacement_m"],
        0.0,
    )
    rotation = finite_float(
        stats["rotation_displacement_deg"],
        0.0,
    )
    body_distance = finite_float(
        stats["body_min_distance_m"],
        float("inf"),
    )
    hand_distance = finite_float(
        stats["hand_min_distance_m"],
        float("inf"),
    )
    return (
        max(body_contacts, hand_contacts),
        hand_contacts,
        body_contacts,
        translation,
        rotation,
        -min(body_distance, hand_distance),
        -int(entity["index"]),
    )


def gt_candidates(
    row: dict[str, Any],
    segment: str,
) -> list[dict[str, Any]]:
    flag = f"{segment}_interaction"
    return [
        entity
        for entity in row["entities"]
        if (
            entity["type"] == "object"
            and bool(entity.get(flag))
            and bool(entity.get("observed_at_last_frame"))
        )
    ]


def select_gt_object(row: dict[str, Any]):
    future = gt_candidates(row, "future")
    observed = gt_candidates(row, "observed")
    if future:
        segment = "future"
        candidates = future
    elif observed:
        segment = "observed"
        candidates = observed
    else:
        return None, None, []
    ranked = sorted(
        candidates,
        key=lambda entity: strength(entity, segment),
        reverse=True,
    )
    return ranked[0], segment, ranked


def compact_candidate(
    entity: dict[str, Any],
    segment: str,
) -> dict[str, Any]:
    stats = entity[segment]
    return {
        "object_id": f"o{int(entity['index'])}",
        "object_index": int(entity["index"]),
        "object_name": str(entity["name"]),
        "segment": segment,
        "strength": list(strength(entity, segment)),
        "body_contact_frames": int(stats["body_contact_frames"]),
        "hand_contact_frames": int(stats["hand_contact_frames"]),
        "translation_displacement_m": finite_float(
            stats["translation_displacement_m"],
            0.0,
        ),
        "rotation_displacement_deg": finite_float(
            stats["rotation_displacement_deg"],
            0.0,
        ),
        "body_min_distance_m": finite_float(
            stats["body_min_distance_m"],
            float("inf"),
        ),
        "hand_min_distance_m": finite_float(
            stats["hand_min_distance_m"],
            float("inf"),
        ),
    }


def build_split(
    args: argparse.Namespace,
    store: HOIM3Store,
    split: str,
) -> dict[str, Any]:
    rows = read_manifest(args.preprocess_root, split, args.protocol)
    points = torch.zeros(
        (len(rows), OBJECT_POINTS, 3),
        dtype=torch.float32,
    )
    mask = torch.zeros((len(rows),), dtype=torch.bool)
    manifest = []
    segment_counts = Counter()
    selected_counts = Counter()
    candidate_count_histogram = Counter()
    for index, row in enumerate(rows):
        selected, segment, ranked = select_gt_object(row)
        entry = {
            "index": index,
            "sample_id": sample_key(row),
            "split": split,
            "sequence": str(row["sequence"]),
            "target_person_index": int(row["target_person_index"]),
            "start_frame": int(row["start_frame"]),
            "last_observed_frame": int(row["last_observed_frame"]),
            "interaction_category": str(row["interaction_category"]),
            "selection_rule": (
                "future_gt_first_then_observed_gt; strongest by "
                "max_contact, hand_contact, body_contact, translation, "
                "rotation, min_distance, object_index"
            ),
            "future_geometry_used": False,
            "gt_identity_used": True,
        }
        if selected is None:
            entry.update(
                {
                    "status": "no_gt_physical_object_human_only",
                    "selected_object_id": "__none__",
                    "selected_object_index": -1,
                    "selected_object_name": (
                        "none / human-human interaction"
                    ),
                    "selection_segment": None,
                    "ranked_gt_candidates": [],
                }
            )
            segment_counts["none_human_only"] += 1
        else:
            object_index = int(selected["index"])
            arrays = store.sequence(str(row["sequence"]))
            last = int(row["last_observed_frame"])
            target = int(row["target_person_index"])
            origin = np.asarray(
                arrays.joints_by_person[target][last, 0],
                dtype=np.float32,
            )
            object_points = (
                store.object_world_points(
                    arrays,
                    last,
                    object_index,
                )
                - origin[None, :]
            ).astype(np.float32, copy=False)
            if object_points.shape != (OBJECT_POINTS, 3):
                raise ValueError(
                    f"{sample_key(row)} object points "
                    f"{object_points.shape}"
                )
            points[index] = torch.from_numpy(object_points)
            mask[index] = True
            entry.update(
                {
                    "status": "ok",
                    "selected_object_id": f"o{object_index}",
                    "selected_object_index": object_index,
                    "selected_object_name": str(selected["name"]),
                    "selection_segment": segment,
                    "ranked_gt_candidates": [
                        compact_candidate(entity, segment)
                        for entity in ranked
                    ],
                }
            )
            segment_counts[segment] += 1
            selected_counts[str(selected["name"])] += 1
            candidate_count_histogram[len(ranked)] += 1
    if not torch.isfinite(points).all():
        raise ValueError(f"{split}: GT object cache contains non-finite values")
    points[~mask] = 0
    split_dir = args.cache_root / split
    points_path = split_dir / args.output_name
    mask_path = split_dir / args.output_name.replace(".pth", "_mask.pth")
    manifest_path = split_dir / args.output_name.replace(
        ".pth",
        "_manifest.json",
    )
    atomic_torch(points_path, points.contiguous())
    atomic_torch(mask_path, mask)
    atomic_json(manifest_path, manifest)
    summary = {
        "split": split,
        "samples": len(rows),
        "shape": list(points.shape),
        "valid_objects": int(mask.sum().item()),
        "none_human_only": int((~mask).sum().item()),
        "segment_counts": dict(sorted(segment_counts.items())),
        "multi_gt_candidate_histogram": {
            str(key): value
            for key, value in sorted(candidate_count_histogram.items())
        },
        "selected_object_counts": dict(sorted(selected_counts.items())),
        "points_path": str(points_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "coordinate_frame": "target_last_observed_root_centered_y_up",
        "points_time": "last_observed_frame",
        "future_geometry_used": False,
        "gt_identity_used": True,
        "oracle": True,
        "deployment_valid": False,
    }
    atomic_json(
        split_dir / args.output_name.replace(".pth", "_summary.json"),
        summary,
    )
    return summary


def main() -> None:
    args = parse_args()
    if not args.output_name.endswith(".pth"):
        raise ValueError("--output-name must end with .pth")
    store = HOIM3Store(args.preprocess_root)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    summaries = {
        split: build_split(args, store, split)
        for split in splits
    }
    atomic_json(
        args.cache_root.parent / "gt_top1_object_cache_summary.json",
        summaries,
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
