#!/usr/bin/env python3
"""Create sample-wise segmentation-noise inventories for train and test.

All conditions are materialized per sample.  Unchanged PLY files are hard-linked
by default so sample-wise inventories remain cheap while each directory is a
complete, independently readable OpenIns3D catalog.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from create_segmentation_rebuttal_variants import (
    partial_noisy_vertices,
    read_vertices,
    split_vertices_pca,
    stable_seed,
    write_vertices,
    xyz,
    xyz_only_vertices,
)


DEFAULT_SOURCE_ROOT = Path(
    "data/OpenIns3D/output/results_demo_rebuttal"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/OpenIns3D/output/results_segmentation_noise_samplewise"
)
DEFAULT_TRAIN_MANIFEST = Path(
    "data/GIMO/SLICES_888s/train/"
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024_manifest.json"
)
DEFAULT_TEST_MANIFEST = Path(
    "data/GIMO/SLICES_888s/test/"
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024_manifest.json"
)
DEFAULT_SEED = 20260725
CONDITIONS = (
    "random_instance_deletion_10pct",
    "random_instance_deletion_20pct",
    "target_neighbor_merge",
    "target_two_way_split",
    "partial_noisy_target_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--link-mode", choices=("hardlink", "copy"), default="hardlink"
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sample_id(sample: dict[str, Any]) -> str:
    return "|".join(str(value) for value in sample["sample_key"])


def sample_uid(split: str, sample: dict[str, Any]) -> str:
    return f"{split}:{int(sample['index']):04d}:{sample_id(sample)}"


def sample_slug(sample: dict[str, Any]) -> str:
    parts = [
        f"sample_{int(sample['index']):04d}",
        str(sample["scene_name"]),
        str(sample["sequence_name"]),
        f"{sample['start_frame']}_{sample['end_frame']}",
    ]
    return "__".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "-", part) for part in parts
    )


def load_catalog(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "label_pointcloud_export.json"
    catalog = read_json(path)
    instances = catalog.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"Missing instance list: {path}")
    ids = [str(entry["instance_id"]) for entry in instances]
    paths = [str(entry["ply_path"]) for entry in instances]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance IDs: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate PLY paths: {path}")
    for relative_path in paths:
        source = scene_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
    return catalog


def entry_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry["instance_id"]): entry for entry in catalog["instances"]
    }


def deletion_count(num_instances: int, fraction: float) -> int:
    count = int(math.floor(num_instances * fraction + 0.5))
    return min(count, max(0, num_instances - 1))


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def materialize(
    source_scene_dir: Path,
    output_dir: Path,
    catalog: dict[str, Any],
    link_mode: str,
    replacements: dict[str, np.ndarray] | None = None,
) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "instances").mkdir()
    replacement_map = replacements or {}
    for entry in catalog["instances"]:
        instance_id = str(entry["instance_id"])
        destination = output_dir / str(entry["ply_path"])
        if instance_id in replacement_map:
            write_vertices(destination, replacement_map[instance_id])
        else:
            link_or_copy(
                source_scene_dir / str(entry["ply_path"]),
                destination,
                link_mode,
            )
    write_json(output_dir / "label_pointcloud_export.json", catalog)


def source_cache(
    source_root: Path, samples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for scene_name in sorted({str(sample["scene_name"]) for sample in samples}):
        scene_dir = source_root / scene_name
        catalog = load_catalog(scene_dir)
        vertices: dict[str, np.ndarray] = {}
        centroids: dict[str, np.ndarray] = {}
        for instance_id, entry in entry_map(catalog).items():
            values = read_vertices(scene_dir / str(entry["ply_path"]))
            vertices[instance_id] = values
            centroids[instance_id] = xyz(values).mean(axis=0)
        cache[scene_name] = {
            "scene_dir": scene_dir,
            "catalog": catalog,
            "vertices": vertices,
            "centroids": centroids,
        }
    return cache


def common_manifest(
    *,
    condition: str,
    split: str,
    sample: dict[str, Any],
    output_dir: Path,
    source_scene: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": condition,
        "split": split,
        "sample_index": int(sample["index"]),
        "sample_id": sample_id(sample),
        "sample_uid": sample_uid(split, sample),
        "sample_key": sample["sample_key"],
        "scene_name": sample["scene_name"],
        "sequence_name": sample["sequence_name"],
        "start_frame": str(sample["start_frame"]),
        "end_frame": str(sample["end_frame"]),
        "gt_target_instance_id": sample["gt_object_id"],
        "output_directory": str(output_dir),
        "num_instances_before": len(source_scene["catalog"]["instances"]),
    }


def build_deletions(
    *,
    split: str,
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_root: Path,
    global_seed: int,
    link_mode: str,
) -> list[dict[str, Any]]:
    catalog = source_scene["catalog"]
    ids = [str(entry["instance_id"]) for entry in catalog["instances"]]
    seed = stable_seed(
        global_seed,
        split,
        *sample["sample_key"],
        int(sample["index"]),
        "random_instance_deletion",
    )
    order = list(ids)
    random.Random(seed).shuffle(order)
    records: list[dict[str, Any]] = []
    previous_deleted: set[str] = set()
    for percent in (10, 20):
        count = deletion_count(len(ids), percent / 100.0)
        deleted_ids = order[:count]
        deleted = set(deleted_ids)
        if not previous_deleted.issubset(deleted):
            raise AssertionError("10% deletion must be nested inside 20%")
        previous_deleted = deleted
        condition = f"random_instance_deletion_{percent}pct"
        output_dir = (
            output_root / condition / split / "samples" / sample_slug(sample)
        )
        corrupted = deepcopy(catalog)
        corrupted["instances"] = [
            entry
            for entry in corrupted["instances"]
            if str(entry["instance_id"]) not in deleted
        ]
        materialize(
            source_scene["scene_dir"],
            output_dir,
            corrupted,
            link_mode,
        )
        manifest = {
            **common_manifest(
                condition=condition,
                split=split,
                sample=sample,
                output_dir=output_dir,
                source_scene=source_scene,
            ),
            "seed": seed,
            "severity_percent": percent,
            "deleted_instance_ids": deleted_ids,
            "num_instances_deleted": len(deleted_ids),
            "num_instances_after": len(corrupted["instances"]),
            "realized_deletion_percent": (
                100.0 * len(deleted_ids) / len(ids)
            ),
            "target_retained": str(sample["gt_object_id"]) not in deleted,
            "nested_severity_order": (
                "The sample-specific 10% deletion set is a subset of 20%."
            ),
        }
        write_json(output_dir / "corruption_manifest.json", manifest)
        records.append(manifest)
    return records


def nearest_neighbor(
    source_scene: dict[str, Any], target_id: str
) -> tuple[str, float]:
    target = source_scene["centroids"][target_id]
    candidates = [
        (
            float(np.linalg.norm(centroid - target)),
            instance_id,
        )
        for instance_id, centroid in source_scene["centroids"].items()
        if instance_id != target_id
    ]
    if not candidates:
        raise ValueError(f"No neighbor exists for {target_id}")
    return min(candidates)


def build_merge(
    *,
    split: str,
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_root: Path,
    link_mode: str,
) -> dict[str, Any]:
    condition = "target_neighbor_merge"
    target_id = str(sample["gt_object_id"])
    distance, neighbor_id = nearest_neighbor(source_scene, target_id)
    target_vertices = source_scene["vertices"][target_id]
    neighbor_vertices = source_scene["vertices"][neighbor_id]
    merged = xyz_only_vertices(target_vertices, neighbor_vertices)
    corrupted = deepcopy(source_scene["catalog"])
    for entry in corrupted["instances"]:
        if str(entry["instance_id"]) == target_id:
            entry["num_points"] = int(len(merged))
    corrupted["instances"] = [
        entry
        for entry in corrupted["instances"]
        if str(entry["instance_id"]) != neighbor_id
    ]
    output_dir = (
        output_root / condition / split / "samples" / sample_slug(sample)
    )
    materialize(
        source_scene["scene_dir"],
        output_dir,
        corrupted,
        link_mode,
        {target_id: merged},
    )
    manifest = {
        **common_manifest(
            condition=condition,
            split=split,
            sample=sample,
            output_dir=output_dir,
            source_scene=source_scene,
        ),
        "target_instance_id_after": target_id,
        "neighbor_instance_id": neighbor_id,
        "neighbor_selection": "minimum centroid Euclidean distance",
        "centroid_distance_m": distance,
        "target_points_before": int(len(target_vertices)),
        "neighbor_points_before": int(len(neighbor_vertices)),
        "merged_points_after": int(len(merged)),
        "num_instances_after": len(corrupted["instances"]),
    }
    write_json(output_dir / "corruption_manifest.json", manifest)
    return manifest


def next_instance_ids(catalog: dict[str, Any]) -> tuple[str, str]:
    values = [
        int(str(entry["instance_id"]).rsplit("_", 1)[-1])
        for entry in catalog["instances"]
    ]
    start = max(values) + 1
    return f"instance_{start:04d}", f"instance_{start + 1:04d}"


def build_split(
    *,
    split: str,
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_root: Path,
    link_mode: str,
) -> dict[str, Any]:
    condition = "target_two_way_split"
    target_id = str(sample["gt_object_id"])
    target_entry = entry_map(source_scene["catalog"])[target_id]
    target_vertices = source_scene["vertices"][target_id]
    children = split_vertices_pca(target_vertices)
    child_ids = next_instance_ids(source_scene["catalog"])
    corrupted = deepcopy(source_scene["catalog"])
    target_position = next(
        index
        for index, entry in enumerate(corrupted["instances"])
        if str(entry["instance_id"]) == target_id
    )
    corrupted["instances"].pop(target_position)
    child_entries: list[dict[str, Any]] = []
    for child_id, child_vertices in zip(child_ids, children):
        entry = deepcopy(target_entry)
        entry["instance_id"] = child_id
        entry["num_points"] = int(len(child_vertices))
        entry["ply_path"] = f"instances/{child_id}.ply"
        child_entries.append(entry)
    corrupted["instances"][target_position:target_position] = child_entries
    output_dir = (
        output_root / condition / split / "samples" / sample_slug(sample)
    )
    materialize(
        source_scene["scene_dir"],
        output_dir,
        corrupted,
        link_mode,
        dict(zip(child_ids, children)),
    )
    manifest = {
        **common_manifest(
            condition=condition,
            split=split,
            sample=sample,
            output_dir=output_dir,
            source_scene=source_scene,
        ),
        "original_target_instance_id_removed": target_id,
        "target_descendant_instance_ids": list(child_ids),
        "split_method": "median split along the first PCA axis",
        "target_points_before": int(len(target_vertices)),
        "child_point_counts": [int(len(value)) for value in children],
        "num_instances_after": len(corrupted["instances"]),
    }
    write_json(output_dir / "corruption_manifest.json", manifest)
    return manifest


def build_partial_noisy(
    *,
    split: str,
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_root: Path,
    global_seed: int,
    link_mode: str,
) -> dict[str, Any]:
    condition = "partial_noisy_target_mask"
    target_id = str(sample["gt_object_id"])
    target_vertices = source_scene["vertices"][target_id]
    seed = stable_seed(
        global_seed,
        split,
        *sample["sample_key"],
        int(sample["index"]),
        condition,
    )
    corrupted_vertices, removed, realized = partial_noisy_vertices(
        target_vertices, np.random.default_rng(seed)
    )
    corrupted = deepcopy(source_scene["catalog"])
    for entry in corrupted["instances"]:
        if str(entry["instance_id"]) == target_id:
            entry["num_points"] = int(len(corrupted_vertices))
    output_dir = (
        output_root / condition / split / "samples" / sample_slug(sample)
    )
    materialize(
        source_scene["scene_dir"],
        output_dir,
        corrupted,
        link_mode,
        {target_id: corrupted_vertices},
    )
    manifest = {
        **common_manifest(
            condition=condition,
            split=split,
            sample=sample,
            output_dir=output_dir,
            source_scene=source_scene,
        ),
        "seed": seed,
        "target_instance_id_after": target_id,
        "partial_mask_method": (
            "remove the extreme 15% slab along a seeded random direction"
        ),
        "coordinate_noise": (
            "isotropic Gaussian sigma=0.01 m, norm clipped at 0.03 m"
        ),
        "target_points_before": int(len(target_vertices)),
        "target_points_removed": int(removed),
        "target_points_after": int(len(corrupted_vertices)),
        "realized_point_removal_percent": float(realized),
        "num_instances_after": len(corrupted["instances"]),
    }
    write_json(output_dir / "corruption_manifest.json", manifest)
    return manifest


def validate_samples(
    split: str,
    samples: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> None:
    seen: set[str] = set()
    for expected_index, sample in enumerate(samples):
        if int(sample["index"]) != expected_index:
            raise ValueError(
                f"{split} manifest index mismatch at {expected_index}"
            )
        if str(sample.get("split", split)) != split:
            raise ValueError(f"Unexpected split in sample {expected_index}")
        identifier = sample_uid(split, sample)
        if identifier in seen:
            raise ValueError(f"Duplicate sample ID: {identifier}")
        seen.add(identifier)
        scene_name = str(sample["scene_name"])
        target_id = str(sample["gt_object_id"])
        if target_id not in entry_map(cache[scene_name]["catalog"]):
            raise ValueError(
                f"{split} sample {expected_index} target {target_id} "
                f"is absent from {scene_name}"
            )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} already exists; use a new output root"
        )

    samples_by_split = {
        "train": read_json(args.train_manifest.resolve()),
        "test": read_json(args.test_manifest.resolve()),
    }
    all_samples = samples_by_split["train"] + samples_by_split["test"]
    cache = source_cache(source_root, all_samples)
    for split, samples in samples_by_split.items():
        validate_samples(split, samples, cache)

    output_root.mkdir(parents=True)
    records_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for split, samples in samples_by_split.items():
        for sample in samples:
            source_scene = cache[str(sample["scene_name"])]
            for record in build_deletions(
                split=split,
                sample=sample,
                source_scene=source_scene,
                output_root=output_root,
                global_seed=args.seed,
                link_mode=args.link_mode,
            ):
                records_by_condition[record["condition"]].append(record)
            merge = build_merge(
                split=split,
                sample=sample,
                source_scene=source_scene,
                output_root=output_root,
                link_mode=args.link_mode,
            )
            records_by_condition[merge["condition"]].append(merge)
            split_record = build_split(
                split=split,
                sample=sample,
                source_scene=source_scene,
                output_root=output_root,
                link_mode=args.link_mode,
            )
            records_by_condition[split_record["condition"]].append(
                split_record
            )
            partial = build_partial_noisy(
                split=split,
                sample=sample,
                source_scene=source_scene,
                output_root=output_root,
                global_seed=args.seed,
                link_mode=args.link_mode,
            )
            records_by_condition[partial["condition"]].append(partial)

    for condition, records in records_by_condition.items():
        write_json(
            output_root / condition / "inventory_manifest.json",
            {
                "condition": condition,
                "num_samples": len(records),
                "split_counts": {
                    split: sum(record["split"] == split for record in records)
                    for split in ("train", "test")
                },
                "samples": records,
            },
        )

    deletion_recall: dict[str, Any] = {}
    for condition in CONDITIONS[:2]:
        deletion_recall[condition] = {}
        for split in ("train", "test"):
            rows = [
                row
                for row in records_by_condition[condition]
                if row["split"] == split
            ]
            retained = sum(bool(row["target_retained"]) for row in rows)
            deletion_recall[condition][split] = {
                "num_samples": len(rows),
                "target_retained": retained,
                "target_missed": len(rows) - retained,
                "target_recall_percent": 100.0 * retained / len(rows),
            }

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "train_manifest": str(args.train_manifest.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "global_seed": args.seed,
        "link_mode": args.link_mode,
        "num_train_samples": len(samples_by_split["train"]),
        "num_test_samples": len(samples_by_split["test"]),
        "num_condition_settings": len(CONDITIONS),
        "num_sample_condition_directories": sum(
            len(records) for records in records_by_condition.values()
        ),
        "condition_counts": {
            condition: len(records)
            for condition, records in records_by_condition.items()
        },
        "random_deletion_target_recall": deletion_recall,
    }
    write_json(output_root / "generation_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
