#!/usr/bin/env python3
"""Validate all source scenes and generated segmentation rebuttal variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from create_segmentation_rebuttal_variants import (
    DEFAULT_SAMPLE_MANIFEST,
    DEFAULT_SCENE_ROOT,
    partial_noisy_vertices,
    read_vertices,
    stable_seed,
    xyz,
)


CONDITIONS = {
    "clean",
    "random_instance_deletion",
    "target_neighbor_merge",
    "target_two_way_split",
    "partial_noisy_target_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument(
        "--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Truncated PLY header: {path}")
            line = raw_line.decode("ascii").strip()
            match = re.fullmatch(r"element vertex (\d+)", line)
            if match:
                count = int(match.group(1))
            if line == "end_header":
                break
    if "count" not in locals():
        raise ValueError(f"PLY vertex count is absent: {path}")
    return count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalog_by_id(
    scene_dir: Path,
    *,
    allow_unreferenced_ply: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    catalog_path = scene_dir / "label_pointcloud_export.json"
    catalog = read_json(catalog_path)
    instances = catalog.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"Missing instance list: {catalog_path}")

    by_id: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for entry in instances:
        instance_id = str(entry.get("instance_id", ""))
        relative_path = str(entry.get("ply_path", ""))
        if not instance_id or instance_id in by_id:
            raise ValueError(
                f"Missing or duplicate instance ID {instance_id!r}: {catalog_path}"
            )
        if not relative_path or relative_path in paths:
            raise ValueError(
                f"Missing or duplicate PLY path {relative_path!r}: {catalog_path}"
            )
        ply_path = scene_dir / relative_path
        if not ply_path.is_file():
            raise FileNotFoundError(ply_path)
        actual_count = ply_vertex_count(ply_path)
        if entry.get("num_points") != actual_count:
            raise ValueError(
                f"Point-count mismatch for {ply_path}: "
                f"JSON={entry.get('num_points')} PLY={actual_count}"
            )
        by_id[instance_id] = entry
        paths.add(relative_path)

    referenced = {str(scene_dir / path) for path in paths}
    materialized = {str(path) for path in (scene_dir / "instances").glob("*.ply")}
    missing = sorted(referenced - materialized)
    extra = sorted(materialized - referenced)
    if missing or (extra and not allow_unreferenced_ply):
        raise ValueError(
            f"PLY inventory mismatch in {scene_dir}: missing={missing}, extra={extra}"
        )
    return by_id, catalog


def assert_entry_metadata_equal(
    source_entry: dict[str, Any], output_entry: dict[str, Any]
) -> None:
    if source_entry != output_entry:
        raise ValueError(
            "Unchanged instance metadata differs: "
            f"source={source_entry}, output={output_entry}"
        )


def assert_file_equal(source: Path, output: Path) -> None:
    if file_sha256(source) != file_sha256(output):
        raise ValueError(f"Copied PLY differs: {source} != {output}")


def sorted_xyz(points: np.ndarray) -> np.ndarray:
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return points[order]


def validate_clean(
    source_dir: Path,
    output_dir: Path,
    source: dict[str, dict[str, Any]],
    output: dict[str, dict[str, Any]],
) -> None:
    if source.keys() != output.keys():
        raise ValueError(f"Clean IDs differ in {output_dir}")
    for instance_id, source_entry in source.items():
        output_entry = output[instance_id]
        assert_entry_metadata_equal(source_entry, output_entry)
        assert_file_equal(
            source_dir / str(source_entry["ply_path"]),
            output_dir / str(output_entry["ply_path"]),
        )


def validate_deletion(
    source_dir: Path,
    output_dir: Path,
    source: dict[str, dict[str, Any]],
    output: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    deleted = list(manifest["deleted_instance_ids"])
    if len(deleted) != len(set(deleted)):
        raise ValueError(f"Duplicate deleted ID in {output_dir}")
    expected_ids = set(source) - set(deleted)
    if set(output) != expected_ids:
        raise ValueError(f"Deletion IDs differ in {output_dir}")
    if manifest["num_instances_deleted"] != len(deleted):
        raise ValueError(f"Deletion count differs in {output_dir}")
    for instance_id, output_entry in output.items():
        source_entry = source[instance_id]
        assert_entry_metadata_equal(source_entry, output_entry)
        assert_file_equal(
            source_dir / str(source_entry["ply_path"]),
            output_dir / str(output_entry["ply_path"]),
        )


def validate_merge(
    source_dir: Path,
    output_dir: Path,
    source: dict[str, dict[str, Any]],
    output: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    target_id = str(manifest["gt_target_instance_id"])
    neighbor_id = str(manifest["neighbor_instance_id"])
    if set(output) != set(source) - {neighbor_id}:
        raise ValueError(f"Merge IDs differ in {output_dir}")

    for instance_id, output_entry in output.items():
        if instance_id == target_id:
            continue
        source_entry = source[instance_id]
        assert_entry_metadata_equal(source_entry, output_entry)
        assert_file_equal(
            source_dir / str(source_entry["ply_path"]),
            output_dir / str(output_entry["ply_path"]),
        )

    target_points = xyz(
        read_vertices(source_dir / str(source[target_id]["ply_path"]))
    )
    neighbor_points = xyz(
        read_vertices(source_dir / str(source[neighbor_id]["ply_path"]))
    )
    merged_points = xyz(
        read_vertices(output_dir / str(output[target_id]["ply_path"]))
    )
    expected = sorted_xyz(np.concatenate([target_points, neighbor_points]))
    if not np.array_equal(sorted_xyz(merged_points), expected):
        raise ValueError(f"Merged coordinates differ in {output_dir}")


def validate_split(
    source_dir: Path,
    output_dir: Path,
    source: dict[str, dict[str, Any]],
    output: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    target_id = str(manifest["gt_target_instance_id"])
    child_ids = [str(value) for value in manifest["target_descendant_instance_ids"]]
    expected_ids = (set(source) - {target_id}) | set(child_ids)
    if set(output) != expected_ids or len(child_ids) != 2:
        raise ValueError(f"Split IDs differ in {output_dir}")

    for instance_id, output_entry in output.items():
        if instance_id in child_ids:
            continue
        source_entry = source[instance_id]
        assert_entry_metadata_equal(source_entry, output_entry)
        assert_file_equal(
            source_dir / str(source_entry["ply_path"]),
            output_dir / str(output_entry["ply_path"]),
        )

    source_points = xyz(
        read_vertices(source_dir / str(source[target_id]["ply_path"]))
    )
    child_points = np.concatenate(
        [
            xyz(read_vertices(output_dir / str(output[child_id]["ply_path"])))
            for child_id in child_ids
        ]
    )
    if not np.array_equal(sorted_xyz(child_points), sorted_xyz(source_points)):
        raise ValueError(f"Split coordinates differ in {output_dir}")


def validate_partial_noisy(
    source_dir: Path,
    output_dir: Path,
    source: dict[str, dict[str, Any]],
    output: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    global_seed: int,
) -> None:
    target_id = str(manifest["gt_target_instance_id"])
    if set(output) != set(source):
        raise ValueError(f"Partial/noisy IDs differ in {output_dir}")

    for instance_id, output_entry in output.items():
        if instance_id == target_id:
            continue
        source_entry = source[instance_id]
        assert_entry_metadata_equal(source_entry, output_entry)
        assert_file_equal(
            source_dir / str(source_entry["ply_path"]),
            output_dir / str(output_entry["ply_path"]),
        )

    source_vertices = read_vertices(
        source_dir / str(source[target_id]["ply_path"])
    )
    sample_index = int(manifest["sample_index"])
    expected_seed = stable_seed(
        global_seed,
        manifest["source_scene"],
        sample_index,
        "partial_noisy",
    )
    if int(manifest["seed"]) != expected_seed:
        raise ValueError(f"Partial/noisy seed differs in {output_dir}")
    expected_vertices, removed, realized = partial_noisy_vertices(
        source_vertices, np.random.default_rng(expected_seed)
    )
    output_vertices = read_vertices(
        output_dir / str(output[target_id]["ply_path"])
    )
    if not np.array_equal(xyz(output_vertices), xyz(expected_vertices)):
        raise ValueError(f"Partial/noisy coordinates differ in {output_dir}")
    if manifest["target_points_removed"] != removed:
        raise ValueError(f"Partial/noisy removed count differs in {output_dir}")
    if not np.isclose(manifest["realized_point_removal_percent"], realized):
        raise ValueError(f"Partial/noisy percentage differs in {output_dir}")


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    samples = read_json(args.sample_manifest.resolve())
    scene_names = sorted({str(sample["scene_name"]) for sample in samples})
    summary = read_json(scene_root / "rebuttal_generation_manifest.json")
    generated_names = list(summary["generated_directories"])
    expected_directory_names = set(scene_names) | set(generated_names)
    actual_directory_names = {
        path.name for path in scene_root.iterdir() if path.is_dir()
    }
    if actual_directory_names != expected_directory_names:
        raise ValueError(
            "Top-level directory mismatch: "
            f"missing={sorted(expected_directory_names - actual_directory_names)}, "
            f"extra={sorted(actual_directory_names - expected_directory_names)}"
        )
    if len(generated_names) != 168 or len(set(generated_names)) != 168:
        raise ValueError("Global generation manifest must list 168 unique directories")

    source_catalogs = {
        scene_name: catalog_by_id(
            scene_root / scene_name, allow_unreferenced_ply=True
        )[0]
        for scene_name in scene_names
    }
    condition_counts = {condition: 0 for condition in CONDITIONS}
    deletion_sets: dict[str, dict[int, set[str]]] = {}
    deletion_totals: dict[int, dict[str, int]] = {
        10: {"before": 0, "deleted": 0},
        20: {"before": 0, "deleted": 0},
    }

    for directory_name in generated_names:
        output_dir = scene_root / directory_name
        manifest = read_json(output_dir / "corruption_manifest.json")
        condition = str(manifest["condition"])
        if condition not in condition_counts:
            raise ValueError(f"Unknown condition {condition} in {output_dir}")
        if manifest["output_directory"] != directory_name:
            raise ValueError(f"Output directory mismatch in {output_dir}")
        scene_name = str(manifest["source_scene"])
        source_dir = scene_root / scene_name
        source = source_catalogs[scene_name]
        output, _ = catalog_by_id(output_dir)
        if manifest["num_instances_before"] != len(source):
            raise ValueError(f"Before count differs in {output_dir}")
        if manifest["num_instances_after"] != len(output):
            raise ValueError(f"After count differs in {output_dir}")

        if condition == "clean":
            validate_clean(source_dir, output_dir, source, output)
        elif condition == "random_instance_deletion":
            validate_deletion(
                source_dir, output_dir, source, output, manifest
            )
            severity = int(manifest["severity_percent"])
            deleted = set(manifest["deleted_instance_ids"])
            deletion_sets.setdefault(scene_name, {})[severity] = deleted
            deletion_totals[severity]["before"] += len(source)
            deletion_totals[severity]["deleted"] += len(deleted)
        elif condition == "target_neighbor_merge":
            validate_merge(source_dir, output_dir, source, output, manifest)
        elif condition == "target_two_way_split":
            validate_split(source_dir, output_dir, source, output, manifest)
        else:
            validate_partial_noisy(
                source_dir,
                output_dir,
                source,
                output,
                manifest,
                int(summary["global_seed"]),
            )
        condition_counts[condition] += 1

    for scene_name, severity_sets in deletion_sets.items():
        if not severity_sets[10].issubset(severity_sets[20]):
            raise ValueError(f"Deletion severities are not nested for {scene_name}")

    expected_counts = {
        "clean": 19,
        "random_instance_deletion": 38,
        "target_neighbor_merge": 37,
        "target_two_way_split": 37,
        "partial_noisy_target_mask": 37,
    }
    if condition_counts != expected_counts:
        raise ValueError(
            f"Condition counts differ: {condition_counts} != {expected_counts}"
        )

    report = {
        "status": "PASS",
        "num_source_scenes": len(scene_names),
        "num_generated_directories": len(generated_names),
        "num_total_directories": len(actual_directory_names),
        "condition_counts": condition_counts,
        "deletion_totals": {
            f"{severity}%": {
                **totals,
                "realized_percent": (
                    100.0 * totals["deleted"] / totals["before"]
                ),
            }
            for severity, totals in deletion_totals.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
