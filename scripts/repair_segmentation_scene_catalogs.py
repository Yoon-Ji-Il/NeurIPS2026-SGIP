#!/usr/bin/env python3
"""Repair source scene catalogs before generating segmentation corruptions.

The repair is intentionally conservative:

* byte-for-byte duplicate instance-ID rows are collapsed;
* a duplicated PLY path is changed only when the affected instance's canonical
  ``instances/<instance_id>.ply`` file already exists; and
* ``num_points`` is synchronized with the referenced PLY vertex count.

Only source scene directories named by the sample manifest are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plyfile import PlyData


DEFAULT_SCENE_ROOT = Path("data/OpenIns3D/output/results_demo_rebuttal")
DEFAULT_SAMPLE_MANIFEST = Path(
    "data/GIMO/SLICES_888s/test/"
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument(
        "--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def vertex_count(path: Path) -> int:
    ply = PlyData.read(path)
    return int(ply["vertex"].count)


def repair_scene(scene_dir: Path, dry_run: bool) -> dict[str, Any]:
    catalog_path = scene_dir / "label_pointcloud_export.json"
    catalog = read_json(catalog_path)
    instances = catalog.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"Missing instance list: {catalog_path}")

    duplicate_ids_removed: list[str] = []
    deduplicated: list[dict[str, Any]] = []
    first_by_id: dict[str, dict[str, Any]] = {}
    for entry in instances:
        instance_id = str(entry.get("instance_id", ""))
        if not instance_id:
            raise ValueError(f"Missing instance_id in {catalog_path}: {entry}")
        previous = first_by_id.get(instance_id)
        if previous is None:
            first_by_id[instance_id] = entry
            deduplicated.append(entry)
            continue
        if entry != previous:
            raise ValueError(
                f"Conflicting duplicate instance ID {instance_id} in {catalog_path}"
            )
        duplicate_ids_removed.append(instance_id)

    repaired_paths: list[dict[str, str]] = []
    path_owner: dict[str, str] = {}
    for entry in deduplicated:
        instance_id = str(entry["instance_id"])
        relative_path = str(entry.get("ply_path", ""))
        if not relative_path:
            raise ValueError(
                f"Missing ply_path for {instance_id} in {catalog_path}"
            )
        previous_owner = path_owner.get(relative_path)
        if previous_owner is not None and previous_owner != instance_id:
            canonical_path = f"instances/{instance_id}.ply"
            canonical_file = scene_dir / canonical_path
            if not canonical_file.is_file():
                raise ValueError(
                    f"PLY path {relative_path} is shared by {previous_owner} and "
                    f"{instance_id}, but {canonical_file} does not exist"
                )
            entry["ply_path"] = canonical_path
            repaired_paths.append(
                {
                    "instance_id": instance_id,
                    "before": relative_path,
                    "after": canonical_path,
                }
            )
            relative_path = canonical_path
        path_owner[relative_path] = instance_id

    point_counts_updated: list[dict[str, int | str]] = []
    for entry in deduplicated:
        instance_id = str(entry["instance_id"])
        ply_path = scene_dir / str(entry["ply_path"])
        if not ply_path.is_file():
            raise FileNotFoundError(ply_path)
        actual_count = vertex_count(ply_path)
        recorded_count = entry.get("num_points")
        if recorded_count != actual_count:
            point_counts_updated.append(
                {
                    "instance_id": instance_id,
                    "before": recorded_count,
                    "after": actual_count,
                }
            )
            entry["num_points"] = actual_count

    changed = bool(
        duplicate_ids_removed or repaired_paths or point_counts_updated
    )
    catalog["instances"] = deduplicated
    if changed and not dry_run:
        write_json_atomic(catalog_path, catalog)

    return {
        "scene_name": scene_dir.name,
        "changed": changed,
        "duplicate_ids_removed": duplicate_ids_removed,
        "repaired_paths": repaired_paths,
        "point_counts_updated": point_counts_updated,
    }


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    samples = read_json(args.sample_manifest.resolve())
    if not isinstance(samples, list):
        raise TypeError("Sample manifest must contain a list")

    scene_names = sorted({str(sample["scene_name"]) for sample in samples})
    reports = [
        repair_scene(scene_root / scene_name, args.dry_run)
        for scene_name in scene_names
    ]
    summary = {
        "dry_run": args.dry_run,
        "num_scenes": len(reports),
        "num_scenes_changed": sum(report["changed"] for report in reports),
        "num_duplicate_ids_removed": sum(
            len(report["duplicate_ids_removed"]) for report in reports
        ),
        "num_paths_repaired": sum(
            len(report["repaired_paths"]) for report in reports
        ),
        "num_point_counts_updated": sum(
            len(report["point_counts_updated"]) for report in reports
        ),
        "reports": [report for report in reports if report["changed"]],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
