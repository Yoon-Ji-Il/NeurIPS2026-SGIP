#!/usr/bin/env python3
"""Create test-only merge/split inventories targeting the clean teacher Top-1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("data/OpenIns3D/output/results_demo")
DEFAULT_TEST_MANIFEST = Path(
    "data/GIMO/SLICES_888s/test/"
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024_manifest.json"
)
DEFAULT_TEACHER_SUMMARY = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "selector_sample_summary.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/OpenIns3D/output/results_teacher_top1_test_segmentation_variants"
)
CONDITIONS = ("target_neighbor_merge", "target_two_way_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument(
        "--teacher-summary-csv", type=Path, default=DEFAULT_TEACHER_SUMMARY
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--link-mode", choices=("hardlink", "copy"), default="hardlink"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove only the exact output root before regenerating it.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                str(key).lstrip("\ufeff").strip(): (
                    str(value).strip() if value is not None else ""
                )
                for key, value in row.items()
                if key
            }
            for row in csv.DictReader(handle)
        ]


def sample_id(sample: dict[str, Any]) -> str:
    return "|".join(str(value) for value in sample["sample_key"])


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_points(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    vertices = getattr(loaded, "vertices", None)
    if vertices is None:
        raise ValueError(f"No vertices in {path}")
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Expected non-empty [N,3] points in {path}")
    return points


def write_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.points.PointCloud(np.asarray(points, dtype=np.float32))
    payload = trimesh.exchange.ply.export_ply(
        cloud, encoding="binary_little_endian"
    )
    path.write_bytes(payload)


def load_source_scenes(
    source_root: Path, samples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for scene_name in sorted({str(row["scene_name"]) for row in samples}):
        scene_dir = source_root / scene_name
        catalog_path = scene_dir / "label_pointcloud_export.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        unique_entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in catalog["instances"]:
            object_id = str(entry["instance_id"])
            if object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            unique_entries.append(entry)
        catalog["instances"] = unique_entries
        entry_by_id = {
            str(entry["instance_id"]): entry
            for entry in catalog["instances"]
        }
        points = {
            object_id: load_points(scene_dir / str(entry["ply_path"]))
            for object_id, entry in entry_by_id.items()
        }
        output[scene_name] = {
            "scene_dir": scene_dir,
            "catalog": catalog,
            "entry_by_id": entry_by_id,
            "points": points,
            "centroids": {
                object_id: values.mean(axis=0)
                for object_id, values in points.items()
            },
        }
    return output


def materialize(
    source_scene: dict[str, Any],
    output_dir: Path,
    catalog: dict[str, Any],
    link_mode: str,
    replacements: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True)
    for entry in catalog["instances"]:
        object_id = str(entry["instance_id"])
        # A legacy clean catalog contains two IDs pointing to the same PLY.
        # Give every materialized instance an unambiguous path while retaining
        # the original source lookup by instance ID.
        entry["ply_path"] = f"instances/{object_id}.ply"
        destination = output_dir / str(entry["ply_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if object_id in replacements:
            write_points(destination, replacements[object_id])
        else:
            source_entry = source_scene["entry_by_id"][object_id]
            source = source_scene["scene_dir"] / str(
                source_entry["ply_path"]
            )
            if link_mode == "hardlink":
                os.link(source, destination)
            else:
                shutil.copy2(source, destination)
    write_json(output_dir / "label_pointcloud_export.json", catalog)


def common_record(
    condition: str,
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "split": "test",
        "sample_index": int(sample["index"]),
        "sample_id": sample_id(sample),
        "sample_uid": f"test:{int(sample['index']):04d}:{sample_id(sample)}",
        "sample_key": sample["sample_key"],
        "scene_name": sample["scene_name"],
        "sequence_name": sample["sequence_name"],
        "start_frame": str(sample["start_frame"]),
        "end_frame": str(sample["end_frame"]),
        "output_directory": str(output_dir),
        "num_instances_before": len(source_scene["catalog"]["instances"]),
    }


def build_merge(
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_root: Path,
    link_mode: str,
) -> dict[str, Any]:
    condition = "target_neighbor_merge"
    target_id = str(sample["teacher_top1_object_id"])
    target_centroid = source_scene["centroids"][target_id]
    distance, neighbor_id = min(
        (
            float(np.linalg.norm(centroid - target_centroid)),
            object_id,
        )
        for object_id, centroid in source_scene["centroids"].items()
        if object_id != target_id
    )
    merged = np.concatenate(
        (
            source_scene["points"][target_id],
            source_scene["points"][neighbor_id],
        ),
        axis=0,
    )
    catalog = deepcopy(source_scene["catalog"])
    for entry in catalog["instances"]:
        if str(entry["instance_id"]) == target_id:
            entry["num_points"] = int(len(merged))
    catalog["instances"] = [
        entry
        for entry in catalog["instances"]
        if str(entry["instance_id"]) != neighbor_id
    ]
    output_dir = (
        output_root / condition / "test" / "samples" / sample_slug(sample)
    )
    materialize(
        source_scene,
        output_dir,
        catalog,
        link_mode,
        {target_id: merged},
    )
    return {
        **common_record(
            condition, sample, source_scene, output_dir
        ),
        "target_instance_id_after": target_id,
        "neighbor_instance_id": neighbor_id,
        "neighbor_selection": "minimum centroid Euclidean distance",
        "centroid_distance_m": distance,
        "target_points_before": int(
            len(source_scene["points"][target_id])
        ),
        "neighbor_points_before": int(
            len(source_scene["points"][neighbor_id])
        ),
        "merged_points_after": int(len(merged)),
        "num_instances_after": len(catalog["instances"]),
    }


def split_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    order = np.argsort(projection, kind="mergesort")
    midpoint = len(order) // 2
    if midpoint <= 0 or midpoint >= len(order):
        raise ValueError("Teacher Top-1 has too few points to split")
    return points[order[:midpoint]], points[order[midpoint:]]


def build_split(
    sample: dict[str, Any],
    source_scene: dict[str, Any],
    output_root: Path,
    link_mode: str,
) -> dict[str, Any]:
    condition = "target_two_way_split"
    target_id = str(sample["teacher_top1_object_id"])
    target_entry = source_scene["entry_by_id"][target_id]
    children = split_points(source_scene["points"][target_id])
    numeric_ids = [
        int(str(entry["instance_id"]).rsplit("_", 1)[-1])
        for entry in source_scene["catalog"]["instances"]
    ]
    first = max(numeric_ids) + 1
    child_ids = (
        f"instance_{first:04d}",
        f"instance_{first + 1:04d}",
    )
    catalog = deepcopy(source_scene["catalog"])
    target_position = next(
        index
        for index, entry in enumerate(catalog["instances"])
        if str(entry["instance_id"]) == target_id
    )
    catalog["instances"].pop(target_position)
    child_entries: list[dict[str, Any]] = []
    for child_id, child in zip(child_ids, children, strict=True):
        entry = deepcopy(target_entry)
        entry["instance_id"] = child_id
        entry["num_points"] = int(len(child))
        entry["ply_path"] = f"instances/{child_id}.ply"
        child_entries.append(entry)
    catalog["instances"][target_position:target_position] = child_entries
    output_dir = (
        output_root / condition / "test" / "samples" / sample_slug(sample)
    )
    materialize(
        source_scene,
        output_dir,
        catalog,
        link_mode,
        dict(zip(child_ids, children, strict=True)),
    )
    return {
        **common_record(
            condition, sample, source_scene, output_dir
        ),
        "original_target_instance_id_removed": target_id,
        "target_descendant_instance_ids": list(child_ids),
        "split_method": "median split along the first PCA axis",
        "target_points_before": int(
            len(source_scene["points"][target_id])
        ),
        "child_point_counts": [int(len(child)) for child in children],
        "num_instances_after": len(catalog["instances"]),
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    test_manifest = args.test_manifest.resolve()
    teacher_summary = args.teacher_summary_csv.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists; pass --overwrite to regenerate"
            )
        shutil.rmtree(output_root)

    clean_samples = json.loads(test_manifest.read_text(encoding="utf-8"))
    teacher_rows = [
        row for row in read_csv(teacher_summary) if row.get("split") == "test"
    ]
    teacher_lookup = {row["sample_id"]: row for row in teacher_rows}
    if len(teacher_lookup) != len(teacher_rows):
        raise ValueError("Duplicate test sample_id in teacher summary")

    samples: list[dict[str, Any]] = []
    for clean_sample in clean_samples:
        identifier = sample_id(clean_sample)
        teacher = teacher_lookup.get(identifier)
        if teacher is None:
            raise KeyError(f"Missing clean teacher Top-1 for {identifier}")
        sample = deepcopy(clean_sample)
        sample["original_gt_object_id"] = str(sample.get("gt_object_id", ""))
        sample["teacher_top1_object_id"] = teacher["selected_object_id"]
        sample["teacher_top1_object_name"] = teacher.get(
            "selected_object_name", ""
        )
        # The existing geometry builders target gt_object_id. Override only in
        # this in-memory record so their corruption target is the teacher Top-1.
        sample["gt_object_id"] = teacher["selected_object_id"]
        samples.append(sample)

    cache = load_source_scenes(source_root, samples)
    for sample in samples:
        target_id = str(sample["teacher_top1_object_id"])
        if target_id not in cache[str(sample["scene_name"])]["entry_by_id"]:
            raise ValueError(
                f"Teacher Top-1 {target_id} is absent for {sample_id(sample)}"
            )
    output_root.mkdir(parents=True)
    records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }

    for sample in samples:
        scene = cache[str(sample["scene_name"])]
        built = (
            build_merge(sample, scene, output_root, args.link_mode),
            build_split(sample, scene, output_root, args.link_mode),
        )
        for record in built:
            record.update(
                {
                    "corruption_target_source": "clean_qwen3_14b_teacher_top1",
                    "teacher_top1_instance_id": sample[
                        "teacher_top1_object_id"
                    ],
                    "teacher_top1_object_name": sample[
                        "teacher_top1_object_name"
                    ],
                    "original_gt_object_id": sample[
                        "original_gt_object_id"
                    ],
                    "train_inventory": "not_created_clean_train_is_unchanged",
                }
            )
            write_json(
                Path(record["output_directory"])
                / "corruption_manifest.json",
                record,
            )
            records[record["condition"]].append(record)

    for condition, rows in records.items():
        write_json(
            output_root / condition / "inventory_manifest.json",
            {
                "schema_version": 1,
                "condition": condition,
                "corruption_target_source": "clean_qwen3_14b_teacher_top1",
                "test_only": True,
                "num_samples": len(rows),
                "split_counts": {"train": 0, "test": len(rows)},
                "samples": rows,
            },
        )

    write_json(
        output_root / "summary.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "test_manifest": str(test_manifest),
            "teacher_summary_csv": str(teacher_summary),
            "output_root": str(output_root),
            "conditions": list(CONDITIONS),
            "num_test_samples": len(samples),
            "train": "unchanged and not materialized",
            "test": "all 37 sequences corrupted at the clean teacher Top-1",
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "conditions": {
                    condition: len(rows) for condition, rows in records.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
