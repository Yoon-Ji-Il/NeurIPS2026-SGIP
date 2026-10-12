#!/usr/bin/env python3
"""Create reproducible OpenIns3D segmentation corruptions for rebuttal tests.

The source and destination roots may be the same. In that case, the original
scene directories (for example, ``bedroom0122``) are treated as immutable
sources, and condition directories are written next to them.

Only ``label_pointcloud_export.json`` and ``instances/*.ply`` are materialized
because those are the inputs consumed by the object-selector preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


DEFAULT_SCENE_ROOT = Path("data/OpenIns3D/output/results_demo_rebuttal")
DEFAULT_SAMPLE_MANIFEST = Path(
    "data/GIMO/SLICES_888s/test/"
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024_manifest.json"
)
DEFAULT_SEED = 20260725


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument(
        "--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only condition directories targeted by this script.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def stable_seed(global_seed: int, *parts: object) -> int:
    text = "|".join([str(global_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def scene_json_path(scene_dir: Path) -> Path:
    return scene_dir / "label_pointcloud_export.json"


def load_scene(scene_dir: Path) -> dict[str, Any]:
    path = scene_json_path(scene_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not isinstance(payload.get("instances"), list):
        raise ValueError(f"Missing instance list: {path}")
    return payload


def instance_map(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["instance_id"]: entry for entry in scene["instances"]}


def source_ply_path(scene_dir: Path, entry: dict[str, Any]) -> Path:
    path = scene_dir / entry["ply_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_vertices(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    vertices = np.array(ply["vertex"].data, copy=True)
    required = {"x", "y", "z"}
    names = set(vertices.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(f"PLY lacks xyz vertex fields: {path}")
    return vertices


def xyz(vertices: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float64, copy=False)


def xyz_only_vertices(*arrays: np.ndarray) -> np.ndarray:
    """Concatenate heterogeneous PLY vertex schemas using their common xyz."""
    coordinates = np.concatenate([xyz(vertices) for vertices in arrays], axis=0)
    merged = np.empty(
        len(coordinates),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )
    for axis_index, field in enumerate(["x", "y", "z"]):
        merged[field] = coordinates[:, axis_index].astype(np.float32, copy=False)
    return merged


def write_vertices(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(vertices, "vertex")
    PlyData([element], text=False, byte_order="<").write(path)


def prepare_output_dir(
    output_root: Path, directory_name: str, overwrite: bool
) -> Path:
    destination = output_root / directory_name
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists; use --overwrite to replace it"
            )
        if destination.parent.resolve() != output_root.resolve():
            raise ValueError(f"Refusing to replace unexpected path: {destination}")
        shutil.rmtree(destination)
    (destination / "instances").mkdir(parents=True)
    return destination


def copy_instance(
    source_scene_dir: Path,
    destination_scene_dir: Path,
    entry: dict[str, Any],
) -> None:
    source = source_ply_path(source_scene_dir, entry)
    destination = destination_scene_dir / entry["ply_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def materialize_scene(
    source_scene_dir: Path,
    destination_scene_dir: Path,
    scene: dict[str, Any],
    replacement_vertices: dict[str, np.ndarray] | None = None,
) -> None:
    replacements = replacement_vertices or {}
    for entry in scene["instances"]:
        instance_id = entry["instance_id"]
        destination = destination_scene_dir / entry["ply_path"]
        if instance_id in replacements:
            write_vertices(destination, replacements[instance_id])
        else:
            copy_instance(source_scene_dir, destination_scene_dir, entry)
    write_json(scene_json_path(destination_scene_dir), scene)


def write_condition_manifest(
    destination_scene_dir: Path, payload: dict[str, Any]
) -> None:
    write_json(destination_scene_dir / "corruption_manifest.json", payload)


def make_clean(
    source_scene_dir: Path,
    output_root: Path,
    scene_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    scene = load_scene(source_scene_dir)
    directory_name = f"{scene_name}_clean"
    destination = prepare_output_dir(output_root, directory_name, overwrite)
    materialize_scene(source_scene_dir, destination, scene)
    manifest = {
        "condition": "clean",
        "source_scene": scene_name,
        "output_directory": directory_name,
        "num_instances_before": len(scene["instances"]),
        "num_instances_after": len(scene["instances"]),
    }
    write_condition_manifest(destination, manifest)
    return manifest


def deletion_count(num_instances: int, fraction: float) -> int:
    # Round half up, while ensuring that at least one candidate survives.
    count = int(math.floor(num_instances * fraction + 0.5))
    return min(count, max(0, num_instances - 1))


def random_instance_order(
    scene: dict[str, Any], scene_name: str, seed: int
) -> list[str]:
    instance_ids = [entry["instance_id"] for entry in scene["instances"]]
    rng = random.Random(stable_seed(seed, scene_name, "random_instance_deletion"))
    rng.shuffle(instance_ids)
    return instance_ids


def make_random_deletion(
    source_scene_dir: Path,
    output_root: Path,
    scene_name: str,
    fraction: float,
    deletion_order: list[str],
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    scene = load_scene(source_scene_dir)
    percent = int(round(fraction * 100))
    directory_name = f"{scene_name}_random_instance_deletion_{percent}%"
    destination = prepare_output_dir(output_root, directory_name, overwrite)

    count = deletion_count(len(scene["instances"]), fraction)
    deleted_ids = deletion_order[:count]
    deleted_set = set(deleted_ids)
    corrupted = deepcopy(scene)
    corrupted["instances"] = [
        entry
        for entry in corrupted["instances"]
        if entry["instance_id"] not in deleted_set
    ]
    materialize_scene(source_scene_dir, destination, corrupted)

    manifest = {
        "condition": "random_instance_deletion",
        "severity_percent": percent,
        "seed": seed,
        "source_scene": scene_name,
        "output_directory": directory_name,
        "num_instances_before": len(scene["instances"]),
        "num_instances_after": len(corrupted["instances"]),
        "num_instances_deleted": len(deleted_ids),
        "realized_deletion_percent": (
            100.0 * len(deleted_ids) / len(scene["instances"])
        ),
        "deleted_instance_ids": deleted_ids,
        "nested_severity_order": (
            "The 10% deletion set is a subset of the 20% deletion set."
        ),
    }
    write_condition_manifest(destination, manifest)
    return manifest


def sample_directory_name(
    scene_name: str, sample_index: int, condition_suffix: str
) -> str:
    return f"{scene_name}_sample{sample_index:03d}_{condition_suffix}"


def sample_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": int(sample["index"]),
        "sample_key": sample["sample_key"],
        "scene_name": sample["scene_name"],
        "sequence_name": sample["sequence_name"],
        "start_frame": sample["start_frame"],
        "end_frame": sample["end_frame"],
        "gt_target_instance_id": sample["gt_object_id"],
        "clean_selected_instance_id": sample.get("selected_object_id", ""),
    }


def nearest_centroid_neighbor(
    source_scene_dir: Path,
    scene: dict[str, Any],
    target_id: str,
) -> tuple[str, float, dict[str, np.ndarray]]:
    entries = instance_map(scene)
    vertices_by_id: dict[str, np.ndarray] = {}
    centroids: dict[str, np.ndarray] = {}
    for instance_id, entry in entries.items():
        vertices = read_vertices(source_ply_path(source_scene_dir, entry))
        vertices_by_id[instance_id] = vertices
        centroids[instance_id] = xyz(vertices).mean(axis=0)

    target_centroid = centroids[target_id]
    candidates = []
    for instance_id, centroid in centroids.items():
        if instance_id == target_id:
            continue
        distance = float(np.linalg.norm(centroid - target_centroid))
        candidates.append((distance, instance_id))
    if not candidates:
        raise ValueError(f"No merge neighbor for {target_id} in {source_scene_dir}")
    distance, neighbor_id = min(candidates)
    return neighbor_id, distance, vertices_by_id


def make_target_neighbor_merge(
    source_scene_dir: Path,
    output_root: Path,
    sample: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    scene_name = sample["scene_name"]
    sample_index = int(sample["index"])
    target_id = sample["gt_object_id"]
    scene = load_scene(source_scene_dir)
    entries = instance_map(scene)
    if target_id not in entries:
        raise KeyError(f"{target_id} is absent from clean scene {scene_name}")

    neighbor_id, distance, vertices_by_id = nearest_centroid_neighbor(
        source_scene_dir, scene, target_id
    )
    merged_vertices = xyz_only_vertices(
        vertices_by_id[target_id], vertices_by_id[neighbor_id]
    )

    corrupted = deepcopy(scene)
    for entry in corrupted["instances"]:
        if entry["instance_id"] == target_id:
            entry["num_points"] = int(len(merged_vertices))
    corrupted["instances"] = [
        entry
        for entry in corrupted["instances"]
        if entry["instance_id"] != neighbor_id
    ]

    directory_name = sample_directory_name(
        scene_name, sample_index, "target_neighbor_merge"
    )
    destination = prepare_output_dir(output_root, directory_name, overwrite)
    materialize_scene(
        source_scene_dir,
        destination,
        corrupted,
        replacement_vertices={target_id: merged_vertices},
    )
    manifest = {
        "condition": "target_neighbor_merge",
        "source_scene": scene_name,
        "output_directory": directory_name,
        **sample_metadata(sample),
        "target_instance_id_after": target_id,
        "neighbor_instance_id": neighbor_id,
        "neighbor_selection": "minimum centroid Euclidean distance",
        "centroid_distance_m": distance,
        "semantic_label_policy": "retain clean target label",
        "target_points_before": int(len(vertices_by_id[target_id])),
        "neighbor_points_before": int(len(vertices_by_id[neighbor_id])),
        "merged_points_after": int(len(merged_vertices)),
        "num_instances_before": len(scene["instances"]),
        "num_instances_after": len(corrupted["instances"]),
    }
    write_condition_manifest(destination, manifest)
    return manifest


def next_instance_ids(scene: dict[str, Any], count: int) -> list[str]:
    pattern = re.compile(r"instance_(\d+)$")
    numeric_ids = []
    for entry in scene["instances"]:
        match = pattern.fullmatch(entry["instance_id"])
        if not match:
            raise ValueError(f"Unexpected instance ID: {entry['instance_id']}")
        numeric_ids.append(int(match.group(1)))
    start = max(numeric_ids, default=0) + 1
    return [f"instance_{value:04d}" for value in range(start, start + count)]


def split_vertices_pca(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = xyz(vertices)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection = centered @ principal_axis
    order = np.argsort(projection, kind="stable")
    split_at = len(order) // 2
    if split_at == 0 or split_at == len(order):
        raise ValueError("Target has too few points for a two-way split")
    return vertices[order[:split_at]].copy(), vertices[order[split_at:]].copy()


def make_target_two_way_split(
    source_scene_dir: Path,
    output_root: Path,
    sample: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    scene_name = sample["scene_name"]
    sample_index = int(sample["index"])
    target_id = sample["gt_object_id"]
    scene = load_scene(source_scene_dir)
    entries = instance_map(scene)
    if target_id not in entries:
        raise KeyError(f"{target_id} is absent from clean scene {scene_name}")

    target_entry = entries[target_id]
    target_vertices = read_vertices(source_ply_path(source_scene_dir, target_entry))
    child_a, child_b = split_vertices_pca(target_vertices)
    child_ids = next_instance_ids(scene, 2)

    corrupted = deepcopy(scene)
    target_position = next(
        index
        for index, entry in enumerate(corrupted["instances"])
        if entry["instance_id"] == target_id
    )
    corrupted["instances"].pop(target_position)
    child_entries = []
    for child_id, child_vertices in zip(child_ids, [child_a, child_b]):
        child_entry = deepcopy(target_entry)
        child_entry["instance_id"] = child_id
        child_entry["num_points"] = int(len(child_vertices))
        child_entry["ply_path"] = f"instances/{child_id}.ply"
        child_entries.append(child_entry)
    corrupted["instances"][target_position:target_position] = child_entries

    directory_name = sample_directory_name(
        scene_name, sample_index, "target_two_way_split"
    )
    destination = prepare_output_dir(output_root, directory_name, overwrite)
    materialize_scene(
        source_scene_dir,
        destination,
        corrupted,
        replacement_vertices={
            child_ids[0]: child_a,
            child_ids[1]: child_b,
        },
    )
    manifest = {
        "condition": "target_two_way_split",
        "source_scene": scene_name,
        "output_directory": directory_name,
        **sample_metadata(sample),
        "split_method": "median split along the first PCA axis",
        "original_target_instance_id_removed": target_id,
        "target_descendant_instance_ids": child_ids,
        "semantic_label_policy": "copy clean target label to both children",
        "target_points_before": int(len(target_vertices)),
        "child_point_counts": [int(len(child_a)), int(len(child_b))],
        "num_instances_before": len(scene["instances"]),
        "num_instances_after": len(corrupted["instances"]),
    }
    write_condition_manifest(destination, manifest)
    return manifest


def partial_noisy_vertices(
    vertices: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int, float]:
    coordinates = xyz(vertices)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    projection = coordinates @ direction
    order = np.argsort(projection, kind="stable")

    remove_count = int(math.floor(len(vertices) * 0.15 + 0.5))
    remove_count = min(max(1, remove_count), len(vertices) - 1)
    kept = vertices[order[:-remove_count]].copy()

    noise = rng.normal(loc=0.0, scale=0.01, size=(len(kept), 3))
    norms = np.linalg.norm(noise, axis=1)
    over_limit = norms > 0.03
    noise[over_limit] *= (0.03 / norms[over_limit])[:, None]
    for axis_index, field in enumerate(["x", "y", "z"]):
        kept[field] = kept[field] + noise[:, axis_index].astype(
            kept[field].dtype, copy=False
        )
    realized = 100.0 * remove_count / len(vertices)
    return kept, remove_count, realized


def make_partial_noisy_target(
    source_scene_dir: Path,
    output_root: Path,
    sample: dict[str, Any],
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    scene_name = sample["scene_name"]
    sample_index = int(sample["index"])
    target_id = sample["gt_object_id"]
    scene = load_scene(source_scene_dir)
    entries = instance_map(scene)
    if target_id not in entries:
        raise KeyError(f"{target_id} is absent from clean scene {scene_name}")

    target_entry = entries[target_id]
    target_vertices = read_vertices(source_ply_path(source_scene_dir, target_entry))
    sample_seed = stable_seed(seed, scene_name, sample_index, "partial_noisy")
    rng = np.random.default_rng(sample_seed)
    corrupted_vertices, removed_count, realized = partial_noisy_vertices(
        target_vertices, rng
    )

    corrupted = deepcopy(scene)
    for entry in corrupted["instances"]:
        if entry["instance_id"] == target_id:
            entry["num_points"] = int(len(corrupted_vertices))

    directory_name = sample_directory_name(
        scene_name, sample_index, "partial_noisy_target_mask"
    )
    destination = prepare_output_dir(output_root, directory_name, overwrite)
    materialize_scene(
        source_scene_dir,
        destination,
        corrupted,
        replacement_vertices={target_id: corrupted_vertices},
    )
    manifest = {
        "condition": "partial_noisy_target_mask",
        "source_scene": scene_name,
        "output_directory": directory_name,
        **sample_metadata(sample),
        "seed": int(sample_seed),
        "target_instance_id_after": target_id,
        "partial_mask_method": (
            "remove the extreme 15% slab along a seeded random direction"
        ),
        "coordinate_noise": "isotropic Gaussian sigma=0.01 m, norm clipped at 0.03 m",
        "target_points_before": int(len(target_vertices)),
        "target_points_removed": int(removed_count),
        "realized_point_removal_percent": realized,
        "target_points_after": int(len(corrupted_vertices)),
        "num_instances_before": len(scene["instances"]),
        "num_instances_after": len(corrupted["instances"]),
    }
    write_condition_manifest(destination, manifest)
    return manifest


def validate_inputs(
    source_root: Path, samples: list[dict[str, Any]]
) -> list[str]:
    scene_names = sorted({sample["scene_name"] for sample in samples})
    errors = []
    for scene_name in scene_names:
        scene_dir = source_root / scene_name
        try:
            scene = load_scene(scene_dir)
            entries = instance_map(scene)
            for entry in scene["instances"]:
                source_ply_path(scene_dir, entry)
        except (FileNotFoundError, KeyError, ValueError) as error:
            errors.append(str(error))
            continue
        for sample in (
            row for row in samples if row["scene_name"] == scene_name
        ):
            if sample.get("gt_object_id") not in entries:
                errors.append(
                    f"Sample {sample.get('index')} target "
                    f"{sample.get('gt_object_id')} absent from {scene_name}"
                )
    if errors:
        raise RuntimeError("Input validation failed:\n" + "\n".join(errors))
    return scene_names


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    sample_manifest_path = args.sample_manifest.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = read_json(sample_manifest_path)
    if not isinstance(samples, list):
        raise TypeError(f"Expected a list in {sample_manifest_path}")
    scene_names = validate_inputs(source_root, samples)

    generated: list[dict[str, Any]] = []
    for scene_name in scene_names:
        source_scene_dir = source_root / scene_name
        scene = load_scene(source_scene_dir)
        deletion_order = random_instance_order(scene, scene_name, args.seed)
        generated.append(
            make_clean(
                source_scene_dir, output_root, scene_name, args.overwrite
            )
        )
        for fraction in [0.10, 0.20]:
            generated.append(
                make_random_deletion(
                    source_scene_dir=source_scene_dir,
                    output_root=output_root,
                    scene_name=scene_name,
                    fraction=fraction,
                    deletion_order=deletion_order,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            )

    for sample in sorted(samples, key=lambda row: int(row["index"])):
        source_scene_dir = source_root / sample["scene_name"]
        generated.append(
            make_target_neighbor_merge(
                source_scene_dir, output_root, sample, args.overwrite
            )
        )
        generated.append(
            make_target_two_way_split(
                source_scene_dir, output_root, sample, args.overwrite
            )
        )
        generated.append(
            make_partial_noisy_target(
                source_scene_dir,
                output_root,
                sample,
                args.seed,
                args.overwrite,
            )
        )

    random_target_recall: dict[str, dict[str, Any]] = {}
    for percent in [10, 20]:
        retained = 0
        missed_sample_indices = []
        for sample in samples:
            directory = (
                output_root
                / f"{sample['scene_name']}_random_instance_deletion_{percent}%"
            )
            scene = load_scene(directory)
            if sample["gt_object_id"] in instance_map(scene):
                retained += 1
            else:
                missed_sample_indices.append(int(sample["index"]))
        random_target_recall[f"{percent}%"] = {
            "num_samples": len(samples),
            "num_target_retained": retained,
            "num_target_missed": len(samples) - retained,
            "target_recall_percent": 100.0 * retained / len(samples),
            "target_missed_sample_indices": missed_sample_indices,
        }

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sample_manifest": str(sample_manifest_path),
        "global_seed": args.seed,
        "num_scenes": len(scene_names),
        "num_samples": len(samples),
        "num_generated_condition_directories": len(generated),
        "directory_counts": {
            "clean": len(scene_names),
            "random_instance_deletion_10%": len(scene_names),
            "random_instance_deletion_20%": len(scene_names),
            "target_neighbor_merge": len(samples),
            "target_two_way_split": len(samples),
            "partial_noisy_target_mask": len(samples),
        },
        "random_deletion_target_recall": random_target_recall,
        "generated_directories": [
            record["output_directory"] for record in generated
        ],
    }
    write_json(output_root / "rebuttal_generation_manifest.json", summary)
    print(json.dumps(summary["directory_counts"], indent=2))
    print(
        json.dumps(
            {"random_deletion_target_recall": random_target_recall},
            indent=2,
        )
    )
    print(f"Generated {len(generated)} condition directories in {output_root}")


if __name__ == "__main__":
    main()
