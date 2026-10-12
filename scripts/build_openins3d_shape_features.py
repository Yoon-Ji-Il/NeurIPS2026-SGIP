#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_geometry.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_openins3d_shape.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"

OUTPUT_COLUMNS = (
    "object_shape_pca_linearity",
    "object_shape_pca_planarity",
    "object_shape_pca_scattering",
    "object_shape_pca_omnivariance",
    "object_shape_bbox_aspect_max_over_min",
    "object_shape_vertical_spread_ratio",
    "object_shape_bottom_half_fraction",
    "object_shape_log_point_density",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair static OpenIns3D object-shape descriptors. "
            "Uses only object point cloud geometry and bbox; no want, future, GT, teacher rank, "
            "category, or semantic label."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        row[clean_key] = str(value).strip() if value is not None else ""
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vector3(values: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_points(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise ValueError(f"No vertices in {path}")
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N,3] point cloud in {path}, got {points.shape}")
    return points


def shape_features(points: np.ndarray, obj: dict[str, Any]) -> dict[str, float]:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    l1, l2, l3 = [float(value) for value in eigenvalues]
    safe_l1 = max(l1, 1.0e-12)

    bbox_min = vector3(obj.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(obj.get("bbox_max"), name="bbox_max")
    extents = np.abs(np.asarray(bbox_max, dtype=np.float64) - np.asarray(bbox_min, dtype=np.float64))
    x_extent = float(extents[0])
    y_extent = float(extents[1])
    z_extent = float(extents[2])
    sorted_extents = sorted([x_extent, y_extent, z_extent])
    min_extent = max(sorted_extents[0], 1.0e-8)
    max_extent = max(sorted_extents[-1], 0.0)
    volume = max(x_extent * y_extent * z_extent, 0.0)
    xz_scale = math.sqrt(max(x_extent * z_extent, 0.0))

    low_y = min(bbox_min[1], bbox_max[1])
    high_y = max(bbox_min[1], bbox_max[1])
    center_y = 0.5 * (low_y + high_y)
    bottom_fraction = float(np.mean(points[:, 1] <= center_y))

    return {
        "object_shape_pca_linearity": (l1 - l2) / safe_l1,
        "object_shape_pca_planarity": (l2 - l3) / safe_l1,
        "object_shape_pca_scattering": l3 / safe_l1,
        "object_shape_pca_omnivariance": float(np.cbrt(max(l1 * l2 * l3, 0.0))),
        "object_shape_bbox_aspect_max_over_min": max_extent / min_extent,
        "object_shape_vertical_spread_ratio": y_extent / max(xz_scale, 1.0e-8),
        "object_shape_bottom_half_fraction": bottom_fraction,
        "object_shape_log_point_density": math.log1p(float(points.shape[0]) / max(volume, 1.0e-8)),
    }


def load_feature_lookup(scene_graph_root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    ply_cache: dict[Path, dict[str, float]] = {}
    duplicate_mismatches: list[dict[str, Any]] = []
    for split in ("train", "test"):
        path = scene_graph_root / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scene graph file: {path}")
        with path.open("r", encoding="utf-8") as file:
            for sample_index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("sample_id", ""))
                for obj in sample.get("objects", []):
                    object_id = str(obj.get("object_id", ""))
                    source_ply = str(obj.get("source_ply_path", "")).strip()
                    if not sample_id or not object_id or not source_ply:
                        continue
                    ply_path = Path(source_ply)
                    if ply_path not in ply_cache:
                        ply_cache[ply_path] = shape_features(load_points(ply_path), obj)
                    features = ply_cache[ply_path]
                    key = (split, sample_id, object_id)
                    previous = lookup.get(key)
                    if previous is not None:
                        max_abs_diff = max(abs(previous[name] - features[name]) for name in OUTPUT_COLUMNS)
                        if max_abs_diff > 1.0e-8:
                            duplicate_mismatches.append(
                                {
                                    "split": split,
                                    "sample_id": sample_id,
                                    "object_id": object_id,
                                    "sample_index": sample_index,
                                    "max_abs_diff": max_abs_diff,
                                }
                            )
                        continue
                    lookup[key] = features
    if duplicate_mismatches:
        raise RuntimeError(
            "Duplicate sample ids produced different shape features: "
            f"{duplicate_mismatches[:5]}"
        )
    return lookup


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    lookup = load_feature_lookup(args.scene_graph_root)
    output_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    zero_features = {column: 0.0 for column in OUTPUT_COLUMNS}
    for row in rows:
        output = dict(row)
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows.append({"split": key[0], "sample_id": key[1], "object_id": key[2]})
            features = zero_features
        output.update(features)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"features={','.join(OUTPUT_COLUMNS)}")
    print(f"missing_rows={len(missing_rows)}")
    print(f"num_rows={len(output_rows)}")


if __name__ == "__main__":
    main()
