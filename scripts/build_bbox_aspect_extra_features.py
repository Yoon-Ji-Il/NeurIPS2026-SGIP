#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_openins3d_shape.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_bbox_aspect_extras.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

ASPECT_COLUMN = "object_shape_bbox_aspect_max_over_min"

INTERACTION_COLUMNS = (
    "object_shape_root_align_x_bbox_aspect",
    "object_shape_torso_align_x_bbox_aspect",
    "object_shape_root_torso_align_x_bbox_aspect",
    "object_shape_face_align_x_bbox_aspect",
    "object_shape_face_line_exp_x_bbox_aspect",
)

SUPPORT_COLUMNS = (
    "scene32768_bbox_bottom_slab_r010_fraction",
    "scene32768_bbox_below_bottom_r010_fraction",
    "scene32768_bbox_bottom_slab_r010_density",
    "scene32768_bbox_bottom_support_ratio",
)

VERTICAL_HISTOGRAM_COLUMNS = (
    "object_shape_vertical_lower_third_fraction",
    "object_shape_vertical_middle_third_fraction",
    "object_shape_vertical_upper_third_fraction",
    "object_shape_vertical_bottom_quarter_fraction",
    "object_shape_vertical_lower_minus_upper_fraction",
)

OUTPUT_COLUMNS = INTERACTION_COLUMNS + SUPPORT_COLUMNS + VERTICAL_HISTOGRAM_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair bbox-aspect extra features. These features use only raw observed-motion "
            "metrics from scene graphs, static scene_points_32768, static bbox geometry, and "
            "OpenIns3D object point clouds."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--scene-points-name", type=str, default="scene_points_32768.pth")
    parser.add_argument("--bottom-margin", type=float, default=0.10)
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


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def vector3(values: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def vector3_tensor(values: Any, *, name: str) -> torch.Tensor:
    x, y, z = vector3(values, name=name)
    return torch.tensor([x, y, z], dtype=torch.float32)


def load_points(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise ValueError(f"No vertices in {path}")
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N,3] point cloud in {path}, got {points.shape}")
    return points


def build_interaction_features(row: dict[str, str], metrics: dict[str, float]) -> dict[str, float]:
    aspect = parse_float(row.get(ASPECT_COLUMN))
    root = parse_float(metrics.get("root_alignment"))
    torso = parse_float(metrics.get("torso_alignment"))
    root_torso = parse_float(metrics.get("root_torso_alignment"))
    face = parse_float(metrics.get("face_normal_alignment"))
    face_line = max(parse_float(metrics.get("face_normal_line_distance_xz")), 0.0)
    return {
        "object_shape_root_align_x_bbox_aspect": root * aspect,
        "object_shape_torso_align_x_bbox_aspect": torso * aspect,
        "object_shape_root_torso_align_x_bbox_aspect": root_torso * aspect,
        "object_shape_face_align_x_bbox_aspect": face * aspect,
        "object_shape_face_line_exp_x_bbox_aspect": math.exp(-face_line) * aspect,
    }


def build_support_features(
    scene_points: torch.Tensor,
    obj: dict[str, Any],
    *,
    bottom_margin: float,
) -> dict[str, float]:
    bbox_min = vector3_tensor(obj.get("bbox_min"), name="bbox_min")
    bbox_max = vector3_tensor(obj.get("bbox_max"), name="bbox_max")
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    extent = (high - low).clamp_min(0.0)
    xz_area = max(float((extent[0] * extent[2]).item()), 1.0e-8)
    total_points = max(int(scene_points.shape[0]), 1)

    inside_xz = (
        (scene_points[:, 0] >= low[0])
        & (scene_points[:, 0] <= high[0])
        & (scene_points[:, 2] >= low[2])
        & (scene_points[:, 2] <= high[2])
    )
    y = scene_points[:, 1]
    bottom = float(low[1].item())
    margin = max(float(bottom_margin), 0.0)
    bottom_slab = inside_xz & (y >= bottom - margin) & (y <= bottom + margin)
    below_bottom = inside_xz & (y >= bottom - margin) & (y <= bottom)

    bottom_count = int(bottom_slab.sum().item())
    below_count = int(below_bottom.sum().item())
    xz_count = int(inside_xz.sum().item())
    return {
        "scene32768_bbox_bottom_slab_r010_fraction": bottom_count / total_points,
        "scene32768_bbox_below_bottom_r010_fraction": below_count / total_points,
        "scene32768_bbox_bottom_slab_r010_density": math.log1p(bottom_count / xz_area),
        "scene32768_bbox_bottom_support_ratio": below_count / max(xz_count, 1),
    }


def build_vertical_histogram_features(points: np.ndarray, obj: dict[str, Any]) -> dict[str, float]:
    bbox_min = vector3(obj.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(obj.get("bbox_max"), name="bbox_max")
    low_y = min(bbox_min[1], bbox_max[1])
    high_y = max(bbox_min[1], bbox_max[1])
    height = max(high_y - low_y, 1.0e-8)
    normalized_y = np.clip((points[:, 1] - low_y) / height, 0.0, 1.0)
    lower = float(np.mean(normalized_y < (1.0 / 3.0)))
    middle = float(np.mean((normalized_y >= (1.0 / 3.0)) & (normalized_y < (2.0 / 3.0))))
    upper = float(np.mean(normalized_y >= (2.0 / 3.0)))
    bottom_quarter = float(np.mean(normalized_y < 0.25))
    return {
        "object_shape_vertical_lower_third_fraction": lower,
        "object_shape_vertical_middle_third_fraction": middle,
        "object_shape_vertical_upper_third_fraction": upper,
        "object_shape_vertical_bottom_quarter_fraction": bottom_quarter,
        "object_shape_vertical_lower_minus_upper_fraction": lower - upper,
    }


def load_scene_graphs(root: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        path = root / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scene graph file: {path}")
        samples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        output[split] = samples
    return output


def build_feature_lookup(
    *,
    scene_graph_root: Path,
    slice_root: Path,
    scene_points_name: str,
    bottom_margin: float,
) -> dict[tuple[str, str, str], dict[str, float]]:
    scene_graphs = load_scene_graphs(scene_graph_root)
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    ply_cache: dict[Path, np.ndarray] = {}
    duplicate_mismatches: list[dict[str, Any]] = []

    for split, samples in scene_graphs.items():
        scene_points_path = slice_root / split / scene_points_name
        scene_points_tensor = torch.load(scene_points_path, map_location="cpu", weights_only=False).float()
        if scene_points_tensor.ndim != 3 or scene_points_tensor.shape[-1] != 3:
            raise TypeError(
                f"Expected [N,P,3] scene points in {scene_points_path}, got {tuple(scene_points_tensor.shape)}"
            )
        if len(samples) != int(scene_points_tensor.shape[0]):
            raise ValueError(
                f"{split}: scene graph sample count {len(samples)} does not match "
                f"{scene_points_path} shape {tuple(scene_points_tensor.shape)}"
            )

        for sample_index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", ""))
            scene_points = scene_points_tensor[sample_index]
            for obj in sample.get("objects", []):
                object_id = str(obj.get("object_id", ""))
                source_ply = str(obj.get("source_ply_path", "")).strip()
                if not sample_id or not object_id or not source_ply:
                    continue
                ply_path = Path(source_ply)
                if ply_path not in ply_cache:
                    ply_cache[ply_path] = load_points(ply_path)
                metrics = {key: parse_float(value) for key, value in (obj.get("metrics") or {}).items()}
                key = (split, sample_id, object_id)
                features = {}
                # The interaction terms need the row aspect value, so this placeholder is completed later.
                features.update({column: 0.0 for column in INTERACTION_COLUMNS})
                features.update(
                    build_support_features(scene_points, obj, bottom_margin=bottom_margin)
                )
                features.update(build_vertical_histogram_features(ply_cache[ply_path], obj))
                for metric_name in (
                    "root_alignment",
                    "torso_alignment",
                    "root_torso_alignment",
                    "face_normal_alignment",
                    "face_normal_line_distance_xz",
                ):
                    features[f"__metric_{metric_name}"] = metrics.get(metric_name, 0.0)

                previous = lookup.get(key)
                if previous is not None:
                    comparable = [column for column in SUPPORT_COLUMNS + VERTICAL_HISTOGRAM_COLUMNS]
                    max_abs_diff = max(abs(previous[name] - features[name]) for name in comparable)
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
            "Duplicate sample ids produced different extra features: "
            f"{duplicate_mismatches[:5]}"
        )
    return lookup


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id", ASPECT_COLUMN}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    lookup = build_feature_lookup(
        scene_graph_root=args.scene_graph_root,
        slice_root=args.slice_root,
        scene_points_name=args.scene_points_name,
        bottom_margin=args.bottom_margin,
    )

    output_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    zero_features = {column: 0.0 for column in OUTPUT_COLUMNS}
    for row in rows:
        output = dict(row)
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows.append({"split": key[0], "sample_id": key[1], "object_id": key[2]})
            output.update(zero_features)
        else:
            metrics = {
                name.removeprefix("__metric_"): value
                for name, value in features.items()
                if name.startswith("__metric_")
            }
            output.update(build_interaction_features(row, metrics))
            for column in SUPPORT_COLUMNS + VERTICAL_HISTOGRAM_COLUMNS:
                output[column] = features[column]
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "num_rows": len(output_rows),
                "missing_rows": len(missing_rows),
                "feature_columns": list(OUTPUT_COLUMNS),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
