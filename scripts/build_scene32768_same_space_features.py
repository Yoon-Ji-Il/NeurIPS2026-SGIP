#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "teacher_candidate_scores.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_local.csv"
)

SCENE32768_FEATURE_COLUMNS = (
    "scene32768_bbox_log_count",
    "scene32768_bbox_fraction",
    "scene32768_bbox_density",
    "scene32768_bbox_y_mean",
    "scene32768_bbox_y_std",
    "scene32768_centroid_r025_log_count",
    "scene32768_centroid_r050_log_count",
    "scene32768_centroid_r100_log_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append no-future local scene summaries from SIF3D scene_points_32768.pth "
            "to a same-space selector candidate CSV. This script never reads "
            "final_object_*_points_1024.pth caches."
        )
    )
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--scene-points-name",
        type=str,
        default="scene_points_32768.pth",
        help="Scene point cache under SLICES_888s/{train,test}/.",
    )
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


def load_scene_samples(scene_graph_root: Path, split: str) -> list[dict[str, Any]]:
    path = scene_graph_root / split / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene graph file: {path}")
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def vector3(values: Any, *, name: str) -> torch.Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return torch.tensor([float(values[0]), float(values[1]), float(values[2])], dtype=torch.float32)


def log_count(count: int) -> float:
    return float(math.log1p(max(int(count), 0)))


def feature_dict_for_object(scene_points: torch.Tensor, scene_object: dict[str, Any]) -> dict[str, float]:
    centroid = vector3(scene_object.get("centroid"), name="centroid")
    bbox_min = vector3(scene_object.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(scene_object.get("bbox_max"), name="bbox_max")
    bbox_low = torch.minimum(bbox_min, bbox_max)
    bbox_high = torch.maximum(bbox_min, bbox_max)
    extent = (bbox_high - bbox_low).clamp_min(0.0)
    volume = float(torch.prod(extent.clamp_min(1.0e-6)).item())

    inside = ((scene_points >= bbox_low) & (scene_points <= bbox_high)).all(dim=-1)
    inside_count = int(inside.sum().item())
    total_points = max(int(scene_points.shape[0]), 1)

    if inside_count > 0:
        local_points = scene_points[inside]
        y_values = local_points[:, 1]
        y_mean = float(y_values.mean().item())
        y_std = float(y_values.std(unbiased=False).item()) if inside_count > 1 else 0.0
    else:
        y_mean = 0.0
        y_std = 0.0

    distances = torch.linalg.norm(scene_points - centroid, dim=-1)
    radius_025_count = int((distances <= 0.25).sum().item())
    radius_050_count = int((distances <= 0.50).sum().item())
    radius_100_count = int((distances <= 1.00).sum().item())

    return {
        "scene32768_bbox_log_count": log_count(inside_count),
        "scene32768_bbox_fraction": float(inside_count / total_points),
        "scene32768_bbox_density": float(inside_count / max(volume, 1.0e-6)),
        "scene32768_bbox_y_mean": y_mean,
        "scene32768_bbox_y_std": y_std,
        "scene32768_centroid_r025_log_count": log_count(radius_025_count),
        "scene32768_centroid_r050_log_count": log_count(radius_050_count),
        "scene32768_centroid_r100_log_count": log_count(radius_100_count),
    }


def build_scene_feature_lookup(
    *,
    scene_graph_root: Path,
    slice_root: Path,
    scene_points_name: str,
) -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    meta: dict[str, Any] = {
        "scene_points_name": scene_points_name,
        "splits": {},
        "duplicate_candidate_keys": 0,
        "duplicate_mismatches": [],
    }

    for split in ("train", "test"):
        samples = load_scene_samples(scene_graph_root, split)
        scene_points_path = slice_root / split / scene_points_name
        if not scene_points_path.exists():
            raise FileNotFoundError(f"Missing scene point cache: {scene_points_path}")
        scene_points_tensor = torch.load(scene_points_path, map_location="cpu", weights_only=False).float()
        if scene_points_tensor.ndim != 3 or scene_points_tensor.shape[-1] != 3:
            raise TypeError(f"Expected [N,P,3] scene points in {scene_points_path}, got {tuple(scene_points_tensor.shape)}")
        if len(samples) != int(scene_points_tensor.shape[0]):
            raise ValueError(
                f"{split}: scene graph sample count {len(samples)} does not match "
                f"{scene_points_path} shape {tuple(scene_points_tensor.shape)}"
            )

        split_object_count = 0
        for sample_index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", ""))
            scene_points = scene_points_tensor[sample_index]
            for scene_object in sample.get("objects", []):
                object_id = str(scene_object.get("object_id", ""))
                if not sample_id or not object_id:
                    continue
                key = (split, sample_id, object_id)
                features = feature_dict_for_object(scene_points, scene_object)
                if key in lookup:
                    meta["duplicate_candidate_keys"] += 1
                    previous = lookup[key]
                    max_abs_diff = max(
                        abs(previous[column] - features[column])
                        for column in SCENE32768_FEATURE_COLUMNS
                    )
                    if max_abs_diff > 1.0e-6:
                        meta["duplicate_mismatches"].append(
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
                split_object_count += 1

        meta["splits"][split] = {
            "num_scene_graph_samples": len(samples),
            "scene_points_shape": list(scene_points_tensor.shape),
            "num_unique_candidate_keys": split_object_count,
        }

    if meta["duplicate_mismatches"]:
        raise RuntimeError(
            "Duplicate sample_id/object_id entries produced different scene32768 features: "
            f"{meta['duplicate_mismatches'][:5]}"
        )
    return lookup, meta


def main() -> None:
    args = parse_args()
    rows = read_csv(args.candidate_csv)
    if not rows:
        raise RuntimeError(f"No candidate rows in {args.candidate_csv}")
    required = {"split", "sample_id", "object_id"}
    missing_required = sorted(required - set(rows[0]))
    if missing_required:
        raise KeyError(f"Missing required columns in {args.candidate_csv}: {missing_required}")

    lookup, meta = build_scene_feature_lookup(
        scene_graph_root=args.scene_graph_root,
        slice_root=args.slice_root,
        scene_points_name=args.scene_points_name,
    )

    missing_rows: list[dict[str, str]] = []
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows.append(
                {
                    "split": key[0],
                    "sample_id": key[1],
                    "object_id": key[2],
                }
            )
            features = {column: 0.0 for column in SCENE32768_FEATURE_COLUMNS}
        output = dict(row)
        output.update(features)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in SCENE32768_FEATURE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)

    meta.update(
        {
            "candidate_csv": str(args.candidate_csv),
            "output_csv": str(args.output_csv),
            "feature_columns": list(SCENE32768_FEATURE_COLUMNS),
            "num_input_rows": len(rows),
            "num_missing_rows": len(missing_rows),
            "missing_rows": missing_rows[:20],
        }
    )
    meta_path = args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
        file.write("\n")

    print(f"wrote {args.output_csv}")
    print(f"features={','.join(SCENE32768_FEATURE_COLUMNS)}")
    print(f"missing_rows={len(missing_rows)} duplicate_candidate_keys={meta['duplicate_candidate_keys']}")
    print(f"meta={meta_path}")


if __name__ == "__main__":
    main()
