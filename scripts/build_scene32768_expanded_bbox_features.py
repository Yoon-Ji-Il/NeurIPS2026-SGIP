#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_local.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_expanded_bbox.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

EXPANSION_SCALES = (1.10, 1.25, 1.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair expanded-bbox occupancy features from scene_points_32768. "
            "This uses only static scene points and candidate bbox geometry."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--scene-points-name", type=str, default="scene_points_32768.pth")
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


def vector3(values: Any, *, name: str) -> torch.Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return torch.tensor([float(values[0]), float(values[1]), float(values[2])], dtype=torch.float32)


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


def expanded_fraction_features(scene_points: torch.Tensor, scene_object: dict[str, Any]) -> dict[str, float]:
    bbox_min = vector3(scene_object.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(scene_object.get("bbox_max"), name="bbox_max")
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    center = (low + high) * 0.5
    extent = (high - low).clamp_min(0.0)
    total_points = max(int(scene_points.shape[0]), 1)

    features: dict[str, float] = {}
    for scale in EXPANSION_SCALES:
        expanded_extent = extent * float(scale)
        expanded_low = center - expanded_extent * 0.5
        expanded_high = center + expanded_extent * 0.5
        inside = ((scene_points >= expanded_low) & (scene_points <= expanded_high)).all(dim=-1)
        key = f"scene32768_expanded_bbox_{str(scale).replace('.', 'p')}_fraction"
        features[key] = float(int(inside.sum().item()) / total_points)
    return features


def build_feature_lookup(
    *,
    scene_graph_root: Path,
    slice_root: Path,
    scene_points_name: str,
) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    duplicate_mismatches: list[dict[str, Any]] = []
    for split in ("train", "test"):
        samples = load_scene_samples(scene_graph_root, split)
        scene_points_path = slice_root / split / scene_points_name
        scene_points_tensor = torch.load(scene_points_path, map_location="cpu", weights_only=False).float()
        if scene_points_tensor.ndim != 3 or scene_points_tensor.shape[-1] != 3:
            raise TypeError(f"Expected [N,P,3] scene points in {scene_points_path}, got {tuple(scene_points_tensor.shape)}")
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
                if not sample_id or not object_id:
                    continue
                key = (split, sample_id, object_id)
                features = expanded_fraction_features(scene_points, obj)
                previous = lookup.get(key)
                if previous is not None:
                    max_abs_diff = max(abs(previous[name] - features[name]) for name in features)
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
            "Duplicate sample ids produced different expanded bbox features: "
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

    lookup = build_feature_lookup(
        scene_graph_root=args.scene_graph_root,
        slice_root=args.slice_root,
        scene_points_name=args.scene_points_name,
    )
    feature_columns = [
        f"scene32768_expanded_bbox_{str(scale).replace('.', 'p')}_fraction"
        for scale in EXPANSION_SCALES
    ]

    output_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows.append({"split": key[0], "sample_id": key[1], "object_id": key[2]})
            features = {column: 0.0 for column in feature_columns}
        output = dict(row)
        output.update(features)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in feature_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"features={','.join(feature_columns)}")
    print(f"missing_rows={len(missing_rows)}")


if __name__ == "__main__":
    main()
