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
    / "teacher_candidate_scores_scene32768_expanded_bbox.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_observed_trajectory.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

FEATURE_COLUMNS = (
    "obs_root_min_bbox_distance_xz",
    "obs_root_approach_delta_bbox_xz",
    "obs_hand_min_bbox_distance_3d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair observed-trajectory-to-bbox features. "
            "Uses only joints_input observed motion and static candidate bbox geometry."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
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


def load_scene_graphs(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "test": {}}
    for split in output:
        path = root / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scene graph jsonl: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("sample_id", "")).strip()
                if not sample_id:
                    continue
                previous = output[split].get(sample_id)
                if previous is not None:
                    # Candidate CSVs are unique-sample based. Keep the first scene graph entry for the known
                    # duplicated sequence row; sequence pth exports must still preserve the full 215 rows.
                    continue
                output[split][sample_id] = sample
    return output


def load_input_joints(slice_root: Path) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for split in ("train", "test"):
        path = slice_root / split / "joints_input.pth"
        if not path.exists():
            raise FileNotFoundError(f"Missing joints_input tensor: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise TypeError(f"Expected joints_input tensor [N,T,J,3] in {path}, got {type(tensor)!r}")
        output[split] = tensor.float()
    return output


def bbox_distance_3d(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    below = torch.clamp(low - points, min=0.0)
    above = torch.clamp(points - high, min=0.0)
    return torch.linalg.norm(below + above, dim=-1)


def bbox_distance_xz(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    x = points[..., 0]
    z = points[..., 2]
    dx = torch.clamp(low[0] - x, min=0.0) + torch.clamp(x - high[0], min=0.0)
    dz = torch.clamp(low[2] - z, min=0.0) + torch.clamp(z - high[2], min=0.0)
    return torch.sqrt(dx * dx + dz * dz)


def safe_indices(num_joints: int, requested: tuple[int, ...]) -> list[int]:
    return [index for index in requested if 0 <= index < num_joints]


def trajectory_features(scene_sample: dict[str, Any], object_id: str, joints_input: torch.Tensor) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in FEATURE_COLUMNS}

    bbox_min = vector3(scene_object.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(scene_object.get("bbox_max"), name="bbox_max")
    num_joints = int(joints_input.shape[1])
    root_indices = safe_indices(num_joints, (0,))
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))

    if root_indices:
        root_points = joints_input[:, root_indices, :].reshape(-1, 3)
        root_distances_xz = bbox_distance_xz(root_points, bbox_min, bbox_max)
        root_start_distance_xz = float(bbox_distance_xz(joints_input[:1, root_indices, :].reshape(-1, 3), bbox_min, bbox_max).min().item())
        root_end_distance_xz = float(bbox_distance_xz(joints_input[-1:, root_indices, :].reshape(-1, 3), bbox_min, bbox_max).min().item())
        root_min_distance_xz = float(root_distances_xz.min().item())
        root_approach_delta_xz = root_start_distance_xz - root_end_distance_xz
    else:
        root_min_distance_xz = 0.0
        root_approach_delta_xz = 0.0

    if hand_indices:
        hand_points = joints_input[:, hand_indices, :].reshape(-1, 3)
        hand_min_distance_3d = float(bbox_distance_3d(hand_points, bbox_min, bbox_max).min().item())
    else:
        hand_min_distance_3d = 0.0

    return {
        "obs_root_min_bbox_distance_xz": root_min_distance_xz,
        "obs_root_approach_delta_bbox_xz": root_approach_delta_xz,
        "obs_hand_min_bbox_distance_3d": hand_min_distance_3d,
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No candidate rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    scene_graphs = load_scene_graphs(args.scene_graph_root)
    input_joints = load_input_joints(args.slice_root)
    output_rows: list[dict[str, Any]] = []
    missing_rows = 0
    for row in rows:
        split = row.get("split", "")
        sample_id = row.get("sample_id", "")
        object_id = row.get("object_id", "")
        scene_sample = scene_graphs.get(split, {}).get(sample_id)
        if scene_sample is None or not object_id:
            missing_rows += 1
            features = {column: 0.0 for column in FEATURE_COLUMNS}
        else:
            sample_index = int(scene_sample["sample_index"])
            features = trajectory_features(scene_sample, object_id, input_joints[split][sample_index])
        enriched = dict(row)
        for column, value in features.items():
            enriched[column] = f"{float(value):.9g}"
        output_rows.append(enriched)

    fieldnames = list(rows[0].keys())
    for column in FEATURE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "num_rows": len(output_rows),
                "missing_rows": missing_rows,
                "feature_columns": list(FEATURE_COLUMNS),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
