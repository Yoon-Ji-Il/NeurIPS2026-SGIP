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
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline9_local_xz_residual_sweep_seed7"
    / "open_max060_anchor070"
    / "best_train_gt_export"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_baseline10_observed_hand_reach.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

FEATURE_COLUMNS = (
    "obs_hand_min_bbox_distance_3d",
    "obs_hand_last_bbox_distance_3d",
    "input_torso_object_vertical_coord",
    "obs_hand_approach_delta_bbox_3d",
    "obs_hand_bbox_y_gap_last",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair observed hand/body reach features. "
            "Uses only joints_input observed motion and static OpenIns3D bbox geometry."
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
        if clean_key:
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
            for line in file:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("sample_id", "")).strip()
                if not sample_id or sample_id in output[split]:
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


def safe_indices(num_joints: int, requested: tuple[int, ...]) -> list[int]:
    return [index for index in requested if 0 <= index < num_joints]


def bbox_distance(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    below = torch.clamp(low - points, min=0.0)
    above = torch.clamp(points - high, min=0.0)
    return torch.linalg.norm(below + above, dim=-1)


def bbox_y_gap(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low_y = torch.minimum(bbox_min, bbox_max)[1]
    high_y = torch.maximum(bbox_min, bbox_max)[1]
    y = points[..., 1]
    below = torch.clamp(low_y - y, min=0.0)
    above = torch.clamp(y - high_y, min=0.0)
    return below + above


def reach_features(scene_sample: dict[str, Any], object_id: str, joints_input: torch.Tensor) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in FEATURE_COLUMNS}

    bbox_min = vector3(scene_object.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(scene_object.get("bbox_max"), name="bbox_max")
    centroid = vector3(scene_object.get("centroid"), name="centroid")

    num_joints = int(joints_input.shape[1])
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    pelvis_index = 0 if num_joints > 0 else None
    if not hand_indices:
        hand_points = joints_input[:, [0], :]
    else:
        hand_points = joints_input[:, hand_indices, :]

    hand_all = hand_points.reshape(-1, 3)
    hand_first = hand_points[:1].reshape(-1, 3)
    hand_last = hand_points[-1:].reshape(-1, 3)
    first_distance = float(bbox_distance(hand_first, bbox_min, bbox_max).min().item())
    last_distance = float(bbox_distance(hand_last, bbox_min, bbox_max).min().item())
    min_distance = float(bbox_distance(hand_all, bbox_min, bbox_max).min().item())
    y_gap_last = float(bbox_y_gap(hand_last, bbox_min, bbox_max).min().item())

    if pelvis_index is None:
        vertical_coord = 0.0
    else:
        vertical_coord = float((centroid[1] - joints_input[-1, pelvis_index, 1]).item())

    return {
        "obs_hand_min_bbox_distance_3d": min_distance,
        "obs_hand_last_bbox_distance_3d": last_distance,
        "input_torso_object_vertical_coord": vertical_coord,
        "obs_hand_approach_delta_bbox_3d": first_distance - last_distance,
        "obs_hand_bbox_y_gap_last": y_gap_last,
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No candidate rows found in {args.input_csv}")
    missing = sorted({"split", "sample_id", "object_id"} - set(rows[0]))
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
            features = reach_features(scene_sample, object_id, input_joints[split][sample_index])
        enriched = dict(row)
        for column, value in features.items():
            if not math.isfinite(float(value)):
                value = 0.0
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
