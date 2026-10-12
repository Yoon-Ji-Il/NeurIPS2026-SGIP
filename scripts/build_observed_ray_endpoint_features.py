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
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_expanded_bbox.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_observed_ray_endpoint.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

RAY_SOURCES = {
    "root": ("pelvis", "root_direction"),
    "torso": ("pelvis", "torso_direction"),
    "face": ("eye_mid", "face_normal_direction"),
}

FEATURE_COLUMNS = (
    "obs_last_root_bbox_distance_xz",
    "obs_root_ray_bbox_signed_along_xz",
    "obs_torso_ray_bbox_signed_along_xz",
    "obs_face_ray_bbox_signed_along_xz",
    "obs_root_ray_bbox_perp_distance_xz",
    "obs_root_ray_bbox_along_distance_xz",
    "obs_torso_ray_bbox_perp_distance_xz",
    "obs_torso_ray_bbox_along_distance_xz",
    "obs_face_ray_bbox_perp_distance_xz",
    "obs_face_ray_bbox_along_distance_xz",
    "obs_constvel_endpoint_bbox_distance_xz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair observed-motion ray/endpoint features. "
            "Uses only joints_input observed motion and static candidate bbox geometry."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument(
        "--constvel-horizon",
        type=float,
        default=10.0,
        help="Observed-frame velocity extrapolation horizon, matching output_seq_len by default.",
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


def vector3(values: Any, *, name: str) -> torch.Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return torch.tensor([float(values[0]), float(values[1]), float(values[2])], dtype=torch.float32)


def vector_xz(values: Any, *, name: str) -> torch.Tensor:
    vec = vector3(values, name=name)
    return torch.tensor([float(vec[0]), float(vec[2])], dtype=torch.float32)


def sample_vector(scene_sample: dict[str, Any], key: str) -> Any:
    if key in scene_sample:
        return scene_sample.get(key)
    human = scene_sample.get("human_final_frame", {})
    if isinstance(human, dict):
        return human.get(key)
    return None


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
                    # Candidate CSVs are unique-sample based. Keep the first row for the known duplicate.
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


def bbox_distance_xz(point: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> float:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    x = float(point[0])
    z = float(point[2])
    dx = max(float(low[0]) - x, 0.0) + max(x - float(high[0]), 0.0)
    dz = max(float(low[2]) - z, 0.0) + max(z - float(high[2]), 0.0)
    return math.sqrt(dx * dx + dz * dz)


def bbox_points_xz(bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    x0, z0 = float(low[0]), float(low[2])
    x1, z1 = float(high[0]), float(high[2])
    cx, cz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
    return torch.tensor(
        [
            [x0, z0],
            [x0, z1],
            [x1, z0],
            [x1, z1],
            [cx, z0],
            [cx, z1],
            [x0, cz],
            [x1, cz],
            [cx, cz],
        ],
        dtype=torch.float32,
    )


def normalize_xz(direction: torch.Tensor) -> tuple[torch.Tensor, bool]:
    vec = torch.tensor([float(direction[0]), float(direction[2])], dtype=torch.float32)
    norm = float(torch.linalg.norm(vec).item())
    if norm <= 1.0e-8:
        return torch.zeros(2, dtype=torch.float32), False
    return vec / norm, True


def ray_bbox_features_xz(
    *,
    origin_xz: torch.Tensor,
    direction_xz: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> tuple[float, float, float]:
    points = bbox_points_xz(bbox_min, bbox_max)
    rel = points - origin_xz[None, :]
    along_values = rel @ direction_xz
    perp_values = torch.abs(rel[:, 0] * direction_xz[1] - rel[:, 1] * direction_xz[0])

    min_along = float(along_values.min().item())
    max_along = float(along_values.max().item())
    if max_along < 0.0:
        signed_along = max_along
    elif min_along > 0.0:
        signed_along = min_along
    else:
        signed_along = 0.0

    ahead_mask = along_values >= 0.0
    if bool(ahead_mask.any().item()):
        perp_distance = float(perp_values[ahead_mask].min().item())
    else:
        pseudo_point = torch.tensor([float(origin_xz[0]), 0.0, float(origin_xz[1])], dtype=torch.float32)
        perp_distance = bbox_distance_xz(pseudo_point, bbox_min, bbox_max)
    along_distance = max(signed_along, 0.0)
    return signed_along, perp_distance, along_distance


def zero_features() -> dict[str, float]:
    return {column: 0.0 for column in FEATURE_COLUMNS}


def build_features(
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
    *,
    constvel_horizon: float,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None or joints_input.shape[1] <= 0:
        return zero_features()

    bbox_min = vector3(scene_object.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(scene_object.get("bbox_max"), name="bbox_max")
    root_traj = joints_input[:, 0, :]
    first_root = root_traj[0]
    last_root = root_traj[-1]
    last_root_distance = bbox_distance_xz(last_root, bbox_min, bbox_max)

    if root_traj.shape[0] > 1:
        velocity = (last_root - first_root) / float(root_traj.shape[0] - 1)
    else:
        velocity = torch.zeros_like(last_root)
    endpoint = last_root + velocity * float(constvel_horizon)
    endpoint_distance = bbox_distance_xz(endpoint, bbox_min, bbox_max)

    features = zero_features()
    features["obs_last_root_bbox_distance_xz"] = last_root_distance
    features["obs_constvel_endpoint_bbox_distance_xz"] = endpoint_distance

    for prefix, (origin_key, direction_key) in RAY_SOURCES.items():
        origin = vector_xz(sample_vector(scene_sample, origin_key), name=origin_key)
        direction3 = vector3(sample_vector(scene_sample, direction_key), name=direction_key)
        direction_xz, is_valid = normalize_xz(direction3)
        if not is_valid:
            continue
        signed_along, perp_distance, along_distance = ray_bbox_features_xz(
            origin_xz=origin,
            direction_xz=direction_xz,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
        features[f"obs_{prefix}_ray_bbox_signed_along_xz"] = signed_along
        features[f"obs_{prefix}_ray_bbox_perp_distance_xz"] = perp_distance
        features[f"obs_{prefix}_ray_bbox_along_distance_xz"] = along_distance

    return features


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
            features = zero_features()
        else:
            sample_index = int(scene_sample["sample_index"])
            features = build_features(
                scene_sample,
                object_id,
                input_joints[split][sample_index],
                constvel_horizon=args.constvel_horizon,
            )
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
                "constvel_horizon": args.constvel_horizon,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
