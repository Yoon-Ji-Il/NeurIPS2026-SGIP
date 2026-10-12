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
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "best_train_gt_export"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_only_teacher_proxy.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

ROOT_PROXY_COLUMNS = (
    "proxy_root_cv2_bbox_distance_xz",
    "proxy_root_cv5_bbox_distance_xz",
    "proxy_root_cv10_bbox_distance_xz",
    "proxy_root_cv10_approach_gain_xz",
    "proxy_root_cv10_object_alignment",
    "proxy_root_cv10_exp_score",
)
HAND_TTC_COLUMNS = (
    "proxy_hand_last_bbox_distance_3d",
    "proxy_hand_min_bbox_distance_3d",
    "proxy_hand_approach_delta_3d",
    "proxy_hand_distance_slope_3d",
    "proxy_hand_speed_toward_bbox",
    "proxy_hand_ttc_score",
    "proxy_hand_contact_dwell_r025",
)
FACE_DWELL_COLUMNS = (
    "proxy_face_ray_perp_last",
    "proxy_face_ray_perp_min",
    "proxy_face_ray_perp_mean",
    "proxy_face_ray_perp_slope",
    "proxy_face_ray_dwell_r050",
    "proxy_face_ray_dwell_r100",
    "proxy_face_ray_ahead_fraction",
)
FEATURE_COLUMNS = (*ROOT_PROXY_COLUMNS, *HAND_TTC_COLUMNS, *FACE_DWELL_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build input-only proxy features for teacher signals. Uses only observed joints_input "
            "and static candidate bbox geometry."
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


def vector3(values: Any) -> torch.Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return torch.zeros(3, dtype=torch.float32)
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
                if sample_id and sample_id not in output[split]:
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


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vector)
    if float(norm.item()) <= 1.0e-8:
        return torch.zeros_like(vector)
    return vector / norm


def horizontal_unit(vector: torch.Tensor) -> torch.Tensor:
    xz = torch.stack((vector[0], vector[2]))
    norm = torch.linalg.norm(xz)
    if float(norm.item()) <= 1.0e-8:
        return torch.zeros_like(xz)
    return xz / norm


def alignment_01(source: torch.Tensor, target: torch.Tensor) -> float:
    source_unit = horizontal_unit(source)
    target_unit = horizontal_unit(target)
    if float(torch.linalg.norm(source_unit).item()) <= 1.0e-8:
        return 0.5
    if float(torch.linalg.norm(target_unit).item()) <= 1.0e-8:
        return 0.5
    cosine = float((source_unit * target_unit).sum().item())
    return float(max(0.0, min(1.0, (cosine + 1.0) * 0.5)))


def orient_direction(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if float(torch.linalg.norm(reference).item()) > 1.0e-8 and float(torch.dot(direction, reference).item()) < 0.0:
        return -direction
    return direction


def temporal_slope(values: torch.Tensor) -> float:
    values = values.flatten().float()
    if values.numel() <= 1:
        return 0.0
    time = torch.arange(values.numel(), dtype=values.dtype)
    time = time - time.mean()
    denom = torch.sum(time * time).clamp_min(1.0e-8)
    return float(torch.sum(time * (values - values.mean())).item() / float(denom.item()))


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


def nearest_point_on_bbox(point: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    return torch.minimum(torch.maximum(point, low), high)


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


def ray_bbox_perp_and_ahead(
    origin: torch.Tensor,
    direction: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> tuple[float, float]:
    direction_xz = horizontal_unit(direction)
    if float(torch.linalg.norm(direction_xz).item()) <= 1.0e-8:
        return float(bbox_distance_xz(origin[None, :], bbox_min, bbox_max).item()), 0.0
    origin_xz = torch.tensor([float(origin[0]), float(origin[2])], dtype=torch.float32)
    points = bbox_points_xz(bbox_min, bbox_max)
    rel = points - origin_xz[None, :]
    along = rel @ direction_xz
    perp = torch.abs(rel[:, 0] * direction_xz[1] - rel[:, 1] * direction_xz[0])
    ahead = along >= 0.0
    if bool(ahead.any().item()):
        return float(perp[ahead].min().item()), 1.0
    return float(perp.min().item()), 0.0


def face_origin_direction(frame: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_joints = int(frame.shape[0])

    def has_joint(index: int) -> bool:
        return 0 <= index < num_joints

    def joint(source: torch.Tensor, index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return source[index]
        if has_joint(fallback):
            return source[fallback]
        return torch.zeros(3, dtype=torch.float32)

    pelvis = joint(frame, 0)
    neck = joint(frame, 12)
    head = joint(frame, 15, fallback=12)
    left_shoulder = joint(frame, 16, fallback=12)
    right_shoulder = joint(frame, 17, fallback=12)
    nose = joint(frame, 55, fallback=15)
    right_eye = joint(frame, 56, fallback=15)
    left_eye = joint(frame, 57, fallback=15)
    root_velocity = pelvis - joint(prev, 0)
    torso_direction = unit_vector(torch.cross(right_shoulder - left_shoulder, neck - pelvis, dim=0))
    torso_direction = orient_direction(torso_direction, root_velocity)
    eye_mid = 0.5 * (left_eye + right_eye)
    if has_joint(55) and has_joint(56) and has_joint(57):
        face_direction = unit_vector(torch.cross(nose - eye_mid, right_eye - left_eye, dim=0))
        face_direction = orient_direction(face_direction, torso_direction)
        return eye_mid, face_direction
    return head, head - joint(prev, 15, fallback=12)


def zero_features() -> dict[str, float]:
    return {column: 0.0 for column in FEATURE_COLUMNS}


def build_features(scene_sample: dict[str, Any], object_id: str, joints_input: torch.Tensor) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None or joints_input.numel() == 0:
        return zero_features()

    bbox_min = vector3(scene_object.get("bbox_min"))
    bbox_max = vector3(scene_object.get("bbox_max"))
    centroid = vector3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    root = joints_input[:, 0, :] if num_joints > 0 else torch.zeros((joints_input.shape[0], 3), dtype=torch.float32)
    first_root = root[0]
    last_root = root[-1]
    root_velocity = (last_root - first_root) / max(int(root.shape[0]) - 1, 1)
    last_root_distance_xz = float(bbox_distance_xz(last_root[None, :], bbox_min, bbox_max).item())

    features = zero_features()
    for horizon, column in (
        (2.0, "proxy_root_cv2_bbox_distance_xz"),
        (5.0, "proxy_root_cv5_bbox_distance_xz"),
        (10.0, "proxy_root_cv10_bbox_distance_xz"),
    ):
        endpoint = last_root + root_velocity * horizon
        features[column] = float(bbox_distance_xz(endpoint[None, :], bbox_min, bbox_max).item())
    endpoint10 = last_root + root_velocity * 10.0
    features["proxy_root_cv10_approach_gain_xz"] = last_root_distance_xz - features["proxy_root_cv10_bbox_distance_xz"]
    features["proxy_root_cv10_object_alignment"] = alignment_01(root_velocity, centroid - endpoint10)
    features["proxy_root_cv10_exp_score"] = float(math.exp(-max(features["proxy_root_cv10_bbox_distance_xz"], 0.0)))

    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    if hand_indices:
        hand_seq = joints_input[:, hand_indices, :].mean(dim=1)
    else:
        hand_seq = root
    hand_distances = bbox_distance_3d(hand_seq, bbox_min, bbox_max)
    hand_last = hand_seq[-1]
    hand_prev = hand_seq[-2] if hand_seq.shape[0] > 1 else hand_seq[0]
    hand_start_distance = float(hand_distances[0].item())
    hand_last_distance = float(hand_distances[-1].item())
    hand_velocity = hand_last - hand_prev
    nearest = nearest_point_on_bbox(hand_last, bbox_min, bbox_max)
    direction_to_bbox = unit_vector(nearest - hand_last)
    hand_speed_toward = float(torch.dot(hand_velocity, direction_to_bbox).item())
    if hand_speed_toward > 1.0e-8:
        ttc = hand_last_distance / hand_speed_toward
        ttc_score = math.exp(-max(ttc, 0.0) / 6.0)
    else:
        ttc_score = 0.0
    features.update(
        {
            "proxy_hand_last_bbox_distance_3d": hand_last_distance,
            "proxy_hand_min_bbox_distance_3d": float(hand_distances.min().item()),
            "proxy_hand_approach_delta_3d": hand_start_distance - hand_last_distance,
            "proxy_hand_distance_slope_3d": temporal_slope(hand_distances),
            "proxy_hand_speed_toward_bbox": hand_speed_toward,
            "proxy_hand_ttc_score": float(ttc_score),
            "proxy_hand_contact_dwell_r025": float((hand_distances <= 0.25).float().mean().item()),
        }
    )

    face_perp: list[float] = []
    face_ahead: list[float] = []
    for frame_index in range(int(joints_input.shape[0])):
        frame = joints_input[frame_index]
        prev = joints_input[max(frame_index - 1, 0)]
        origin, direction = face_origin_direction(frame, prev)
        perp, ahead = ray_bbox_perp_and_ahead(origin, direction, bbox_min, bbox_max)
        face_perp.append(perp)
        face_ahead.append(ahead)
    face_tensor = torch.tensor(face_perp, dtype=torch.float32)
    features.update(
        {
            "proxy_face_ray_perp_last": float(face_tensor[-1].item()),
            "proxy_face_ray_perp_min": float(face_tensor.min().item()),
            "proxy_face_ray_perp_mean": float(face_tensor.mean().item()),
            "proxy_face_ray_perp_slope": temporal_slope(face_tensor),
            "proxy_face_ray_dwell_r050": float((face_tensor <= 0.5).float().mean().item()),
            "proxy_face_ray_dwell_r100": float((face_tensor <= 1.0).float().mean().item()),
            "proxy_face_ray_ahead_fraction": float(torch.tensor(face_ahead, dtype=torch.float32).mean().item()),
        }
    )
    return features


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No candidate rows found in {args.input_csv}")
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
            features = build_features(scene_sample, object_id, input_joints[split][sample_index])
        enriched = {
            "split": split,
            "sample_id": sample_id,
            "object_id": object_id,
        }
        for column in FEATURE_COLUMNS:
            value = float(features.get(column, 0.0))
            if not math.isfinite(value):
                value = 0.0
            enriched[column] = f"{value:.9g}"
        output_rows.append(enriched)

    write_csv(args.output_csv, output_rows, ["split", "sample_id", "object_id", *FEATURE_COLUMNS])
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
