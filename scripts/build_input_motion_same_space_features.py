#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    / "teacher_candidate_scores_input_motion_7f.csv"
)
DEFAULT_CENTROID_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_centroid_4f.csv"
)
DEFAULT_FULL_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_centroid_align_13f.csv"
)
DEFAULT_SLOPE_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_centroid_align_slope_16f.csv"
)
DEFAULT_BODY_FACING_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_centroid_align_bodyface_16f.csv"
)
DEFAULT_LOCAL_FRAME_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_centroid_align_bodyface_local_19f.csv"
)
DEFAULT_PROXY_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_align_bodyface_proxy_16f.csv"
)
DEFAULT_GEOMETRY_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_align_bodyface_geometry_17f.csv"
)
DEFAULT_RAY_ENDPOINT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_ray_endpoint2.csv"
)
DEFAULT_TEMPORAL_ATTENTION_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_temporal_attention6.csv"
)

INPUT_MOTION_FEATURE_COLUMNS = (
    "input_root_bbox_distance",
    "input_hand_bbox_distance",
    "input_face_bbox_distance",
    "input_root_approach_delta",
    "input_hand_approach_delta",
    "input_face_approach_delta",
    "input_root_motion_alignment",
)
INPUT_CENTROID_FEATURE_COLUMNS = (
    "input_root_min_centroid_distance",
    "input_hand_min_centroid_distance",
    "input_face_min_centroid_distance",
    "input_body_min_centroid_distance",
)
INPUT_ALIGNMENT_FEATURE_COLUMNS = (
    "input_hand_motion_alignment",
    "input_face_motion_alignment",
)
INPUT_SLOPE_FEATURE_COLUMNS = (
    "input_root_distance_slope",
    "input_hand_distance_slope",
    "input_face_distance_slope",
)
INPUT_BODY_FACING_FEATURE_COLUMNS = (
    "input_torso_object_alignment",
    "input_head_object_alignment",
    "input_shoulder_object_alignment",
)
INPUT_LOCAL_FRAME_FEATURE_COLUMNS = (
    "input_torso_object_forward_coord",
    "input_torso_object_lateral_coord",
    "input_head_object_lateral_coord",
)
INPUT_PROXY_FEATURE_COLUMNS = (
    "input_root_pred_endpoint_distance",
    "input_root_pred_endpoint_alignment",
    "input_hand_speed_toward_object",
    "input_face_ray_distance_to_object",
)
INPUT_RAY_ENDPOINT_FEATURE_COLUMNS = (
    "input_root_ray_endpoint_score",
    "input_hand_ray_endpoint_score",
)
INPUT_GEOMETRY_FEATURE_COLUMNS = (
    "input_object_bbox_size_x",
    "input_object_bbox_size_y",
    "input_object_bbox_size_z",
    "input_object_bbox_volume",
    "input_object_centroid_height",
)
INPUT_TEMPORAL_ATTENTION_FEATURE_COLUMNS = (
    "input_torso_object_cone_fraction",
    "input_head_object_cone_fraction",
    "input_root_heading_object_fraction",
    "input_hand_heading_object_fraction",
    "input_face_ray_perp_mean",
    "input_face_ray_perp_min",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append student-visible input-motion features to a teacher candidate CSV. "
            "The features use only observed joints_input and static object geometry."
        )
    )
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument(
        "--feature-set",
        choices=(
            "input7_bbox",
            "centroid4",
            "input7_bbox_centroid4",
            "input7_bbox_centroid4_align2",
            "input7_bbox_centroid4_align2_slope3",
            "input7_bbox_centroid4_align2_bodyface3",
            "input7_bbox_centroid4_align2_bodyface3_local3",
            "input7_bbox_align2_bodyface3_proxy4",
            "input7_bbox_align2_bodyface3_geometry5",
            "ray_endpoint2",
            "temporal_attention6",
        ),
        default="input7_bbox",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
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
                if sample_id:
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


def as_tensor3(values: Any) -> torch.Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return torch.zeros(3, dtype=torch.float32)
    return torch.tensor([float(values[0]), float(values[1]), float(values[2])], dtype=torch.float32)


def bbox_distance(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    below = torch.clamp(bbox_min - points, min=0.0)
    above = torch.clamp(points - bbox_max, min=0.0)
    return torch.linalg.norm(below + above, dim=-1)


def safe_indices(num_joints: int, requested: tuple[int, ...]) -> list[int]:
    return [index for index in requested if 0 <= index < num_joints]


def horizontal_unit(vector: torch.Tensor) -> torch.Tensor:
    xz = torch.stack((vector[0], vector[2]))
    norm = torch.linalg.norm(xz)
    if float(norm.item()) <= 1.0e-8:
        return torch.zeros_like(xz)
    return xz / norm


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vector)
    if float(norm.item()) <= 1.0e-8:
        return torch.zeros_like(vector)
    return vector / norm


def horizontal_unit3(vector: torch.Tensor) -> torch.Tensor:
    flattened = vector.clone()
    flattened[1] = 0.0
    return unit_vector(flattened)


def horizontal_right3(forward: torch.Tensor) -> torch.Tensor:
    forward = horizontal_unit3(forward)
    if float(torch.linalg.norm(forward).item()) <= 1.0e-8:
        return torch.zeros_like(forward)
    return torch.tensor([forward[2], 0.0, -forward[0]], dtype=torch.float32)


def orient_direction(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if float(torch.linalg.norm(reference).item()) > 1.0e-8 and float(torch.dot(direction, reference).item()) < 0.0:
        return -direction
    return direction


def alignment_01(source: torch.Tensor, target: torch.Tensor) -> float:
    source_unit = horizontal_unit(source)
    target_unit = horizontal_unit(target)
    if float(torch.linalg.norm(source_unit).item()) <= 1.0e-8:
        return 0.5
    if float(torch.linalg.norm(target_unit).item()) <= 1.0e-8:
        return 0.5
    cosine = float((source_unit * target_unit).sum().item())
    return float(max(0.0, min(1.0, (cosine + 1.0) * 0.5)))


def horizontal_ray_distance(point: torch.Tensor, origin: torch.Tensor, direction: torch.Tensor) -> float:
    rel = torch.stack((point[0] - origin[0], point[2] - origin[2]))
    direction_xz = horizontal_unit(direction)
    if float(torch.linalg.norm(direction_xz).item()) <= 1.0e-8:
        return float(torch.linalg.norm(rel).item())
    along = torch.clamp(torch.dot(rel, direction_xz), min=0.0)
    projected = along * direction_xz
    return float(torch.linalg.norm(rel - projected).item())


def exp_score(distance: float, scale: float = 1.0) -> float:
    return float(math.exp(-max(float(distance), 0.0) / max(float(scale), 1.0e-6)))


def endpoint_distance(joints_input: torch.Tensor, indices: list[int], bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> float:
    if not indices:
        return 0.0
    return float(bbox_distance(joints_input[-1:, indices, :], bbox_min, bbox_max).min().item())


def min_centroid_distance(points: torch.Tensor, centroid: torch.Tensor) -> float:
    return float(torch.linalg.norm(points - centroid, dim=-1).min().item())


def centroid_distance_sequence(points: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(points - centroid, dim=-1)


def temporal_slope(values: torch.Tensor) -> float:
    values = values.flatten().float()
    if values.numel() <= 1:
        return 0.0
    time = torch.arange(values.numel(), dtype=values.dtype)
    time = time - time.mean()
    denom = torch.sum(time * time).clamp_min(1.0e-8)
    return float(torch.sum(time * (values - values.mean())).item() / float(denom.item()))


def approach_delta(joints_input: torch.Tensor, indices: list[int], bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> float:
    if not indices:
        return 0.0
    start = float(bbox_distance(joints_input[:1, indices, :], bbox_min, bbox_max).min().item())
    end = float(bbox_distance(joints_input[-1:, indices, :], bbox_min, bbox_max).min().item())
    return start - end


def face_points(joints_input: torch.Tensor, face_indices: list[int], head_indices: list[int]) -> torch.Tensor:
    if face_indices:
        return joints_input[:, face_indices, :].mean(dim=1, keepdim=True)
    if head_indices:
        return joints_input[:, head_indices, :]
    return joints_input[:, [0], :]


def joint_group_sequence(joints_input: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return joints_input[:, [0], :]
    return joints_input[:, indices, :].mean(dim=1, keepdim=True)


def centroid_motion_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_CENTROID_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    root_indices = safe_indices(num_joints, (0,))
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    body_indices = safe_indices(num_joints, (0, 12, 15))
    head_indices = safe_indices(num_joints, (15,))
    eye_nose_indices = safe_indices(num_joints, (55, 56, 57))
    face_sequence = face_points(joints_input, eye_nose_indices, head_indices)

    return {
        "input_root_min_centroid_distance": min_centroid_distance(joints_input[:, root_indices, :], centroid)
        if root_indices
        else 0.0,
        "input_hand_min_centroid_distance": min_centroid_distance(joints_input[:, hand_indices, :], centroid)
        if hand_indices
        else 0.0,
        "input_face_min_centroid_distance": min_centroid_distance(face_sequence, centroid),
        "input_body_min_centroid_distance": min_centroid_distance(joints_input[:, body_indices, :], centroid)
        if body_indices
        else 0.0,
    }


def alignment_motion_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_ALIGNMENT_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    head_indices = safe_indices(num_joints, (15,))
    eye_nose_indices = safe_indices(num_joints, (55, 56, 57))

    hand_sequence = joint_group_sequence(joints_input, hand_indices)
    face_sequence = face_points(joints_input, eye_nose_indices, head_indices)
    hand_motion = hand_sequence[-1, 0, :] - hand_sequence[0, 0, :]
    face_motion = face_sequence[-1, 0, :] - face_sequence[0, 0, :]
    hand_to_object = centroid - hand_sequence[-1, 0, :]
    face_to_object = centroid - face_sequence[-1, 0, :]

    return {
        "input_hand_motion_alignment": alignment_01(hand_motion, hand_to_object),
        "input_face_motion_alignment": alignment_01(face_motion, face_to_object),
    }


def slope_motion_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_SLOPE_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    head_indices = safe_indices(num_joints, (15,))
    eye_nose_indices = safe_indices(num_joints, (55, 56, 57))

    root_distances = centroid_distance_sequence(joints_input[:, 0, :], centroid)
    if hand_indices:
        hand_distances = centroid_distance_sequence(joints_input[:, hand_indices, :], centroid).min(dim=1).values
    else:
        hand_distances = root_distances
    face_sequence = face_points(joints_input, eye_nose_indices, head_indices)[:, 0, :]
    face_distances = centroid_distance_sequence(face_sequence, centroid)

    return {
        "input_root_distance_slope": temporal_slope(root_distances),
        "input_hand_distance_slope": temporal_slope(hand_distances),
        "input_face_distance_slope": temporal_slope(face_distances),
    }


def body_facing_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_BODY_FACING_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    last_joints = joints_input[-1]
    first_joints = joints_input[0]

    def has_joint(index: int) -> bool:
        return 0 <= index < num_joints

    def joint(index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return last_joints[index]
        if has_joint(fallback):
            return last_joints[fallback]
        return torch.zeros(3, dtype=torch.float32)

    def motion(index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return last_joints[index] - first_joints[index]
        if has_joint(fallback):
            return last_joints[fallback] - first_joints[fallback]
        return torch.zeros(3, dtype=torch.float32)

    pelvis = joint(0)
    neck = joint(12)
    head = joint(15, fallback=12)
    left_shoulder = joint(16, fallback=12)
    right_shoulder = joint(17, fallback=12)
    nose = joint(55, fallback=15)
    right_eye = joint(56, fallback=15)
    left_eye = joint(57, fallback=15)

    root_motion = unit_vector(motion(0))
    shoulder_axis = right_shoulder - left_shoulder
    torso_direction = unit_vector(torch.cross(shoulder_axis, neck - pelvis, dim=0))
    torso_direction = orient_direction(torso_direction, root_motion)

    eye_mid = 0.5 * (left_eye + right_eye)
    if has_joint(55) and has_joint(56) and has_joint(57):
        face_direction = unit_vector(torch.cross(nose - eye_mid, right_eye - left_eye, dim=0))
        face_direction = orient_direction(face_direction, torso_direction)
        head_origin = eye_mid
    else:
        face_direction = motion(15, fallback=12)
        head_origin = head

    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    shoulder_direction = unit_vector(torch.cross(shoulder_axis, world_up, dim=0))
    shoulder_reference = torso_direction if float(torch.linalg.norm(torso_direction).item()) > 1.0e-8 else root_motion
    shoulder_direction = orient_direction(shoulder_direction, shoulder_reference)
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)

    return {
        "input_torso_object_alignment": alignment_01(torso_direction, centroid - pelvis),
        "input_head_object_alignment": alignment_01(face_direction, centroid - head_origin),
        "input_shoulder_object_alignment": alignment_01(shoulder_direction, centroid - shoulder_center),
    }


def local_frame_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_LOCAL_FRAME_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    last_joints = joints_input[-1]
    first_joints = joints_input[0]

    def has_joint(index: int) -> bool:
        return 0 <= index < num_joints

    def joint(index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return last_joints[index]
        if has_joint(fallback):
            return last_joints[fallback]
        return torch.zeros(3, dtype=torch.float32)

    def motion(index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return last_joints[index] - first_joints[index]
        if has_joint(fallback):
            return last_joints[fallback] - first_joints[fallback]
        return torch.zeros(3, dtype=torch.float32)

    pelvis = joint(0)
    neck = joint(12)
    head = joint(15, fallback=12)
    left_shoulder = joint(16, fallback=12)
    right_shoulder = joint(17, fallback=12)
    nose = joint(55, fallback=15)
    right_eye = joint(56, fallback=15)
    left_eye = joint(57, fallback=15)

    root_motion = unit_vector(motion(0))
    torso_forward = unit_vector(torch.cross(right_shoulder - left_shoulder, neck - pelvis, dim=0))
    torso_forward = horizontal_unit3(orient_direction(torso_forward, root_motion))
    torso_right = horizontal_right3(torso_forward)

    eye_mid = 0.5 * (left_eye + right_eye)
    if has_joint(55) and has_joint(56) and has_joint(57):
        head_forward = unit_vector(torch.cross(nose - eye_mid, right_eye - left_eye, dim=0))
        head_forward = orient_direction(head_forward, torso_forward)
        head_origin = eye_mid
    else:
        head_forward = motion(15, fallback=12)
        head_origin = head
    head_forward = horizontal_unit3(head_forward)
    head_right = horizontal_right3(head_forward)

    torso_to_object = centroid - pelvis
    head_to_object = centroid - head_origin
    return {
        "input_torso_object_forward_coord": float(torch.dot(torso_to_object, torso_forward).item()),
        "input_torso_object_lateral_coord": float(torch.dot(torso_to_object, torso_right).item()),
        "input_head_object_lateral_coord": float(torch.dot(head_to_object, head_right).item()),
    }


def proxy_motion_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_PROXY_FEATURE_COLUMNS}

    bbox_min = as_tensor3(scene_object.get("bbox_min"))
    bbox_max = as_tensor3(scene_object.get("bbox_max"))
    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    last_joints = joints_input[-1]
    first_joints = joints_input[0]
    prev_joints = joints_input[-2] if int(joints_input.shape[0]) >= 2 else first_joints

    def has_joint(index: int) -> bool:
        return 0 <= index < num_joints

    def joint(index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return last_joints[index]
        if has_joint(fallback):
            return last_joints[fallback]
        return torch.zeros(3, dtype=torch.float32)

    pelvis = joint(0)
    root_motion = pelvis - first_joints[0] if has_joint(0) else torch.zeros(3, dtype=torch.float32)
    predicted_root = pelvis + root_motion

    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    if hand_indices:
        hand_last = last_joints[hand_indices, :].mean(dim=0)
        hand_prev = prev_joints[hand_indices, :].mean(dim=0)
    else:
        hand_last = pelvis
        hand_prev = prev_joints[0] if has_joint(0) else pelvis
    hand_velocity = hand_last - hand_prev
    hand_to_object = centroid - hand_last
    hand_direction = unit_vector(hand_to_object)
    hand_speed_toward = float(torch.dot(hand_velocity, hand_direction).item())

    neck = joint(12)
    head = joint(15, fallback=12)
    left_shoulder = joint(16, fallback=12)
    right_shoulder = joint(17, fallback=12)
    nose = joint(55, fallback=15)
    right_eye = joint(56, fallback=15)
    left_eye = joint(57, fallback=15)
    torso_direction = unit_vector(torch.cross(right_shoulder - left_shoulder, neck - pelvis, dim=0))
    torso_direction = orient_direction(torso_direction, root_motion)

    eye_mid = 0.5 * (left_eye + right_eye)
    if has_joint(55) and has_joint(56) and has_joint(57):
        face_direction = unit_vector(torch.cross(nose - eye_mid, right_eye - left_eye, dim=0))
        face_direction = orient_direction(face_direction, torso_direction)
        face_origin = eye_mid
    else:
        face_direction = head - (prev_joints[15] if has_joint(15) else prev_joints[0])
        face_origin = head

    return {
        "input_root_pred_endpoint_distance": float(
            bbox_distance(predicted_root[None, None, :], bbox_min, bbox_max).min().item()
        ),
        "input_root_pred_endpoint_alignment": alignment_01(root_motion, centroid - predicted_root),
        "input_hand_speed_toward_object": hand_speed_toward,
        "input_face_ray_distance_to_object": horizontal_ray_distance(centroid, face_origin, face_direction),
    }


def ray_endpoint_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_RAY_ENDPOINT_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    last_joints = joints_input[-1]
    prev_joints = joints_input[-2] if int(joints_input.shape[0]) >= 2 else joints_input[0]

    def has_joint(index: int) -> bool:
        return 0 <= index < num_joints

    root_origin = last_joints[0] if has_joint(0) else torch.zeros(3, dtype=torch.float32)
    root_direction = (last_joints[0] - prev_joints[0]) if has_joint(0) else torch.zeros(3, dtype=torch.float32)
    root_ray_distance = horizontal_ray_distance(centroid, root_origin, root_direction)

    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    hand_ray_distances: list[float] = []
    for hand_index in hand_indices:
        hand_origin = last_joints[hand_index]
        hand_direction = last_joints[hand_index] - prev_joints[hand_index]
        hand_ray_distances.append(horizontal_ray_distance(centroid, hand_origin, hand_direction))
    hand_ray_distance = min(hand_ray_distances) if hand_ray_distances else root_ray_distance

    return {
        "input_root_ray_endpoint_score": exp_score(root_ray_distance, scale=1.0),
        "input_hand_ray_endpoint_score": exp_score(hand_ray_distance, scale=1.0),
    }


def temporal_attention_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_TEMPORAL_ATTENTION_FEATURE_COLUMNS}

    centroid = as_tensor3(scene_object.get("centroid"))
    num_joints = int(joints_input.shape[1])
    if num_joints <= 0:
        return {column: 0.0 for column in INPUT_TEMPORAL_ATTENTION_FEATURE_COLUMNS}

    def has_joint(index: int) -> bool:
        return 0 <= index < num_joints

    def joint(frame_joints: torch.Tensor, index: int, fallback: int = 0) -> torch.Tensor:
        if has_joint(index):
            return frame_joints[index]
        if has_joint(fallback):
            return frame_joints[fallback]
        return torch.zeros(3, dtype=torch.float32)

    def horizontal_cosine(source: torch.Tensor, target: torch.Tensor) -> float:
        source_unit = horizontal_unit(source)
        target_unit = horizontal_unit(target)
        if float(torch.linalg.norm(source_unit).item()) <= 1.0e-8:
            return 0.0
        if float(torch.linalg.norm(target_unit).item()) <= 1.0e-8:
            return 0.0
        return float((source_unit * target_unit).sum().item())

    def is_in_cone(source: torch.Tensor, target: torch.Tensor, threshold: float = 0.70710678) -> float:
        return float(horizontal_cosine(source, target) >= threshold)

    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    cone_torso: list[float] = []
    cone_head: list[float] = []
    heading_root: list[float] = []
    heading_hand: list[float] = []
    face_ray_distances: list[float] = []

    prev_torso_direction = torch.zeros(3, dtype=torch.float32)
    for frame_index in range(int(joints_input.shape[0])):
        frame = joints_input[frame_index]
        prev = joints_input[max(frame_index - 1, 0)]

        pelvis = joint(frame, 0)
        neck = joint(frame, 12)
        head = joint(frame, 15, fallback=12)
        left_shoulder = joint(frame, 16, fallback=12)
        right_shoulder = joint(frame, 17, fallback=12)
        nose = joint(frame, 55, fallback=15)
        right_eye = joint(frame, 56, fallback=15)
        left_eye = joint(frame, 57, fallback=15)

        root_velocity = pelvis - joint(prev, 0)
        shoulder_axis = right_shoulder - left_shoulder
        torso_direction = unit_vector(torch.cross(shoulder_axis, neck - pelvis, dim=0))
        torso_reference = root_velocity
        if float(torch.linalg.norm(torso_reference).item()) <= 1.0e-8:
            torso_reference = prev_torso_direction
        torso_direction = orient_direction(torso_direction, torso_reference)
        prev_torso_direction = torso_direction

        eye_mid = 0.5 * (left_eye + right_eye)
        if has_joint(55) and has_joint(56) and has_joint(57):
            face_direction = unit_vector(torch.cross(nose - eye_mid, right_eye - left_eye, dim=0))
            face_direction = orient_direction(face_direction, torso_direction)
            face_origin = eye_mid
        else:
            face_direction = head - joint(prev, 15, fallback=12)
            face_origin = head

        if hand_indices:
            hand_pos = frame[hand_indices, :].mean(dim=0)
            hand_prev = prev[hand_indices, :].mean(dim=0)
        else:
            hand_pos = pelvis
            hand_prev = joint(prev, 0)

        cone_torso.append(is_in_cone(torso_direction, centroid - pelvis))
        cone_head.append(is_in_cone(face_direction, centroid - face_origin))
        heading_root.append(is_in_cone(root_velocity, centroid - pelvis))
        heading_hand.append(is_in_cone(hand_pos - hand_prev, centroid - hand_pos))
        face_ray_distances.append(horizontal_ray_distance(centroid, face_origin, face_direction))

    face_ray_tensor = torch.tensor(face_ray_distances, dtype=torch.float32)
    return {
        "input_torso_object_cone_fraction": float(torch.tensor(cone_torso).mean().item()) if cone_torso else 0.0,
        "input_head_object_cone_fraction": float(torch.tensor(cone_head).mean().item()) if cone_head else 0.0,
        "input_root_heading_object_fraction": float(torch.tensor(heading_root).mean().item()) if heading_root else 0.0,
        "input_hand_heading_object_fraction": float(torch.tensor(heading_hand).mean().item()) if heading_hand else 0.0,
        "input_face_ray_perp_mean": float(face_ray_tensor.mean().item()) if face_ray_distances else 0.0,
        "input_face_ray_perp_min": float(face_ray_tensor.min().item()) if face_ray_distances else 0.0,
    }


def object_geometry_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_GEOMETRY_FEATURE_COLUMNS}

    bbox_min = as_tensor3(scene_object.get("bbox_min"))
    bbox_max = as_tensor3(scene_object.get("bbox_max"))
    centroid = as_tensor3(scene_object.get("centroid"))
    size = torch.clamp(bbox_max - bbox_min, min=0.0)
    volume = float(torch.prod(size).item())
    return {
        "input_object_bbox_size_x": float(size[0].item()),
        "input_object_bbox_size_y": float(size[1].item()),
        "input_object_bbox_size_z": float(size[2].item()),
        "input_object_bbox_volume": volume,
        "input_object_centroid_height": float(centroid[1].item()),
    }


def input_motion_features_for_candidate(
    *,
    scene_sample: dict[str, Any],
    object_id: str,
    joints_input: torch.Tensor,
) -> dict[str, float]:
    objects = {str(item.get("object_id", "")): item for item in scene_sample.get("objects", [])}
    scene_object = objects.get(object_id)
    if scene_object is None:
        return {column: 0.0 for column in INPUT_MOTION_FEATURE_COLUMNS}

    bbox_min = as_tensor3(scene_object.get("bbox_min"))
    bbox_max = as_tensor3(scene_object.get("bbox_max"))
    centroid = as_tensor3(scene_object.get("centroid"))

    num_joints = int(joints_input.shape[1])
    root_indices = safe_indices(num_joints, (0,))
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    head_indices = safe_indices(num_joints, (15,))
    eye_nose_indices = safe_indices(num_joints, (55, 56, 57))
    face_sequence = face_points(joints_input, eye_nose_indices, head_indices)

    root_motion = joints_input[-1, 0, :] - joints_input[0, 0, :]
    root_to_object = centroid - joints_input[-1, 0, :]

    return {
        "input_root_bbox_distance": endpoint_distance(joints_input, root_indices, bbox_min, bbox_max),
        "input_hand_bbox_distance": endpoint_distance(joints_input, hand_indices, bbox_min, bbox_max),
        "input_face_bbox_distance": float(bbox_distance(face_sequence[-1:], bbox_min, bbox_max).min().item()),
        "input_root_approach_delta": approach_delta(joints_input, root_indices, bbox_min, bbox_max),
        "input_hand_approach_delta": approach_delta(joints_input, hand_indices, bbox_min, bbox_max),
        "input_face_approach_delta": float(
            bbox_distance(face_sequence[:1], bbox_min, bbox_max).min().item()
            - bbox_distance(face_sequence[-1:], bbox_min, bbox_max).min().item()
        ),
        "input_root_motion_alignment": alignment_01(root_motion, root_to_object),
    }


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        if args.feature_set == "centroid4":
            output_csv = DEFAULT_CENTROID_OUTPUT_CSV
        elif args.feature_set == "input7_bbox_centroid4_align2_slope3":
            output_csv = DEFAULT_SLOPE_OUTPUT_CSV
        elif args.feature_set == "input7_bbox_centroid4_align2_bodyface3":
            output_csv = DEFAULT_BODY_FACING_OUTPUT_CSV
        elif args.feature_set == "input7_bbox_centroid4_align2_bodyface3_local3":
            output_csv = DEFAULT_LOCAL_FRAME_OUTPUT_CSV
        elif args.feature_set == "input7_bbox_align2_bodyface3_proxy4":
            output_csv = DEFAULT_PROXY_OUTPUT_CSV
        elif args.feature_set == "input7_bbox_align2_bodyface3_geometry5":
            output_csv = DEFAULT_GEOMETRY_OUTPUT_CSV
        elif args.feature_set == "ray_endpoint2":
            output_csv = DEFAULT_RAY_ENDPOINT_OUTPUT_CSV
        elif args.feature_set == "temporal_attention6":
            output_csv = DEFAULT_TEMPORAL_ATTENTION_OUTPUT_CSV
        elif args.feature_set == "input7_bbox_centroid4_align2":
            output_csv = DEFAULT_FULL_OUTPUT_CSV
        else:
            output_csv = DEFAULT_OUTPUT_CSV

    if args.feature_set == "input7_bbox":
        feature_columns = INPUT_MOTION_FEATURE_COLUMNS
    elif args.feature_set == "centroid4":
        feature_columns = INPUT_CENTROID_FEATURE_COLUMNS
    elif args.feature_set == "input7_bbox_centroid4_align2":
        feature_columns = (
            *INPUT_MOTION_FEATURE_COLUMNS,
            *INPUT_CENTROID_FEATURE_COLUMNS,
            *INPUT_ALIGNMENT_FEATURE_COLUMNS,
        )
    elif args.feature_set == "input7_bbox_centroid4_align2_slope3":
        feature_columns = (
            *INPUT_MOTION_FEATURE_COLUMNS,
            *INPUT_CENTROID_FEATURE_COLUMNS,
            *INPUT_ALIGNMENT_FEATURE_COLUMNS,
            *INPUT_SLOPE_FEATURE_COLUMNS,
        )
    elif args.feature_set == "input7_bbox_centroid4_align2_bodyface3":
        feature_columns = (
            *INPUT_MOTION_FEATURE_COLUMNS,
            *INPUT_CENTROID_FEATURE_COLUMNS,
            *INPUT_ALIGNMENT_FEATURE_COLUMNS,
            *INPUT_BODY_FACING_FEATURE_COLUMNS,
        )
    elif args.feature_set == "input7_bbox_centroid4_align2_bodyface3_local3":
        feature_columns = (
            *INPUT_MOTION_FEATURE_COLUMNS,
            *INPUT_CENTROID_FEATURE_COLUMNS,
            *INPUT_ALIGNMENT_FEATURE_COLUMNS,
            *INPUT_BODY_FACING_FEATURE_COLUMNS,
            *INPUT_LOCAL_FRAME_FEATURE_COLUMNS,
        )
    elif args.feature_set == "input7_bbox_align2_bodyface3_proxy4":
        feature_columns = (
            *INPUT_MOTION_FEATURE_COLUMNS,
            *INPUT_ALIGNMENT_FEATURE_COLUMNS,
            *INPUT_BODY_FACING_FEATURE_COLUMNS,
            *INPUT_PROXY_FEATURE_COLUMNS,
        )
    elif args.feature_set == "input7_bbox_align2_bodyface3_geometry5":
        feature_columns = (
            *INPUT_MOTION_FEATURE_COLUMNS,
            *INPUT_ALIGNMENT_FEATURE_COLUMNS,
            *INPUT_BODY_FACING_FEATURE_COLUMNS,
            *INPUT_GEOMETRY_FEATURE_COLUMNS,
        )
    elif args.feature_set == "ray_endpoint2":
        feature_columns = INPUT_RAY_ENDPOINT_FEATURE_COLUMNS
    elif args.feature_set == "temporal_attention6":
        feature_columns = INPUT_TEMPORAL_ATTENTION_FEATURE_COLUMNS
    else:
        feature_columns = (*INPUT_MOTION_FEATURE_COLUMNS, *INPUT_CENTROID_FEATURE_COLUMNS)

    rows = read_csv(args.candidate_csv)
    if not rows:
        raise RuntimeError(f"No candidate rows found in {args.candidate_csv}")

    scene_graphs = load_scene_graphs(args.scene_graph_root)
    input_joints = load_input_joints(args.slice_root)
    missing_rows = 0
    output_rows: list[dict[str, Any]] = []

    for row in rows:
        split = row.get("split", "")
        sample_id = row.get("sample_id", "")
        object_id = row.get("object_id", "")
        scene_sample = scene_graphs.get(split, {}).get(sample_id)
        if scene_sample is None or not object_id:
            missing_rows += 1
            features = {column: 0.0 for column in feature_columns}
        else:
            sample_index = int(scene_sample["sample_index"])
            features = {}
            if args.feature_set in {
                "input7_bbox",
                "input7_bbox_centroid4",
                "input7_bbox_centroid4_align2",
                "input7_bbox_centroid4_align2_slope3",
                "input7_bbox_centroid4_align2_bodyface3",
                "input7_bbox_centroid4_align2_bodyface3_local3",
                "input7_bbox_align2_bodyface3_proxy4",
                "input7_bbox_align2_bodyface3_geometry5",
            }:
                features.update(
                    input_motion_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set in {"centroid4", "input7_bbox_centroid4"}:
                features.update(
                    centroid_motion_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set in {
                "input7_bbox_centroid4_align2",
                "input7_bbox_centroid4_align2_slope3",
                "input7_bbox_centroid4_align2_bodyface3",
                "input7_bbox_centroid4_align2_bodyface3_local3",
            }:
                features.update(
                    centroid_motion_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set == "input7_bbox_centroid4_align2_slope3":
                features.update(
                    slope_motion_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set in {
                "input7_bbox_centroid4_align2",
                "input7_bbox_centroid4_align2_slope3",
                "input7_bbox_centroid4_align2_bodyface3",
                "input7_bbox_centroid4_align2_bodyface3_local3",
                "input7_bbox_align2_bodyface3_proxy4",
                "input7_bbox_align2_bodyface3_geometry5",
            }:
                features.update(
                    alignment_motion_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set in {
                "input7_bbox_centroid4_align2_bodyface3",
                "input7_bbox_centroid4_align2_bodyface3_local3",
                "input7_bbox_align2_bodyface3_proxy4",
                "input7_bbox_align2_bodyface3_geometry5",
            }:
                features.update(
                    body_facing_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set == "input7_bbox_centroid4_align2_bodyface3_local3":
                features.update(
                    local_frame_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set == "input7_bbox_align2_bodyface3_proxy4":
                features.update(
                    proxy_motion_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set == "input7_bbox_align2_bodyface3_geometry5":
                features.update(
                    object_geometry_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                    )
                )
            if args.feature_set == "ray_endpoint2":
                features.update(
                    ray_endpoint_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )
            if args.feature_set == "temporal_attention6":
                features.update(
                    temporal_attention_features_for_candidate(
                        scene_sample=scene_sample,
                        object_id=object_id,
                        joints_input=input_joints[split][sample_index],
                    )
                )

        enriched = dict(row)
        for column, value in features.items():
            if not math.isfinite(float(value)):
                value = 0.0
            enriched[column] = f"{float(value):.9g}"
        output_rows.append(enriched)

    fieldnames = list(rows[0].keys())
    for column in feature_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(output_csv, output_rows, fieldnames)
    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "num_rows": len(output_rows),
                "missing_rows": missing_rows,
                "feature_set": args.feature_set,
                "feature_columns": list(feature_columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
