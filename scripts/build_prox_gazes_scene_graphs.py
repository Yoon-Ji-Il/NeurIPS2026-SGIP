#!/usr/bin/env python
"""Build synthetic gaze points and object-selector scene graphs for PROX/COINS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


SCHEMA_VERSION = "object_selector_scene_graph_v1"
FLOOR_WALL = {"floor", "wall"}


@dataclass(frozen=True)
class DatasetRow:
    split: str
    split_index: int
    sample_id: str
    sequence_path: str
    scene_name: str
    start_frame: int
    end_frame: int
    training: str


@dataclass(frozen=True)
class ObjectInfo:
    object_id: str
    label: str
    semantic_label_id: int | None
    object_idx: int | None
    source_ply_path: Path
    raw_num_points: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    is_floor_wall: bool


@dataclass(frozen=True)
class MotionCues:
    pelvis: np.ndarray
    head: np.ndarray
    neck: np.ndarray
    eye_mid: np.ndarray
    root_motion_direction: np.ndarray
    root_pose_direction: np.ndarray
    root_direction: np.ndarray
    torso_direction: np.ndarray
    face_normal_raw_direction: np.ndarray
    face_normal_direction: np.ndarray
    valid: dict[str, bool]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha1("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def split_from_training(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "train"}:
        return "train"
    if text in {"val", "valid", "validation"}:
        return "val"
    return "test"


def normalize_row_keys(row: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None or key == "":
            continue
        cleaned[key.lstrip("\ufeff").strip()] = value.strip() if isinstance(value, str) else value
    return cleaned


def normalize_frame_value(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def graph_sample_id(scene_name: str, sequence_path: str, start_frame: int, end_frame: int) -> str:
    return "|".join(
        [
            str(scene_name).strip(),
            str(sequence_path).strip(),
            normalize_frame_value(start_frame),
            normalize_frame_value(end_frame),
        ]
    )


def load_dataset_rows(dataset_csv: Path) -> dict[str, list[DatasetRow]]:
    split_indices: dict[str, int] = defaultdict(int)
    rows_by_split: dict[str, list[DatasetRow]] = defaultdict(list)
    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for raw_row in csv.DictReader(file):
            row = normalize_row_keys(raw_row)
            if not row.get("sequence_path") or not row.get("scene"):
                continue
            split = split_from_training(row.get("training", "0"))
            split_index = split_indices[split]
            split_indices[split] += 1
            start_frame = int(float(row["start_frame"]))
            end_frame = int(float(row["end_frame"]))
            scene_name = row["scene"]
            sequence_path = row["sequence_path"]
            rows_by_split[split].append(
                DatasetRow(
                    split=split,
                    split_index=split_index,
                    sample_id=graph_sample_id(scene_name, sequence_path, start_frame, end_frame),
                    sequence_path=sequence_path,
                    scene_name=scene_name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    training=row.get("training", ""),
                )
            )
    return dict(rows_by_split)


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing tensor: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(loaded)!r}")
    return loaded.detach().cpu().float()


def normalize_vector(vector: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, bool]:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32), False
    return (vector / norm).astype(np.float32), True


def apply_rotvec(rotvec: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1.0e-8:
        return vector.copy()
    axis = rotvec / theta
    cos = math.cos(theta)
    sin = math.sin(theta)
    return (
        vector * cos
        + np.cross(axis, vector) * sin
        + axis * float(np.dot(axis, vector)) * (1.0 - cos)
    ).astype(np.float32)


def orient_direction(direction: np.ndarray, reference: np.ndarray, reference_valid: bool) -> np.ndarray:
    if reference_valid and float(np.dot(direction, reference)) < 0.0:
        return (-direction).astype(np.float32)
    return direction.astype(np.float32)


def xz(vector: np.ndarray) -> np.ndarray:
    return np.asarray([vector[0], vector[2]], dtype=np.float32)


def alignment_3d(point: np.ndarray, origin: np.ndarray, direction: np.ndarray, direction_valid: bool) -> float | None:
    if not direction_valid:
        return None
    point_direction, valid = normalize_vector(point - origin)
    if not valid:
        return None
    return float(np.dot(point_direction, direction))


def horizontal_alignment(point: np.ndarray, origin: np.ndarray, direction: np.ndarray, direction_valid: bool) -> float | None:
    if not direction_valid:
        return None
    point_direction_2d, point_valid = normalize_vector(xz(point) - xz(origin))
    heading_2d, heading_valid = normalize_vector(xz(direction))
    if not point_valid or not heading_valid:
        return None
    return float(np.dot(point_direction_2d[:2], heading_2d[:2]))


def horizontal_line_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray, direction_valid: bool) -> float | None:
    rel = xz(point) - xz(origin)
    if not direction_valid:
        return float(np.linalg.norm(rel))
    heading_2d, heading_valid = normalize_vector(xz(direction))
    if not heading_valid:
        return float(np.linalg.norm(rel))
    heading = heading_2d[:2]
    along = max(float(np.dot(rel, heading)), 0.0)
    projected = along * heading
    return float(np.linalg.norm(rel - projected))


def min_distance(points: np.ndarray, origin: np.ndarray) -> float | None:
    if points.size == 0:
        return None
    return float(np.linalg.norm(points - origin[None, :], axis=1).min())


def mean_non_null(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def finite_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def vector_payload(vector: np.ndarray, digits: int) -> list[float]:
    return [round(float(component), digits) for component in vector.tolist()]


def build_motion_cue_for_frame(
    joints: np.ndarray,
    poses: np.ndarray,
    frame_idx: int,
    face_orientation: str,
) -> MotionCues:
    frame_joints = joints[frame_idx]
    first_joints = joints[0]
    num_joints = frame_joints.shape[0]

    def joint(index: int) -> np.ndarray:
        if index >= num_joints:
            return np.zeros(3, dtype=np.float32)
        return frame_joints[index].astype(np.float32)

    pelvis = joint(0)
    neck = joint(12)
    head = joint(15)
    left_shoulder = joint(16)
    right_shoulder = joint(17)
    nose = joint(55)
    right_eye = joint(56)
    left_eye = joint(57)
    eye_mid = (0.5 * (left_eye + right_eye)).astype(np.float32)

    root_motion_direction, root_motion_valid = normalize_vector(frame_joints[0] - first_joints[0])
    root_pose_direction, root_pose_valid = normalize_vector(
        apply_rotvec(poses[frame_idx, :3], np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    )
    root_direction = root_pose_direction if root_pose_valid else root_motion_direction
    root_valid = root_pose_valid or root_motion_valid

    torso_direction_raw = np.cross(right_shoulder - left_shoulder, neck - pelvis)
    torso_direction, torso_valid = normalize_vector(torso_direction_raw)
    torso_direction = orient_direction(torso_direction, root_direction, root_valid)

    eye_axis = right_eye - left_eye
    nose_axis = nose - eye_mid
    face_normal_raw_direction, face_raw_valid = normalize_vector(np.cross(nose_axis, eye_axis))
    references = {
        "none": (np.zeros(3, dtype=np.float32), False),
        "root": (root_direction, root_valid),
        "torso": (torso_direction, torso_valid),
    }
    face_reference, face_reference_valid = references[face_orientation]
    face_normal_direction = orient_direction(face_normal_raw_direction, face_reference, face_reference_valid)

    return MotionCues(
        pelvis=pelvis,
        head=head,
        neck=neck,
        eye_mid=eye_mid,
        root_motion_direction=root_motion_direction,
        root_pose_direction=root_pose_direction,
        root_direction=root_direction,
        torso_direction=torso_direction,
        face_normal_raw_direction=face_normal_raw_direction,
        face_normal_direction=face_normal_direction,
        valid={
            "root_motion_direction": root_motion_valid,
            "root_pose_direction": root_pose_valid,
            "root_direction": root_valid,
            "torso_direction": torso_valid,
            "face_normal_raw_direction": face_raw_valid,
            "face_normal_direction": face_raw_valid,
        },
    )


def ray_hit_scene_points(
    origin: np.ndarray,
    direction: np.ndarray,
    direction_valid: bool,
    scene_points: np.ndarray,
    fallback_distance: float,
    min_forward: float = 0.05,
    tie_perp_margin: float = 0.05,
) -> np.ndarray:
    if not direction_valid or scene_points.size == 0:
        return (origin + np.asarray([0.0, 0.0, fallback_distance], dtype=np.float32)).astype(np.float32)

    rel = scene_points - origin[None, :]
    along = rel @ direction
    forward_mask = along > min_forward
    if not np.any(forward_mask):
        return (origin + direction * fallback_distance).astype(np.float32)

    forward_points = scene_points[forward_mask]
    forward_rel = rel[forward_mask]
    forward_along = along[forward_mask]
    rel_norm2 = np.einsum("ij,ij->i", forward_rel, forward_rel)
    perp2 = np.maximum(rel_norm2 - forward_along * forward_along, 0.0)
    best_perp = float(perp2.min())
    tie_mask = perp2 <= best_perp + tie_perp_margin * tie_perp_margin
    tied_indices = np.flatnonzero(tie_mask)
    chosen_local = tied_indices[int(np.argmin(forward_along[tie_mask]))]
    return forward_points[chosen_local].astype(np.float32)


def build_gaze_tensor(
    joints: torch.Tensor,
    poses: torch.Tensor,
    scene_points: torch.Tensor,
    face_orientation: str,
    fallback_distance: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    n_samples, input_len = joints.shape[:2]
    gazes = torch.zeros((n_samples, input_len, 1, 3), dtype=torch.float32)
    meta: list[dict[str, Any]] = []
    for sample_idx in range(n_samples):
        sample_joints = joints[sample_idx].numpy()
        sample_poses = poses[sample_idx].numpy()
        sample_scene = scene_points[sample_idx].numpy()
        frame_meta = []
        for frame_idx in range(input_len):
            cues = build_motion_cue_for_frame(sample_joints, sample_poses, frame_idx, face_orientation)
            hit = ray_hit_scene_points(
                cues.eye_mid,
                cues.face_normal_direction,
                cues.valid["face_normal_direction"],
                sample_scene,
                fallback_distance=fallback_distance,
            )
            gazes[sample_idx, frame_idx, 0] = torch.from_numpy(hit)
            frame_meta.append(
                {
                    "frame_index": frame_idx,
                    "eye_mid": cues.eye_mid.tolist(),
                    "face_normal_direction": cues.face_normal_direction.tolist(),
                    "valid_face_normal": bool(cues.valid["face_normal_direction"]),
                    "gaze_point": hit.tolist(),
                }
            )
        meta.append({"sample_index": sample_idx, "frames": frame_meta})
    return gazes, meta


def load_instance_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        points = []
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices"):
                points.append(np.asarray(geom.vertices, dtype=np.float32))
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(points, axis=0).astype(np.float32)
    if hasattr(loaded, "vertices"):
        return np.asarray(loaded.vertices, dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def sample_object_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if points.size == 0 or count <= 0 or points.shape[0] <= count:
        return points.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=count, replace=False)
    return points[idx].astype(np.float32, copy=False)


def load_scene_objects(scene_objects_root: Path, scene_name: str, include_floor_wall: bool) -> tuple[list[ObjectInfo], list[ObjectInfo]]:
    catalog_path = scene_objects_root / scene_name / "label_pointcloud_export.json"
    if not catalog_path.exists():
        return [], []
    catalog = json.loads(catalog_path.read_text())
    candidates: list[ObjectInfo] = []
    excluded: list[ObjectInfo] = []
    for row in catalog.get("instances", []):
        label = str(row.get("semantic_label_name") or row.get("label") or "unknown")
        is_floor_wall = label.strip().lower() in FLOOR_WALL
        ply_path = scene_objects_root / scene_name / str(row.get("ply_path", ""))
        info = ObjectInfo(
            object_id=str(row.get("instance_id") or f"instance_{int(row.get('object_idx', -1)):04d}"),
            label=label,
            semantic_label_id=row.get("semantic_label_id"),
            object_idx=row.get("object_idx"),
            source_ply_path=ply_path,
            raw_num_points=int(row.get("num_points") or 0),
            centroid=np.asarray(row.get("centroid") or [0.0, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.asarray(row.get("bbox_min") or [0.0, 0.0, 0.0], dtype=np.float32),
            bbox_max=np.asarray(row.get("bbox_max") or [0.0, 0.0, 0.0], dtype=np.float32),
            is_floor_wall=is_floor_wall,
        )
        if is_floor_wall and not include_floor_wall:
            excluded.append(info)
        else:
            candidates.append(info)
    candidates.sort(key=lambda obj: obj.object_id)
    excluded.sort(key=lambda obj: obj.object_id)
    return candidates, excluded


def object_metric_payload(
    scene_object: ObjectInfo,
    raw_points_cache: dict[Path, np.ndarray],
    cues: MotionCues,
    point_sample: int,
    round_digits: int,
) -> dict[str, Any]:
    points = raw_points_cache.setdefault(scene_object.source_ply_path, load_instance_points(scene_object.source_ply_path))
    sampled_points = sample_object_points(
        points,
        point_sample,
        stable_seed(scene_object.source_ply_path, point_sample),
    )
    bbox_min = sampled_points.min(axis=0) if sampled_points.size else scene_object.bbox_min
    bbox_max = sampled_points.max(axis=0) if sampled_points.size else scene_object.bbox_max
    centroid = scene_object.centroid

    root_distance_3d = float(np.linalg.norm(centroid - cues.pelvis))
    root_distance_xz = float(np.linalg.norm(xz(centroid) - xz(cues.pelvis)))
    head_distance_3d = float(np.linalg.norm(centroid - cues.head))
    head_distance_xz = float(np.linalg.norm(xz(centroid) - xz(cues.head)))

    face_alignment = alignment_3d(centroid, cues.eye_mid, cues.face_normal_direction, cues.valid["face_normal_direction"])
    face_horizontal_alignment = horizontal_alignment(
        centroid, cues.eye_mid, cues.face_normal_direction, cues.valid["face_normal_direction"]
    )
    face_raw_alignment = alignment_3d(centroid, cues.eye_mid, cues.face_normal_raw_direction, cues.valid["face_normal_raw_direction"])
    face_line_distance_xz = horizontal_line_distance(
        centroid, cues.eye_mid, cues.face_normal_direction, cues.valid["face_normal_direction"]
    )
    root_alignment = alignment_3d(centroid, cues.pelvis, cues.root_direction, cues.valid["root_direction"])
    root_horizontal_alignment = horizontal_alignment(centroid, cues.pelvis, cues.root_direction, cues.valid["root_direction"])
    root_line_distance_xz = horizontal_line_distance(centroid, cues.pelvis, cues.root_direction, cues.valid["root_direction"])
    torso_alignment = alignment_3d(centroid, cues.pelvis, cues.torso_direction, cues.valid["torso_direction"])
    torso_horizontal_alignment = horizontal_alignment(centroid, cues.pelvis, cues.torso_direction, cues.valid["torso_direction"])
    torso_line_distance_xz = horizontal_line_distance(centroid, cues.pelvis, cues.torso_direction, cues.valid["torso_direction"])

    metrics = {
        "distance": finite_or_none(root_distance_3d, round_digits),
        "root_centroid_distance_3d": finite_or_none(root_distance_3d, round_digits),
        "root_centroid_distance_xz": finite_or_none(root_distance_xz, round_digits),
        "head_centroid_distance_3d": finite_or_none(head_distance_3d, round_digits),
        "head_centroid_distance_xz": finite_or_none(head_distance_xz, round_digits),
        "root_min_point_distance_3d": finite_or_none(min_distance(sampled_points, cues.pelvis), round_digits),
        "head_min_point_distance_3d": finite_or_none(min_distance(sampled_points, cues.head), round_digits),
        "face_normal_alignment": finite_or_none(face_alignment, round_digits),
        "face_normal_horizontal_alignment": finite_or_none(face_horizontal_alignment, round_digits),
        "face_normal_raw_alignment": finite_or_none(face_raw_alignment, round_digits),
        "face_normal_line_distance_xz": finite_or_none(face_line_distance_xz, round_digits),
        "root_alignment": finite_or_none(root_alignment, round_digits),
        "root_horizontal_alignment": finite_or_none(root_horizontal_alignment, round_digits),
        "root_line_distance_xz": finite_or_none(root_line_distance_xz, round_digits),
        "torso_alignment": finite_or_none(torso_alignment, round_digits),
        "torso_horizontal_alignment": finite_or_none(torso_horizontal_alignment, round_digits),
        "torso_line_distance_xz": finite_or_none(torso_line_distance_xz, round_digits),
        "root_torso_alignment": finite_or_none(mean_non_null([root_alignment, torso_alignment]), round_digits),
        "root_torso_horizontal_alignment": finite_or_none(
            mean_non_null([root_horizontal_alignment, torso_horizontal_alignment]),
            round_digits,
        ),
    }
    return {
        "object_id": scene_object.object_id,
        "label": scene_object.label,
        "semantic_label_id": scene_object.semantic_label_id,
        "object_idx": scene_object.object_idx,
        "source_ply_path": str(scene_object.source_ply_path),
        "raw_num_points": scene_object.raw_num_points,
        "sampled_num_points_for_metrics": int(sampled_points.shape[0]),
        "centroid": vector_payload(centroid, round_digits),
        "bbox_min": vector_payload(bbox_min, round_digits),
        "bbox_max": vector_payload(bbox_max, round_digits),
        "is_floor_wall": scene_object.is_floor_wall,
        "metrics": metrics,
    }


def human_payload(cues: MotionCues, gaze_point: np.ndarray, round_digits: int, face_orientation: str) -> dict[str, Any]:
    return {
        "last_input_frame_index": -1,
        "joint_indices": {
            "pelvis": 0,
            "neck": 12,
            "head": 15,
            "left_shoulder": 16,
            "right_shoulder": 17,
            "nose": 55,
            "right_eye": 56,
            "left_eye": 57,
        },
        "face_orientation_mode": face_orientation,
        "pelvis": vector_payload(cues.pelvis, round_digits),
        "head": vector_payload(cues.head, round_digits),
        "neck": vector_payload(cues.neck, round_digits),
        "eye_mid": vector_payload(cues.eye_mid, round_digits),
        "root_motion_direction": vector_payload(cues.root_motion_direction, round_digits),
        "root_pose_direction": vector_payload(cues.root_pose_direction, round_digits),
        "root_direction": vector_payload(cues.root_direction, round_digits),
        "torso_direction": vector_payload(cues.torso_direction, round_digits),
        "face_normal_raw_direction": vector_payload(cues.face_normal_raw_direction, round_digits),
        "face_normal_direction": vector_payload(cues.face_normal_direction, round_digits),
        "gaze_point": vector_payload(gaze_point, round_digits),
        "face_normal_ray": {
            "origin": vector_payload(cues.eye_mid, round_digits),
            "direction": vector_payload(cues.face_normal_direction, round_digits),
            "hit_point": vector_payload(gaze_point, round_digits),
        },
        "valid": cues.valid,
    }


def static_object_catalog(objects: list[ObjectInfo], round_digits: int) -> list[dict[str, Any]]:
    return [
        {
            "object_id": obj.object_id,
            "label": obj.label,
            "semantic_label_id": obj.semantic_label_id,
            "object_idx": obj.object_idx,
            "source_ply_path": str(obj.source_ply_path),
            "raw_num_points": obj.raw_num_points,
            "centroid": vector_payload(obj.centroid, round_digits),
            "bbox_min": vector_payload(obj.bbox_min, round_digits),
            "bbox_max": vector_payload(obj.bbox_max, round_digits),
            "is_floor_wall": obj.is_floor_wall,
        }
        for obj in objects
    ]


def build_scene_graphs_for_split(
    split: str,
    rows: list[DatasetRow],
    slice_root: Path,
    scene_objects_root: Path,
    output_root: Path,
    manifests: list[dict[str, Any]],
    face_orientation: str,
    point_sample: int,
    round_digits: int,
    include_floor_wall: bool,
) -> dict[str, Any]:
    split_root = slice_root / split
    joints = load_tensor(split_root / "joints_input.pth")
    poses = load_tensor(split_root / "poses_input.pth")
    gazes = load_tensor(split_root / "gazes.pth")
    if joints.shape[0] != len(rows) or poses.shape[0] != len(rows) or gazes.shape[0] != len(rows):
        raise ValueError(f"{split}: tensor lengths do not match dataset rows")

    manifest_by_index = {int(row["sample_index"]): row for row in manifests}
    scene_cache: dict[str, tuple[list[ObjectInfo], list[ObjectInfo]]] = {}
    raw_points_cache: dict[Path, np.ndarray] = {}
    split_dir = output_root / split
    by_scene_dir = split_dir / "by_scene"
    split_dir.mkdir(parents=True, exist_ok=True)
    by_scene_dir.mkdir(parents=True, exist_ok=True)

    scene_payloads: dict[str, dict[str, Any]] = {}
    sample_summaries: list[dict[str, Any]] = []
    total_objects = 0
    jsonl_path = split_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for row in rows:
            scene_objects, excluded_objects = scene_cache.setdefault(
                row.scene_name,
                load_scene_objects(scene_objects_root, row.scene_name, include_floor_wall),
            )
            sample_joints = joints[row.split_index].numpy()
            sample_poses = poses[row.split_index].numpy()
            final_cues = build_motion_cue_for_frame(sample_joints, sample_poses, sample_joints.shape[0] - 1, face_orientation)
            final_gaze_point = gazes[row.split_index, -1, 0].numpy()
            objects = [
                object_metric_payload(
                    scene_object=obj,
                    raw_points_cache=raw_points_cache,
                    cues=final_cues,
                    point_sample=point_sample,
                    round_digits=round_digits,
                )
                for obj in scene_objects
            ]
            objects.sort(key=lambda item: item["object_id"])
            manifest = manifest_by_index.get(row.split_index, {})
            target = manifest.get("target_object")
            target_object_id = None
            if target and target.get("object_idx") is not None:
                target_object_id = f"instance_{int(target['object_idx']):04d}"
            graph = {
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "sample_index": row.split_index,
                "sample_id": row.sample_id,
                "prox_sample_id": row.sequence_path,
                "scene_name": row.scene_name,
                "sequence_name": row.sequence_path,
                "start_frame": row.start_frame,
                "end_frame": row.end_frame,
                "coordinate_frame": "PROX_scene",
                "source_coordinate_frame": "PROX_scene",
                "transform": {"scale": 1.0, "transform_norm_applied": False},
                "human_final_frame": human_payload(final_cues, final_gaze_point, round_digits, face_orientation),
                "objects": objects,
                "excluded_objects": [
                    {
                        "object_id": obj.object_id,
                        "label": obj.label,
                        "object_idx": obj.object_idx,
                        "reason": "floor_wall_excluded_from_target_candidates",
                    }
                    for obj in excluded_objects
                ],
                "gt": {
                    "target_rule": manifest.get("target_rule"),
                    "target_object": target,
                    "target_object_id": target_object_id,
                    "target_object_mask": bool(manifest.get("target_object_mask", False)),
                    "interaction_labels": manifest.get("interaction_labels", []),
                    "interaction_obj_idx": manifest.get("interaction_obj_idx", []),
                    "final_frame_interactions": manifest.get("final_frame_interactions", []),
                    "sampled_frame_indices": manifest.get("sampled_frame_indices", []),
                },
            }
            jsonl.write(json.dumps(graph, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            jsonl.write("\n")

            scene_payload = scene_payloads.setdefault(
                row.scene_name,
                {
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "scene_name": row.scene_name,
                    "coordinate_frame": "PROX_scene",
                    "source_coordinate_frame": "PROX_scene",
                    "object_inventory": static_object_catalog(scene_objects, round_digits),
                    "excluded_object_inventory": static_object_catalog(excluded_objects, round_digits),
                    "sequences": [],
                },
            )
            scene_payload["sequences"].append(graph)
            sample_summaries.append(
                {
                    "sample_index": row.split_index,
                    "sample_id": row.sample_id,
                    "prox_sample_id": row.sequence_path,
                    "scene_name": row.scene_name,
                    "sequence_name": row.sequence_path,
                    "num_objects": len(objects),
                    "target_object_id": target_object_id,
                    "target_object_mask": bool(manifest.get("target_object_mask", False)),
                }
            )
            total_objects += len(objects)

    for scene_name, payload in sorted(scene_payloads.items()):
        write_json(by_scene_dir / f"{scene_name}.json", payload)

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "num_samples": len(sample_summaries),
        "num_scenes": len(scene_payloads),
        "total_object_entries": total_objects,
        "samples_jsonl": str(jsonl_path),
        "by_scene_dir": str(by_scene_dir),
        "include_floor_wall_candidates": include_floor_wall,
        "samples": sample_summaries,
    }
    write_json(split_dir / "index.json", index_payload)
    return index_payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-gimo-root", type=Path, default=Path("data/PROX/PROX_COINS_GIMO"))
    parser.add_argument("--slice-root", type=Path, default=None)
    parser.add_argument("--dataset-csv", type=Path, default=None)
    parser.add_argument("--scene-objects-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--face-orientation", choices=("none", "root", "torso"), default="torso")
    parser.add_argument("--fallback-distance", type=float, default=2.0)
    parser.add_argument("--point-sample", type=int, default=2048)
    parser.add_argument("--round-digits", type=int, default=6)
    parser.add_argument("--include-floor-wall", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prox_root = args.prox_gimo_root
    slice_root = args.slice_root or prox_root / "SLICES_888s"
    dataset_csv = args.dataset_csv or prox_root / "dataset.csv"
    scene_objects_root = args.scene_objects_root or prox_root / "scene_objects"
    output_dir = args.output_dir or prox_root / "object_selector" / "scene_graphs"
    rows_by_split = load_dataset_rows(dataset_csv)
    selected_splits = [split for split in ("train", "val", "test") if split in rows_by_split] if args.split == "all" else [args.split]

    split_summaries = []
    for split in selected_splits:
        split_root = slice_root / split
        joints = load_tensor(split_root / "joints_input.pth")
        poses = load_tensor(split_root / "poses_input.pth")
        scene_points = load_tensor(split_root / "scene_points_32768.pth")
        gazes, gaze_meta = build_gaze_tensor(
            joints=joints,
            poses=poses,
            scene_points=scene_points,
            face_orientation=args.face_orientation,
            fallback_distance=args.fallback_distance,
        )
        torch.save(gazes, split_root / "gazes.pth")
        write_json(
            split_root / "gazes_meta.json",
            {
                "split": split,
                "shape": list(gazes.shape),
                "coordinate_frame": "PROX_scene",
                "method": "face_normal_ray_nearest_forward_scene_point",
                "face_orientation": args.face_orientation,
                "fallback_distance": args.fallback_distance,
                "frames": "one ray hit per input frame",
                "samples": gaze_meta,
            },
        )

        manifest_path = split_root / "segment_manifest.json"
        manifests = json.loads(manifest_path.read_text())
        split_summary = build_scene_graphs_for_split(
            split=split,
            rows=rows_by_split[split],
            slice_root=slice_root,
            scene_objects_root=scene_objects_root,
            output_root=output_dir,
            manifests=manifests,
            face_orientation=args.face_orientation,
            point_sample=args.point_sample,
            round_digits=args.round_digits,
            include_floor_wall=args.include_floor_wall,
        )
        split_summaries.append(split_summary)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "prox_gimo_root": str(prox_root),
        "slice_root": str(slice_root),
        "scene_objects_root": str(scene_objects_root),
        "dataset_csv": str(dataset_csv),
        "output_dir": str(output_dir),
        "coordinate_frame": "PROX_scene",
        "gaze_method": "face_normal_ray_nearest_forward_scene_point",
        "face_orientation": args.face_orientation,
        "include_floor_wall_candidates": args.include_floor_wall,
        "splits": split_summaries,
        "metric_notes": {
            "distance": "Alias of root_centroid_distance_3d in PROX scene coordinates.",
            "face_normal_alignment": "3D cosine from final eye_mid to object centroid against face_normal_direction.",
            "face_normal_line_distance_xz": "Horizontal distance from object centroid to the forward face-normal ray.",
            "root_alignment": "3D cosine from final pelvis to object centroid against final SMPL-X root orientation.",
            "torso_alignment": "3D cosine from final pelvis to object centroid against shoulder/neck torso direction.",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
