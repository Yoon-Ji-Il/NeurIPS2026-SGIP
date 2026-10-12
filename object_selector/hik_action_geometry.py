"""HIK H25F50 scene graphs and selected-object point-cloud helpers.

This module is intentionally separate from the existing GIMO selector code.
It preserves the same downstream object-cache contract while reading HIK's
native per-frame object geometry directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dataset.hik_raw_1s2s import (
    BOX_TRIANGLES,
    HIK1s2sTemporalSpec,
    RawHIKIndex,
    audit_test_windows,
    hik_points_to_model,
)


SCHEMA_VERSION = "hik_action_geometry_scene_graph_v1"
NONE_OBJECT_ID = "__none__"
TRAIN_KITCHENS = frozenset(("A", "B", "C"))
TEST_KITCHENS = frozenset(("D",))
DEFAULT_CONE_COSINE = float(math.cos(math.radians(45.0)))

EVENT_TARGET_TYPE: dict[str, str | None] = {
    "walking": None,
    "sitting_down": "SITTABLE",
    "whiteboard": "WHITEBOARD",
    "sink": "SINK",
    "cupboard": "CUPBOARD",
    "coffee": "COFFEE_MACHINE",
}

# Official HIK activity2index order.
HIK_ACTIVITIES: tuple[str, ...] = (
    "carry cake",
    "carry cup",
    "carry plate",
    "carry whiteboard eraser",
    "carry whiteboard marker",
    "check water in coffee machine",
    "clean countertop",
    "clean dish",
    "close cupboard",
    "close dishwasher",
    "close drawer",
    "close fridge",
    "close window",
    "cut cake in pieces",
    "draw on whiteboard",
    "drink",
    "eat cake",
    "eat fruit",
    "empty cup in sink",
    "empty dishwasher",
    "empty ground from coffee machine",
    "empty water from coffee machine",
    "erase on whiteboard",
    "fill coffee beans",
    "fill water tank",
    "fill water to coffee machine",
    "fussball",
    "kneeling",
    "kneeling down",
    "leaning",
    "leaning down",
    "listening",
    "make coffee",
    "mark coffee",
    "open cupboard",
    "open dishwasher",
    "open drawer",
    "open fridge",
    "open window",
    "peel fruit",
    "phone call",
    "place cake on plate",
    "place cake on table",
    "place cup onto coffee machine",
    "place in dishwasher",
    "place sheet onto whiteboard",
    "place water tank in coffee machine",
    "pour kettle",
    "pour milk",
    "press coffee button",
    "put cake in fridge",
    "put cup in microwave",
    "put sugar in cup",
    "put teabag in cup",
    "put water in glass",
    "put water in kettle",
    "read paper",
    "remove sheet from whiteboard",
    "sitting",
    "sitting down",
    "squatting",
    "standing",
    "standing up",
    "start dishwasher",
    "start microwave",
    "steps",
    "take cake out of fridge",
    "take cup from coffee machine",
    "take cup out of microwave",
    "take dish out of cupboard",
    "take kettle",
    "take milk",
    "take piece of cake",
    "take teabag",
    "take water from sink",
    "take water tank from coffee machine",
    "talking",
    "throw in trash",
    "use laptop",
    "use smartphone",
    "walking",
    "washing hands",
)


@dataclass(frozen=True)
class HIKSceneObject:
    kitchen: str
    object_id: str
    label: str
    shape: str
    frame_multiplier: int
    json_path: Path
    geometry_path: Path
    geometry: np.ndarray

    def geometry_at(self, frame: int) -> np.ndarray:
        index = 0 if self.frame_multiplier == 0 else int(frame) // self.frame_multiplier
        index = min(max(index, 0), len(self.geometry) - 1)
        return np.asarray(self.geometry[index], dtype=np.float32)


def split_for_kitchen(kitchen: str) -> str:
    if kitchen in TRAIN_KITCHENS:
        return "train"
    if kitchen in TEST_KITCHENS:
        return "test"
    raise ValueError(f"Unknown HIK kitchen: {kitchen!r}")


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_id_for_event(event: Mapping[str, Any]) -> str:
    return (
        f"{event['kitchen']}|{int(event['pid'])}|{event['action']}|"
        f"{int(event['action_index'])}|{int(event['future_anchor'])}"
    )


def normalize_type(value: object) -> str:
    return str(value).replace("StaticObjectType.", "").strip().upper()


class HIKObjectCatalog:
    def __init__(self, raw_root: str | Path):
        self.raw_root = Path(raw_root).expanduser().resolve()
        self.by_kitchen: dict[str, list[HIKSceneObject]] = {}
        self.by_id: dict[tuple[str, str], HIKSceneObject] = {}
        for kitchen in "ABCD":
            objects: list[HIKSceneObject] = []
            scene_dir = self.raw_root / "scenes" / f"{kitchen}_scene"
            for json_path in sorted(scene_dir.glob("*.json")):
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                object_id = str(metadata["name"])
                geometry_path = json_path.with_suffix(".npy")
                if not geometry_path.is_file():
                    raise FileNotFoundError(geometry_path)
                scene_object = HIKSceneObject(
                    kitchen=kitchen,
                    object_id=object_id,
                    label=normalize_type(metadata["type"]),
                    shape=str(metadata.get("shape", "box")).lower(),
                    frame_multiplier=int(metadata.get("frame_multiplier", 0)),
                    json_path=json_path,
                    geometry_path=geometry_path,
                    geometry=np.load(geometry_path, mmap_mode="r"),
                )
                objects.append(scene_object)
                self.by_id[(kitchen, object_id)] = scene_object
            self.by_kitchen[kitchen] = objects

    def get(self, kitchen: str, object_id: str) -> HIKSceneObject | None:
        return self.by_id.get((kitchen, object_id))


def _unit(vector: np.ndarray) -> tuple[np.ndarray, bool]:
    vector = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8 or not math.isfinite(norm):
        return np.zeros(3, dtype=np.float32), False
    return (vector / norm).astype(np.float32), True


def _orient(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if float(np.dot(direction, reference)) < 0.0:
        return -direction
    return direction


def _horizontal_unit(vector: np.ndarray) -> tuple[np.ndarray, bool]:
    horizontal = np.asarray([vector[0], 0.0, vector[2]], dtype=np.float32)
    return _unit(horizontal)


def _alignment(direction: np.ndarray, target: np.ndarray) -> float:
    a, valid_a = _unit(direction)
    b, valid_b = _unit(target)
    if not valid_a or not valid_b:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _horizontal_alignment(direction: np.ndarray, target: np.ndarray) -> float:
    a, valid_a = _horizontal_unit(direction)
    b, valid_b = _horizontal_unit(target)
    if not valid_a or not valid_b:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _bbox_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float32)
    delta = np.maximum(np.maximum(bbox_min - point, point - bbox_max), 0.0)
    return float(np.linalg.norm(delta))


def _ray_line_distance_xz(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    d, valid = _horizontal_unit(direction)
    if not valid:
        return float(np.linalg.norm((point - origin)[[0, 2]]))
    offset = point - origin
    forward = max(float(np.dot(offset, d)), 0.0)
    closest = origin + forward * d
    return float(np.linalg.norm((point - closest)[[0, 2]]))


def _geometry_model_points(
    scene_object: HIKSceneObject,
    frame: int,
    scene_origin: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    geometry = scene_object.geometry_at(frame)
    if geometry.ndim == 2 and geometry.shape[-1] == 3:
        points = hik_points_to_model(geometry) - scene_origin[None, :]
        return points.astype(np.float32), True, 0.0
    flat = geometry.reshape(-1)
    if flat.size < 4:
        raise ValueError(
            f"Unsupported HIK object geometry {scene_object.geometry_path}: "
            f"{geometry.shape}"
        )
    center = hik_points_to_model(flat[:3]) - scene_origin
    radius = float(flat[3])
    return center.reshape(1, 3).astype(np.float32), False, radius


def _bbox_for_geometry(
    points: np.ndarray,
    is_box: bool,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if is_box:
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        centroid = points.mean(axis=0)
    else:
        centroid = points[0]
        bbox_min = centroid - radius
        bbox_max = centroid + radius
    return (
        centroid.astype(np.float32),
        bbox_min.astype(np.float32),
        bbox_max.astype(np.float32),
    )


def _sample_box_surface(
    corners: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    triangles = corners[BOX_TRIANGLES]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 1.0e-10:
        indices = rng.choice(len(corners), size=num_points, replace=True)
        return corners[indices].astype(np.float32)
    triangle_indices = rng.choice(
        len(triangles),
        size=num_points,
        replace=True,
        p=areas / areas.sum(),
    )
    selected = triangles[triangle_indices]
    u = rng.random(num_points)
    v = rng.random(num_points)
    reflect = u + v > 1.0
    u[reflect] = 1.0 - u[reflect]
    v[reflect] = 1.0 - v[reflect]
    sampled = (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    )
    return sampled.astype(np.float32)


def sample_object_surface_points(
    scene_object: HIKSceneObject,
    frame: int,
    scene_origin: np.ndarray,
    num_points: int,
    seed: int,
) -> np.ndarray:
    points, is_box, radius = _geometry_model_points(
        scene_object,
        frame,
        np.asarray(scene_origin, dtype=np.float32).reshape(3),
    )
    rng = np.random.default_rng(seed)
    if is_box:
        return _sample_box_surface(points, int(num_points), rng)
    directions = rng.normal(size=(int(num_points), 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1.0e-8)
    return (points[0][None, :] + radius * directions).astype(np.float32)


def _human_cues(model_poses: np.ndarray) -> dict[str, Any]:
    if model_poses.ndim != 3 or model_poses.shape[1:] != (29, 3):
        raise ValueError(f"Expected [T,29,3] HIK poses, got {model_poses.shape}")
    torso_directions: list[np.ndarray] = []
    face_directions: list[np.ndarray] = []
    root_velocities: list[np.ndarray] = []
    hand_positions: list[np.ndarray] = []
    previous_torso = np.zeros(3, dtype=np.float32)
    sequence_motion = model_poses[-1, 0] - model_poses[0, 0]

    for index, joints in enumerate(model_poses):
        previous = model_poses[max(index - 1, 0)]
        root_velocity = joints[0] - previous[0]
        root_reference = root_velocity
        if float(np.linalg.norm(root_reference)) <= 1.0e-8:
            root_reference = sequence_motion
        if float(np.linalg.norm(root_reference)) <= 1.0e-8:
            root_reference = previous_torso

        shoulder_axis = joints[17] - joints[16]
        torso, torso_valid = _unit(np.cross(shoulder_axis, joints[12] - joints[0]))
        if torso_valid:
            torso = _orient(torso, root_reference)
            previous_torso = torso
        else:
            torso = previous_torso.copy()

        eye_mid = 0.5 * (joints[25] + joints[26])
        face, face_valid = _unit(
            np.cross(joints[24] - eye_mid, joints[26] - joints[25])
        )
        if face_valid:
            face = _orient(face, torso)
        else:
            face = torso.copy()

        torso_directions.append(torso)
        face_directions.append(face)
        root_velocities.append(root_velocity.astype(np.float32))
        hand_positions.append((0.5 * (joints[20] + joints[21])).astype(np.float32))

    last = model_poses[-1]
    root_direction, root_valid = _unit(sequence_motion)
    torso_direction = torso_directions[-1]
    face_direction = face_directions[-1]
    eye_mid = 0.5 * (last[25] + last[26])
    return {
        "pelvis": last[0],
        "neck": last[12],
        "head": last[15],
        "eye_mid": eye_mid.astype(np.float32),
        "hands_mid": hand_positions[-1],
        "root_direction": root_direction,
        "torso_direction": torso_direction,
        "face_normal_direction": face_direction,
        "torso_directions": np.stack(torso_directions),
        "face_directions": np.stack(face_directions),
        "root_velocities": np.stack(root_velocities),
        "hand_positions": np.stack(hand_positions),
        "valid": {
            "root_direction": bool(root_valid),
            "torso_direction": bool(np.linalg.norm(torso_direction) > 1.0e-8),
            "face_normal_direction": bool(np.linalg.norm(face_direction) > 1.0e-8),
        },
    }


def _cone_fraction(
    directions: np.ndarray,
    origins: np.ndarray,
    centroid: np.ndarray,
    threshold: float,
) -> float:
    values = []
    for direction, origin in zip(directions, origins, strict=True):
        values.append(
            float(_horizontal_alignment(direction, centroid - origin) >= threshold)
        )
    return float(np.mean(values)) if values else 0.0


def _geometry_score(metrics: Mapping[str, float]) -> float:
    distance_score = math.exp(-max(float(metrics["root_bbox_distance_3d"]), 0.0) / 1.5)
    hand_score = math.exp(-max(float(metrics["hand_bbox_distance_3d"]), 0.0) / 1.0)
    cone_score = (
        0.60 * float(metrics["torso_cone_fraction"])
        + 0.20 * float(metrics["face_cone_fraction"])
        + 0.20 * float(metrics["root_heading_cone_fraction"])
    )
    alignment_score = np.mean(
        [
            0.5 * (float(metrics["root_horizontal_alignment"]) + 1.0),
            0.5 * (float(metrics["torso_horizontal_alignment"]) + 1.0),
            0.5 * (float(metrics["face_horizontal_alignment"]) + 1.0),
        ]
    )
    return float(
        np.clip(
            0.35 * distance_score
            + 0.15 * hand_score
            + 0.30 * cone_score
            + 0.20 * alignment_score,
            0.0,
            1.0,
        )
    )


def _round_vector(value: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(component), digits) for component in value.tolist()]


def _object_payload(
    scene_object: HIKSceneObject,
    frame: int,
    scene_origin: np.ndarray,
    model_poses: np.ndarray,
    cues: Mapping[str, Any],
    cone_cosine: float,
) -> dict[str, Any]:
    points, is_box, radius = _geometry_model_points(
        scene_object,
        frame,
        scene_origin,
    )
    centroid, bbox_min, bbox_max = _bbox_for_geometry(points, is_box, radius)
    pelvis = np.asarray(cues["pelvis"])
    head = np.asarray(cues["head"])
    eye_mid = np.asarray(cues["eye_mid"])
    hands_mid = np.asarray(cues["hands_mid"])
    target_from_root = centroid - pelvis
    target_from_face = centroid - eye_mid
    target_from_hand = centroid - hands_mid
    hand_origins = np.asarray(cues["hand_positions"])

    metrics: dict[str, float] = {
        "distance": float(np.linalg.norm(target_from_root)),
        "root_centroid_distance_3d": float(np.linalg.norm(target_from_root)),
        "root_centroid_distance_xz": float(np.linalg.norm(target_from_root[[0, 2]])),
        "head_centroid_distance_3d": float(np.linalg.norm(centroid - head)),
        "root_bbox_distance_3d": _bbox_distance(pelvis, bbox_min, bbox_max),
        "head_bbox_distance_3d": _bbox_distance(head, bbox_min, bbox_max),
        "hand_bbox_distance_3d": _bbox_distance(hands_mid, bbox_min, bbox_max),
        "root_alignment": _alignment(cues["root_direction"], target_from_root),
        "root_horizontal_alignment": _horizontal_alignment(
            cues["root_direction"], target_from_root
        ),
        "torso_alignment": _alignment(cues["torso_direction"], target_from_root),
        "torso_horizontal_alignment": _horizontal_alignment(
            cues["torso_direction"], target_from_root
        ),
        "face_alignment": _alignment(cues["face_normal_direction"], target_from_face),
        "face_horizontal_alignment": _horizontal_alignment(
            cues["face_normal_direction"], target_from_face
        ),
        "hand_alignment": _alignment(
            hand_origins[-1] - hand_origins[-2]
            if len(hand_origins) > 1
            else np.zeros(3, dtype=np.float32),
            target_from_hand,
        ),
        "root_line_distance_xz": _ray_line_distance_xz(
            centroid, pelvis, cues["root_direction"]
        ),
        "torso_line_distance_xz": _ray_line_distance_xz(
            centroid, pelvis, cues["torso_direction"]
        ),
        "face_line_distance_xz": _ray_line_distance_xz(
            centroid, eye_mid, cues["face_normal_direction"]
        ),
        "torso_cone_fraction": _cone_fraction(
            np.asarray(cues["torso_directions"]),
            model_poses[:, 0],
            centroid,
            cone_cosine,
        ),
        "face_cone_fraction": _cone_fraction(
            np.asarray(cues["face_directions"]),
            0.5 * (model_poses[:, 25] + model_poses[:, 26]),
            centroid,
            cone_cosine,
        ),
        "root_heading_cone_fraction": _cone_fraction(
            np.asarray(cues["root_velocities"]),
            model_poses[:, 0],
            centroid,
            cone_cosine,
        ),
    }
    metrics["geometry_score"] = _geometry_score(metrics)
    return {
        "object_id": scene_object.object_id,
        "label": scene_object.label,
        "shape": scene_object.shape,
        "frame_multiplier": scene_object.frame_multiplier,
        "centroid": _round_vector(centroid),
        "bbox_min": _round_vector(bbox_min),
        "bbox_max": _round_vector(bbox_max),
        "metrics": {
            key: round(float(value), 6)
            for key, value in metrics.items()
        },
    }


def _activity_payload(act: np.ndarray) -> dict[str, Any]:
    active = np.asarray(act) > 0.5
    frequency = active.mean(axis=0)
    observed = [
        {
            "name": HIK_ACTIVITIES[index],
            "frame_fraction": round(float(frequency[index]), 4),
        }
        for index in np.flatnonzero(frequency > 0.0)
    ]
    last = [
        HIK_ACTIVITIES[index]
        for index in np.flatnonzero(active[-1])
    ]
    return {
        "active_actions_last_observed": last,
        "active_actions_observed": observed,
    }


def _candidate_filter(
    objects: list[dict[str, Any]],
    event_action: str,
    radius: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    spatial = [
        row
        for row in objects
        if float(row["metrics"]["root_bbox_distance_3d"]) <= float(radius)
    ]
    spatial.sort(
        key=lambda row: (
            -float(row["metrics"]["geometry_score"]),
            float(row["metrics"]["root_bbox_distance_3d"]),
            row["object_id"],
        )
    )
    selected: dict[str, dict[str, Any]] = {}

    for row in spatial:
        metrics = row["metrics"]
        if max(
            float(metrics["torso_cone_fraction"]),
            float(metrics["face_cone_fraction"]),
            float(metrics["root_heading_cone_fraction"]),
        ) > 0.0:
            copied = dict(row)
            copied["candidate_reason"] = "cone"
            selected[row["object_id"]] = copied

    # Geometry-only safety net when the torso direction is noisy or stationary.
    for row in sorted(
        spatial,
        key=lambda item: (
            float(item["metrics"]["root_bbox_distance_3d"]),
            item["object_id"],
        ),
    )[:3]:
        if row["object_id"] not in selected:
            copied = dict(row)
            copied["candidate_reason"] = "nearest_geometry_fallback"
            selected[row["object_id"]] = copied

    # Do not select an instance here. This only prevents all objects of the
    # event's coarse semantic family from being accidentally removed by cone
    # noise or the radius crop; Qwen still chooses the final instance. HIK's
    # official action anchor can occur while a person is still approaching
    # the target, so the compatible family must be rescued from the complete
    # kitchen catalog rather than only from the current-radius subset.
    expected_type = EVENT_TARGET_TYPE.get(event_action)
    if expected_type is not None:
        for row in objects:
            if row["label"] == expected_type and row["object_id"] not in selected:
                copied = dict(row)
                copied["candidate_reason"] = "semantic_family_rescue"
                selected[row["object_id"]] = copied

    candidates = list(selected.values())
    candidates.sort(
        key=lambda row: (
            -float(row["metrics"]["geometry_score"]),
            float(row["metrics"]["root_bbox_distance_3d"]),
            row["object_id"],
        )
    )
    if len(candidates) > int(max_candidates):
        semantic = [
            row for row in candidates
            if row.get("label") == expected_type
        ]
        semantic_ids = {row["object_id"] for row in semantic}
        keep = semantic[: int(max_candidates)]
        for row in candidates:
            if len(keep) >= int(max_candidates):
                break
            if row["object_id"] not in semantic_ids:
                keep.append(row)
        candidates = keep
    return candidates


def build_event_scene_graph(
    index: RawHIKIndex,
    catalog: HIKObjectCatalog,
    event: Mapping[str, Any],
    *,
    scene_radius: float = 3.5,
    cone_cosine: float = DEFAULT_CONE_COSINE,
    max_candidates: int = 24,
) -> dict[str, Any]:
    input_frames = tuple(int(frame) for frame in event["input_frames"])
    track = index.find_complete_track(
        str(event["kitchen"]),
        int(event["pid"]),
        event["frames"],
    )
    if track is None:
        raise KeyError(f"Complete H25F50 track disappeared for {event}")
    rows = index.load_rows(track, input_frames)
    raw_poses = np.asarray(rows["poses3d"], dtype=np.float32)
    absolute_model_poses = hik_points_to_model(raw_poses)
    scene_origin = absolute_model_poses[-1, 0].copy()
    model_poses = absolute_model_poses - scene_origin[None, None, :]
    cues = _human_cues(model_poses)
    last_observed_frame = input_frames[-1]

    object_rows = [
        _object_payload(
            scene_object,
            last_observed_frame,
            scene_origin,
            model_poses,
            cues,
            cone_cosine,
        )
        for scene_object in catalog.by_kitchen[str(event["kitchen"])]
    ]
    candidates = _candidate_filter(
        object_rows,
        str(event["action"]),
        scene_radius,
        max_candidates,
    )
    if str(event["action"]) == "walking":
        candidates.append(
            {
                "object_id": NONE_OBJECT_ID,
                "label": "NONE",
                "shape": "none",
                "centroid": [0.0, 0.0, 0.0],
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [0.0, 0.0, 0.0],
                "candidate_reason": "no_target_object_for_walking",
                "metrics": {
                    "geometry_score": 1.0,
                    "root_bbox_distance_3d": 0.0,
                    "torso_cone_fraction": 0.0,
                    "face_cone_fraction": 0.0,
                    "root_heading_cone_fraction": 0.0,
                },
            }
        )

    activity = _activity_payload(rows["act"])
    sample_id = sample_id_for_event(event)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": split_for_kitchen(str(event["kitchen"])),
        "kitchen": str(event["kitchen"]),
        "pid": int(event["pid"]),
        "event_action": str(event["action"]),
        "event_action_index": int(event["action_index"]),
        "expected_object_type_hint": EVENT_TARGET_TYPE.get(str(event["action"])),
        "future_anchor": int(event["future_anchor"]),
        "input_frames": list(input_frames),
        "future_frames": [int(frame) for frame in event["future_frames"]],
        "last_observed_frame": int(last_observed_frame),
        "track": str(track.path),
        "coordinate_frame": "target_root_centered_model_y_up",
        "scene_origin_model_absolute": _round_vector(scene_origin),
        "human_final_frame": {
            "joint_indices": {
                "pelvis": 0,
                "neck": 12,
                "head": 15,
                "left_shoulder": 16,
                "right_shoulder": 17,
                "left_wrist": 20,
                "right_wrist": 21,
                "nose": 24,
                "left_eye": 25,
                "right_eye": 26,
            },
            "pelvis": _round_vector(cues["pelvis"]),
            "head": _round_vector(cues["head"]),
            "eye_mid": _round_vector(cues["eye_mid"]),
            "hands_mid": _round_vector(cues["hands_mid"]),
            "root_direction": _round_vector(cues["root_direction"]),
            "torso_direction": _round_vector(cues["torso_direction"]),
            "face_normal_direction": _round_vector(cues["face_normal_direction"]),
            "valid": dict(cues["valid"]),
        },
        **activity,
        "candidate_policy": {
            "scene_radius_m": float(scene_radius),
            "cone_half_angle_degrees": round(
                math.degrees(math.acos(float(cone_cosine))),
                4,
            ),
            "max_candidates": int(max_candidates),
            "semantic_hint_selects_instance": False,
        },
        "scene_object_count": len(object_rows),
        "candidate_count": len(candidates),
        "objects": candidates,
    }


def load_valid_events(
    raw_root: str | Path,
) -> tuple[RawHIKIndex, list[dict[str, Any]], dict[str, Any]]:
    index = RawHIKIndex(raw_root)
    manifest, audit = audit_test_windows(index, HIK1s2sTemporalSpec())
    return index, manifest, audit


def write_graphs(
    *,
    raw_root: str | Path,
    output_root: str | Path,
    split: str = "all",
    action: str = "",
    limit: int = -1,
    scene_radius: float = 3.5,
    cone_cosine: float = DEFAULT_CONE_COSINE,
    max_candidates: int = 24,
) -> dict[str, Any]:
    if split not in {"train", "test", "all"}:
        raise ValueError(split)
    index, events, audit = load_valid_events(raw_root)
    catalog = HIKObjectCatalog(raw_root)
    filtered = [
        event
        for event in events
        if (split == "all" or split_for_kitchen(event["kitchen"]) == split)
        and (not action or event["action"] == action)
    ]
    if limit >= 0:
        filtered = filtered[:limit]

    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    counts = Counter()
    action_counts = Counter()
    candidate_counts = Counter()
    try:
        for split_name in ("train", "test"):
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            handles[split_name] = (split_dir / "samples.jsonl").open(
                "w",
                encoding="utf-8",
            )
        for event in filtered:
            graph = build_event_scene_graph(
                index,
                catalog,
                event,
                scene_radius=scene_radius,
                cone_cosine=cone_cosine,
                max_candidates=max_candidates,
            )
            split_name = graph["split"]
            handles[split_name].write(
                json.dumps(graph, ensure_ascii=False, allow_nan=False) + "\n"
            )
            counts[split_name] += 1
            action_counts[(split_name, graph["event_action"])] += 1
            candidate_counts[graph["candidate_count"]] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_root": str(Path(raw_root).expanduser().resolve()),
        "output_root": str(output_path),
        "official_audit": audit,
        "filters": {
            "split": split,
            "action": action,
            "limit": limit,
        },
        "counts": dict(counts),
        "action_counts": {
            split_name: {
                action_name: action_counts[(split_name, action_name)]
                for action_name in EVENT_TARGET_TYPE
            }
            for split_name in ("train", "test")
        },
        "candidate_count_histogram": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "other_people_policy": (
            "non-target people are intentionally excluded from the LLM graph; "
            "they remain part of the separate 1000-point motion scene cache"
        ),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc


def load_selection_lookup(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in iter_jsonl(path)
        if row.get("sample_id")
    }


__all__ = [
    "DEFAULT_CONE_COSINE",
    "EVENT_TARGET_TYPE",
    "HIKObjectCatalog",
    "HIKSceneObject",
    "HIK_ACTIVITIES",
    "NONE_OBJECT_ID",
    "SCHEMA_VERSION",
    "build_event_scene_graph",
    "iter_jsonl",
    "load_selection_lookup",
    "load_valid_events",
    "sample_object_surface_points",
    "split_for_kitchen",
    "stable_seed",
    "write_graphs",
]
