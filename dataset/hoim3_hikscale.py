"""Observed-only HOI-M3 HIK-scale utilities for the direct-24 experiment.

The module intentionally mirrors the native HIK H25F50 preprocessing rules:

* keep the dataset's native FPS (HOI-M3 is 30 FPS);
* use 30 observed and 60 future frames;
* center every spatial tensor at the target root in the last observed frame;
* construct the scene at the last observed frame;
* preserve nearby non-target people as sparse last-frame joint point sets;
* never use future interaction annotations for student inputs or geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL = "hoim3-entity-interaction-hikscale-v1"
INPUT_FRAMES = 30
FUTURE_FRAMES = 60
TOTAL_FRAMES = INPUT_FRAMES + FUTURE_FRAMES
FPS = 30
N_JOINTS = 24
SCENE_POINTS = 1000
OBJECT_POINTS = 1024
SCENE_RADIUS_M = 3.5
OTHER_PERSON_RADIUS_M = 2.5
MAX_OTHER_PEOPLE = 4
MAX_PEOPLE_POINTS = MAX_OTHER_PEOPLE * N_JOINTS
DEFAULT_CONE_COSINE = float(math.cos(math.radians(45.0)))


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_key(record: Mapping[str, Any]) -> str:
    return (
        f"{record['sequence']}|p{int(record['target_person_index'])}|"
        f"s{int(record['start_frame'])}"
    )


def read_manifest(
    preprocess_root: str | Path,
    split: str,
    protocol: str = PROTOCOL,
) -> list[dict[str, Any]]:
    path = (
        Path(preprocess_root)
        / "manifests"
        / protocol
        / f"{split}.jsonl.gz"
    )
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    expected = 3332 if split == "train" else 689
    if protocol == PROTOCOL and len(rows) != expected:
        raise ValueError(
            f"{path}: expected {expected} {split} samples, got {len(rows)}"
        )
    return rows


def sample_key_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(sample_key(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class SequenceArrays:
    sequence: str
    person_ids: list[int]
    joints_by_person: dict[int, np.ndarray]
    presence_by_person: dict[int, np.ndarray]
    object_rotations: np.ndarray
    object_translations: np.ndarray
    object_presence: np.ndarray
    object_names: list[str]
    root_orientations: np.ndarray
    smpl_presence: np.ndarray


class HOIM3Store:
    """Lazy read-only access to the 12 HIK-scale canonical sequences."""

    def __init__(self, preprocess_root: str | Path):
        self.root = Path(preprocess_root).expanduser().resolve()
        self.canonical_root = self.root / "A-FRAME_DSratio=2"
        self.object_root = self.root / "entity_object_points" / "N=1024"
        self._sequences: dict[str, SequenceArrays] = {}
        self._object_points: dict[str, np.ndarray] = {}

    def sequence(self, name: str) -> SequenceArrays:
        if name in self._sequences:
            return self._sequences[name]
        directory = self.canonical_root / name
        joints = np.load(
            directory / "ip_2_iF2Joints.npy",
            allow_pickle=True,
        ).item()
        presence = np.load(
            directory / "ip_2_iF2Present.npy",
            allow_pickle=True,
        ).item()
        person_ids = [
            int(value)
            for value in json.loads(
                (directory / "person_ids.json").read_text(encoding="utf-8")
            )
        ]
        smpl = np.load(
            directory / "smpl_params_30fps.npz",
            allow_pickle=False,
        )
        arrays = SequenceArrays(
            sequence=name,
            person_ids=person_ids,
            joints_by_person={
                local_index: np.asarray(joints[person_id], dtype=np.float32)
                for local_index, person_id in enumerate(person_ids)
            },
            presence_by_person={
                local_index: np.asarray(presence[person_id], dtype=bool)
                for local_index, person_id in enumerate(person_ids)
            },
            object_rotations=np.load(
                directory / "iFrame_2_objsRs.npy",
                mmap_mode="r",
            ),
            object_translations=np.load(
                directory / "iFrame_2_objsTs.npy",
                mmap_mode="r",
            ),
            object_presence=np.load(
                directory / "iFrame_2_objsPresent.npy",
                mmap_mode="r",
            ),
            object_names=json.loads(
                (directory / "object_names.json").read_text(encoding="utf-8")
            ),
            root_orientations=np.asarray(smpl["Rh"], dtype=np.float32),
            smpl_presence=np.asarray(smpl["presence"], dtype=bool),
        )
        self._sequences[name] = arrays
        return arrays

    def local_object_points(self, object_name: str) -> np.ndarray:
        if object_name not in self._object_points:
            path = self.object_root / f"{object_name}.npy"
            points = np.load(path).astype(np.float32, copy=False)
            if points.shape != (OBJECT_POINTS, 3):
                raise ValueError(
                    f"{path}: expected {(OBJECT_POINTS, 3)}, got {points.shape}"
                )
            self._object_points[object_name] = points
        return self._object_points[object_name]

    def object_world_points(
        self,
        arrays: SequenceArrays,
        frame: int,
        object_index: int,
    ) -> np.ndarray:
        local = self.local_object_points(arrays.object_names[object_index])
        rotation = np.asarray(
            arrays.object_rotations[frame, object_index],
            dtype=np.float32,
        )
        translation = np.asarray(
            arrays.object_translations[frame, object_index],
            dtype=np.float32,
        )
        return (local @ rotation + translation).astype(np.float32, copy=False)


def _unit(vector: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-8 or not math.isfinite(norm):
        return np.zeros(3, dtype=np.float32), False
    return (value / norm).astype(np.float32), True


def _orient(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if float(np.dot(direction, reference)) < 0.0:
        return -direction
    return direction


def _horizontal_unit(vector: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray([vector[0], 0.0, vector[2]], dtype=np.float32)
    return _unit(value)


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


def _bbox_distance(
    point: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> float:
    delta = np.maximum(
        np.maximum(bbox_min - point, point - bbox_max),
        0.0,
    )
    return float(np.linalg.norm(delta))


def _ray_line_distance_xz(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    horizontal, valid = _horizontal_unit(direction)
    if not valid:
        return float(np.linalg.norm((point - origin)[[0, 2]]))
    offset = point - origin
    forward = max(float(np.dot(offset, horizontal)), 0.0)
    closest = origin + forward * horizontal
    return float(np.linalg.norm((point - closest)[[0, 2]]))


def _cone_fraction(
    directions: np.ndarray,
    origins: np.ndarray,
    centroid: np.ndarray,
    threshold: float,
) -> float:
    values = [
        float(_horizontal_alignment(direction, centroid - origin) >= threshold)
        for direction, origin in zip(directions, origins, strict=True)
    ]
    return float(np.mean(values)) if values else 0.0


def human_cues(joints_history: np.ndarray) -> dict[str, Any]:
    """Build the HIK torso/cone cues using the native SMPL 24-joint layout."""

    joints_history = np.asarray(joints_history, dtype=np.float32)
    if joints_history.ndim != 3 or joints_history.shape[1:] != (N_JOINTS, 3):
        raise ValueError(
            f"Expected [T,{N_JOINTS},3], got {joints_history.shape}"
        )
    torso_directions = []
    root_velocities = []
    hand_positions = []
    previous_torso = np.zeros(3, dtype=np.float32)
    sequence_motion = joints_history[-1, 0] - joints_history[0, 0]
    for index, joints in enumerate(joints_history):
        previous = joints_history[max(index - 1, 0)]
        root_velocity = joints[0] - previous[0]
        reference = root_velocity
        if float(np.linalg.norm(reference)) <= 1.0e-8:
            reference = sequence_motion
        if float(np.linalg.norm(reference)) <= 1.0e-8:
            reference = previous_torso
        shoulder_axis = joints[17] - joints[16]
        torso, valid = _unit(np.cross(shoulder_axis, joints[12] - joints[0]))
        if valid:
            torso = _orient(torso, reference)
            previous_torso = torso
        else:
            torso = previous_torso.copy()
        torso_directions.append(torso)
        root_velocities.append(root_velocity.astype(np.float32))
        hand_positions.append(
            (0.5 * (joints[20] + joints[21])).astype(np.float32)
        )
    last = joints_history[-1]
    root_direction, root_valid = _unit(sequence_motion)
    torso_direction = torso_directions[-1]
    return {
        "pelvis": last[0],
        "neck": last[12],
        "head": last[15],
        "eye_mid": last[15],
        "hands_mid": hand_positions[-1],
        "root_direction": root_direction,
        "torso_direction": torso_direction,
        "face_normal_direction": torso_direction,
        "torso_directions": np.stack(torso_directions),
        "face_directions": np.stack(torso_directions),
        "root_velocities": np.stack(root_velocities),
        "hand_positions": np.stack(hand_positions),
        "valid": {
            "root_direction": bool(root_valid),
            "torso_direction": bool(np.linalg.norm(torso_direction) > 1.0e-8),
            "face_normal_direction": bool(
                np.linalg.norm(torso_direction) > 1.0e-8
            ),
        },
    }


def _geometry_score(metrics: Mapping[str, float]) -> float:
    """The same distance/hand/cone/alignment rubric used by HIK."""

    distance_score = math.exp(
        -max(float(metrics["root_bbox_distance_3d"]), 0.0) / 1.5
    )
    hand_score = math.exp(
        -max(float(metrics["hand_bbox_distance_3d"]), 0.0) / 1.0
    )
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


def _geometry_score_no_cone(metrics: Mapping[str, float]) -> float:
    """Observed geometry ranking that never uses cone membership."""

    distance_score = math.exp(
        -max(float(metrics["root_bbox_distance_3d"]), 0.0) / 1.5
    )
    hand_score = math.exp(
        -max(float(metrics["hand_bbox_distance_3d"]), 0.0) / 1.0
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
            0.45 * distance_score
            + 0.30 * hand_score
            + 0.25 * alignment_score,
            0.0,
            1.0,
        )
    )


def object_geometry_payload(
    *,
    object_index: int,
    object_name: str,
    points: np.ndarray,
    joints_history: np.ndarray,
    cues: Mapping[str, Any],
    cone_cosine: float = DEFAULT_CONE_COSINE,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    centroid = points.mean(axis=0)
    pelvis = np.asarray(cues["pelvis"])
    head = np.asarray(cues["head"])
    eye_mid = np.asarray(cues["eye_mid"])
    hands_mid = np.asarray(cues["hands_mid"])
    hand_origins = np.asarray(cues["hand_positions"])
    metrics: dict[str, float] = {
        "distance": float(np.linalg.norm(centroid - pelvis)),
        "root_centroid_distance_3d": float(np.linalg.norm(centroid - pelvis)),
        "root_centroid_distance_xz": float(
            np.linalg.norm((centroid - pelvis)[[0, 2]])
        ),
        "head_centroid_distance_3d": float(np.linalg.norm(centroid - head)),
        "root_bbox_distance_3d": _bbox_distance(
            pelvis, bbox_min, bbox_max
        ),
        "head_bbox_distance_3d": _bbox_distance(head, bbox_min, bbox_max),
        "hand_bbox_distance_3d": _bbox_distance(
            hands_mid, bbox_min, bbox_max
        ),
        "root_alignment": _alignment(
            cues["root_direction"], centroid - pelvis
        ),
        "root_horizontal_alignment": _horizontal_alignment(
            cues["root_direction"], centroid - pelvis
        ),
        "torso_alignment": _alignment(
            cues["torso_direction"], centroid - pelvis
        ),
        "torso_horizontal_alignment": _horizontal_alignment(
            cues["torso_direction"], centroid - pelvis
        ),
        "face_alignment": _alignment(
            cues["face_normal_direction"], centroid - eye_mid
        ),
        "face_horizontal_alignment": _horizontal_alignment(
            cues["face_normal_direction"], centroid - eye_mid
        ),
        "hand_alignment": _alignment(
            hand_origins[-1] - hand_origins[-2],
            centroid - hands_mid,
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
            joints_history[:, 0],
            centroid,
            cone_cosine,
        ),
        "face_cone_fraction": _cone_fraction(
            np.asarray(cues["face_directions"]),
            joints_history[:, 15],
            centroid,
            cone_cosine,
        ),
        "root_heading_cone_fraction": _cone_fraction(
            np.asarray(cues["root_velocities"]),
            joints_history[:, 0],
            centroid,
            cone_cosine,
        ),
    }
    metrics["geometry_score"] = _geometry_score(metrics)
    metrics["geometry_score_no_cone"] = _geometry_score_no_cone(metrics)
    return {
        "object_id": f"o{int(object_index)}",
        "object_index": int(object_index),
        "name": object_name,
        "label": object_name,
        "shape": "mesh_point_cloud",
        "centroid": [round(float(value), 6) for value in centroid],
        "bbox_min": [round(float(value), 6) for value in bbox_min],
        "bbox_max": [round(float(value), 6) for value in bbox_max],
        "metrics": {
            key: round(float(value), 6)
            for key, value in metrics.items()
        },
    }


def filter_candidates(
    objects: list[dict[str, Any]],
    *,
    radius: float = SCENE_RADIUS_M,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    del radius
    candidates = []
    for row in objects:
        copied = dict(row)
        copied["candidate_reason"] = "all_scene_objects"
        candidates.append(copied)
    candidates.sort(
        key=lambda row: (
            -float(row["metrics"]["geometry_score_no_cone"]),
            float(row["metrics"]["root_bbox_distance_3d"]),
            row["object_id"],
        )
    )
    return candidates[: int(max_candidates)]


def merge_environment_and_people(
    environment_points: np.ndarray,
    people_points: np.ndarray,
    *,
    total_points: int,
    max_people_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Exact HIK fixed-size merge rule with a HOI-M3 24-joint budget."""

    environment = np.asarray(environment_points, dtype=np.float32).reshape(-1, 3)
    people = np.asarray(people_points, dtype=np.float32).reshape(-1, 3)
    max_people_points = min(int(max_people_points), int(total_points) - 1)
    if len(people) > max_people_points:
        keep = rng.choice(
            len(people),
            size=max_people_points,
            replace=False,
        )
        people = people[keep]
    environment_budget = int(total_points) - len(people)
    if not len(environment):
        raise RuntimeError("No observed object points remained after scene crop")
    keep = rng.choice(
        len(environment),
        size=environment_budget,
        replace=len(environment) < environment_budget,
    )
    scene = np.concatenate([environment[keep], people], axis=0)
    scene = scene[rng.permutation(len(scene))]
    return scene.astype(np.float32, copy=False), {
        "total_points": int(len(scene)),
        "environment_points": int(environment_budget),
        "people_points": int(len(people)),
    }


def build_observed_sample(
    store: HOIM3Store,
    record: Mapping[str, Any],
    *,
    scene_points: int = SCENE_POINTS,
    scene_radius: float = SCENE_RADIUS_M,
    other_person_radius: float = OTHER_PERSON_RADIUS_M,
    max_other_people: int = MAX_OTHER_PEOPLE,
    max_candidates: int = 24,
) -> dict[str, Any]:
    """Build direct joints, HIK-style scene, and observed geometry candidates."""

    sequence = str(record["sequence"])
    arrays = store.sequence(sequence)
    target_index = int(record["target_person_index"])
    start = int(record["start_frame"])
    last_observed = start + INPUT_FRAMES - 1
    end = start + TOTAL_FRAMES
    if int(record["last_observed_frame"]) != last_observed:
        raise ValueError(f"{sample_key(record)}: last-observed mismatch")
    if int(record["end_frame_exclusive"]) != end:
        raise ValueError(f"{sample_key(record)}: window mismatch")

    target_presence = arrays.presence_by_person[target_index][start:end]
    if len(target_presence) != TOTAL_FRAMES or not bool(target_presence.all()):
        raise ValueError(f"{sample_key(record)}: target is not present for 90 frames")
    target_world = np.asarray(
        arrays.joints_by_person[target_index][start:end],
        dtype=np.float32,
    )
    if target_world.shape != (TOTAL_FRAMES, N_JOINTS, 3):
        raise ValueError(
            f"{sample_key(record)}: target joints {target_world.shape}"
        )
    origin = target_world[INPUT_FRAMES - 1, 0].copy()
    target = target_world - origin[None, None, :]
    orientation = np.asarray(
        arrays.root_orientations[target_index, start:end],
        dtype=np.float32,
    )
    if orientation.shape != (TOTAL_FRAMES, 3):
        raise ValueError(
            f"{sample_key(record)}: orientation {orientation.shape}"
        )

    object_payloads = []
    object_points_by_index: dict[int, np.ndarray] = {}
    environment_parts = []
    cues = human_cues(target[:INPUT_FRAMES])
    for object_index, object_name in enumerate(arrays.object_names):
        if not bool(arrays.object_presence[last_observed, object_index]):
            continue
        points = (
            store.object_world_points(
                arrays,
                last_observed,
                object_index,
            )
            - origin[None, :]
        ).astype(np.float32, copy=False)
        object_points_by_index[object_index] = points
        distances = np.linalg.norm(points, axis=-1)
        nearby = points[distances <= float(scene_radius)]
        if len(nearby):
            environment_parts.append(nearby)
        object_payloads.append(
            object_geometry_payload(
                object_index=object_index,
                object_name=object_name,
                points=points,
                joints_history=target[:INPUT_FRAMES],
                cues=cues,
            )
        )
    candidates = filter_candidates(
        object_payloads,
        radius=scene_radius,
        max_candidates=max_candidates,
    )

    people_rows = []
    people_parts = []
    for other_index in record.get("other_person_indices", []):
        other_index = int(other_index)
        if other_index == target_index:
            continue
        presence = arrays.presence_by_person.get(other_index)
        joints = arrays.joints_by_person.get(other_index)
        if presence is None or joints is None or not bool(presence[last_observed]):
            continue
        points = np.asarray(joints[last_observed], dtype=np.float32) - origin
        root_distance = float(np.linalg.norm(points[0]))
        if root_distance > float(other_person_radius):
            continue
        people_rows.append(
            {
                "person_index": other_index,
                "root_distance_m": root_distance,
                "points": points,
            }
        )
    people_rows.sort(
        key=lambda row: (row["root_distance_m"], row["person_index"])
    )
    people_rows = people_rows[: int(max_other_people)]
    people_parts = [row["points"] for row in people_rows]
    people_points = (
        np.concatenate(people_parts, axis=0)
        if people_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    environment = (
        np.concatenate(environment_parts, axis=0)
        if environment_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    rng = np.random.default_rng(stable_seed("scene", sample_key(record)))
    scene, scene_meta = merge_environment_and_people(
        environment,
        people_points,
        total_points=scene_points,
        max_people_points=max_other_people * N_JOINTS,
        rng=rng,
    )
    return {
        "sample_id": sample_key(record),
        "sequence": sequence,
        "target_person_index": target_index,
        "start_frame": start,
        "last_observed_frame": last_observed,
        "end_frame_exclusive": end,
        "origin": origin.astype(np.float32),
        "joints_input": target[:INPUT_FRAMES].astype(np.float32),
        "joints_label": target[INPUT_FRAMES:].astype(np.float32),
        "orient_input": orientation[:INPUT_FRAMES].astype(np.float32),
        "orient_label": orientation[INPUT_FRAMES:].astype(np.float32),
        "scene": scene,
        "scene_meta": scene_meta,
        "other_people": {
            "representation": "last_observed_24_joint_point_set",
            "other_person_indices": [
                row["person_index"] for row in people_rows
            ],
            "root_distances_m": [
                row["root_distance_m"] for row in people_rows
            ],
            "num_people": len(people_rows),
            "joints_per_person": N_JOINTS,
            "points": int(len(people_points)),
            "last_observed_frame": last_observed,
            "future_person_data_used": False,
        },
        "human_cues": cues,
        "all_objects": object_payloads,
        "candidates": candidates,
        "object_points_by_index": object_points_by_index,
    }
