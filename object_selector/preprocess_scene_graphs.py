from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_GIMO_ROOT = Path("data/GIMO")
DEFAULT_SLICE_ROOT = DEFAULT_GIMO_ROOT / "SLICES_888s"
DEFAULT_INVENTORY_ROOT = Path("data/OpenIns3D/output/results_demo")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "scene_graphs"
SCHEMA_VERSION = "object_selector_scene_graph_v1"


PLY_STRUCT_TYPES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class DatasetSample:
    split: str
    split_index: int
    sample_id: str
    sample_uid: str
    scene_name: str
    sequence_name: str
    start_frame: int
    end_frame: int
    transform_path: str
    training: str


@dataclass
class SceneObject:
    object_id: str
    label: str
    semantic_label_id: int | None
    source_ply_path: Path
    raw_points: np.ndarray
    raw_centroid: np.ndarray
    raw_num_points: int


@dataclass
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-sequence scene graphs from SLICES_888s motion tensors and "
            "OpenIns3D object instance point clouds."
        )
    )
    parser.add_argument("--gimo-root", type=Path, default=DEFAULT_GIMO_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--inventory-root", type=Path, default=DEFAULT_INVENTORY_ROOT)
    parser.add_argument(
        "--sample-inventory-manifest",
        type=Path,
        default=None,
        help=(
            "Optional sample-wise inventory_manifest.json. When provided, each "
            "sample_id is loaded from its own output_directory instead of "
            "<inventory-root>/<scene_name>."
        ),
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_GIMO_ROOT / "dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--scene", type=str, default="", help="Optional scene-name filter.")
    parser.add_argument("--limit", type=int, default=-1, help="Optional max sample count after filtering.")
    parser.add_argument(
        "--point-sample",
        type=int,
        default=2048,
        help="Deterministic max points per object for min-distance and bbox metrics. <=0 uses all points.",
    )
    parser.add_argument(
        "--face-orientation",
        choices=("none", "root", "torso"),
        default="torso",
        help="Flip the face-normal ray toward this reference before reporting face_normal_alignment.",
    )
    parser.add_argument("--pelvis-joint-idx", type=int, default=0)
    parser.add_argument("--neck-joint-idx", type=int, default=12)
    parser.add_argument("--head-joint-idx", type=int, default=15)
    parser.add_argument("--left-shoulder-joint-idx", type=int, default=16)
    parser.add_argument("--right-shoulder-joint-idx", type=int, default=17)
    parser.add_argument("--nose-joint-idx", type=int, default=55)
    parser.add_argument("--right-eye-joint-idx", type=int, default=56)
    parser.add_argument("--left-eye-joint-idx", type=int, default=57)
    parser.add_argument("--round-digits", type=int, default=6)
    return parser.parse_args()


def split_from_training(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"true", "train"}:
        return "train"
    if text in {"false", "test"}:
        return "test"
    try:
        return "train" if int(float(text)) == 1 else "test"
    except ValueError:
        return "test"


def normalize_row_keys(row: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None or key == "":
            continue
        cleaned[key.lstrip("\ufeff").strip()] = value.strip() if isinstance(value, str) else value
    return cleaned


def build_sample_id(scene_name: str, sequence_name: str, start_frame: int, end_frame: int) -> str:
    return f"{scene_name}|{sequence_name}|{start_frame}|{end_frame}"


def build_sample_uid(split: str, split_index: int, sample_id: str) -> str:
    return f"{split}:{split_index:04d}:{sample_id}"


def load_samples(dataset_csv: Path, split_filter: str) -> list[DatasetSample]:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Missing dataset csv: {dataset_csv}")

    split_indices = {"train": 0, "test": 0}
    samples: list[DatasetSample] = []
    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for raw_row in csv.DictReader(file):
            row = normalize_row_keys(raw_row)
            if not row.get("sequence_path") or not row.get("scene"):
                continue
            split = split_from_training(row.get("training", "0"))
            split_index = split_indices[split]
            split_indices[split] += 1
            if split_filter != "all" and split != split_filter:
                continue

            start_frame = int(float(row["start_frame"]))
            end_frame = int(float(row["end_frame"]))
            scene_name = row["scene"]
            sequence_name = row["sequence_path"]
            sample_id = build_sample_id(
                scene_name, sequence_name, start_frame, end_frame
            )
            samples.append(
                DatasetSample(
                    split=split,
                    split_index=split_index,
                    sample_id=sample_id,
                    sample_uid=build_sample_uid(
                        split, split_index, sample_id
                    ),
                    scene_name=scene_name,
                    sequence_name=sequence_name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    transform_path=row.get("transformation", "transform_info.json") or "transform_info.json",
                    training=row.get("training", ""),
                )
            )
    return samples


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing tensor file: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(loaded)!r}")
    return loaded.detach().cpu().float()


def read_ply_header(file_obj: Any) -> tuple[str, int, list[tuple[str, str]]]:
    first_line = file_obj.readline().decode("ascii", errors="strict").strip()
    if first_line != "ply":
        raise ValueError("Not a PLY file.")

    ply_format = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    for raw_line in iter(file_obj.readline, b""):
        line = raw_line.decode("ascii", errors="strict").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[:1] == ["format"]:
            ply_format = parts[1]
        elif parts[:1] == ["element"]:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[:1] == ["property"] and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List properties are not supported in vertex PLY fallback reader.")
            vertex_properties.append((parts[2], parts[1]))
    return ply_format, vertex_count, vertex_properties


def read_ply_points(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        ply_format, vertex_count, vertex_properties = read_ply_header(file)
        property_names = [name for name, _ in vertex_properties]
        if not {"x", "y", "z"}.issubset(property_names):
            raise ValueError(f"PLY file has no x/y/z vertex properties: {path}")
        xyz_indices = [property_names.index(axis) for axis in ("x", "y", "z")]

        if ply_format == "ascii":
            points = np.empty((vertex_count, 3), dtype=np.float32)
            for index in range(vertex_count):
                values = file.readline().decode("ascii", errors="strict").split()
                points[index] = [float(values[axis_index]) for axis_index in xyz_indices]
            return points

        if ply_format != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {ply_format!r} in {path}")

        try:
            struct_format = "<" + "".join(PLY_STRUCT_TYPES[prop_type] for _, prop_type in vertex_properties)
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY property type {exc.args[0]!r} in {path}") from exc
        row_size = struct.calcsize(struct_format)
        data = file.read(row_size * vertex_count)
        points = np.empty((vertex_count, 3), dtype=np.float32)
        for index, row in enumerate(struct.iter_unpack(struct_format, data)):
            points[index] = [row[axis_index] for axis_index in xyz_indices]
        return points


class SceneInventory:
    def __init__(
        self,
        inventory_root: Path,
        sample_inventory_manifest: Path | None = None,
    ) -> None:
        self.inventory_root = inventory_root
        self._cache: dict[str, list[SceneObject]] = {}
        self._points_cache: dict[tuple[int, int], np.ndarray] = {}
        self.sample_directories: dict[str, Path] = {}
        if sample_inventory_manifest is not None:
            payload = json.loads(
                sample_inventory_manifest.read_text(encoding="utf-8")
            )
            rows = payload.get("samples")
            if not isinstance(rows, list):
                raise ValueError(
                    "sample inventory manifest must contain a samples list: "
                    f"{sample_inventory_manifest}"
                )
            for row in rows:
                sample_id = str(row.get("sample_uid", "")).strip()
                output_directory = str(
                    row.get("output_directory", "")
                ).strip()
                if not sample_id or not output_directory:
                    raise ValueError(
                        f"Invalid sample inventory row: {row}"
                    )
                if sample_id in self.sample_directories:
                    raise ValueError(
                        f"Duplicate sample inventory entry: {sample_id}"
                    )
                self.sample_directories[sample_id] = Path(output_directory)

    @property
    def is_sample_specific(self) -> bool:
        return bool(self.sample_directories)

    def directory_for(self, scene_name: str, sample_id: str) -> Path | None:
        if self.sample_directories:
            return self.sample_directories.get(sample_id)
        return self.inventory_root / scene_name

    def load(self, scene_name: str, sample_id: str) -> list[SceneObject]:
        scene_dir = self.directory_for(scene_name, sample_id)
        if scene_dir is None:
            return []
        cache_key = str(scene_dir.resolve())
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_scene(scene_name, scene_dir)
        return self._cache[cache_key]

    def _read_points(self, ply_path: Path) -> np.ndarray:
        stat = ply_path.stat()
        cache_key = (int(stat.st_dev), int(stat.st_ino))
        if cache_key not in self._points_cache:
            self._points_cache[cache_key] = read_ply_points(ply_path)
        return self._points_cache[cache_key]

    def _load_scene(
        self, scene_name: str, scene_dir: Path
    ) -> list[SceneObject]:
        label_path = scene_dir / "label_pointcloud_export.json"
        if not label_path.exists():
            return []
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        objects: list[SceneObject] = []
        seen: set[str] = set()
        for row in payload.get("instances", []):
            object_id = str(row.get("instance_id", "")).strip()
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)
            ply_path = scene_dir / str(row.get("ply_path", ""))
            if not ply_path.exists():
                continue
            raw_points = self._read_points(ply_path)
            if raw_points.size == 0:
                continue
            semantic_label_id = row.get("semantic_label_id")
            objects.append(
                SceneObject(
                    object_id=object_id,
                    label=str(row.get("semantic_label_name") or row.get("name") or object_id),
                    semantic_label_id=int(semantic_label_id) if semantic_label_id is not None else None,
                    source_ply_path=ply_path,
                    raw_points=raw_points,
                    raw_centroid=raw_points.mean(axis=0).astype(np.float32),
                    raw_num_points=int(raw_points.shape[0]),
                )
            )
        return objects


def load_sample_transform(gimo_root: Path, sample: DatasetSample) -> tuple[np.ndarray, float]:
    transform_info_path = gimo_root / sample.scene_name / sample.sequence_name / sample.transform_path
    transform_info = json.loads(transform_info_path.read_text(encoding="utf-8"))
    scale = float(transform_info["scale"])
    transform_norm = np.loadtxt(
        gimo_root / sample.scene_name / "scene_obj" / "transform_norm.txt",
        dtype=np.float32,
    ).reshape(4, 4)
    transform_norm = transform_norm.astype(np.float32)
    transform_norm[:3, 3] /= scale
    return transform_norm, scale


def raw_to_model_points(raw_points: np.ndarray, transform_norm: np.ndarray, scale: float) -> np.ndarray:
    points = raw_points.astype(np.float32) / float(scale)
    return (transform_norm[:3, :3] @ points.T + transform_norm[:3, 3:4]).T.astype(np.float32)


def sample_object_points(raw_points: np.ndarray, point_sample: int) -> np.ndarray:
    if point_sample <= 0 or raw_points.shape[0] <= point_sample:
        return raw_points
    indices = np.linspace(0, raw_points.shape[0] - 1, num=point_sample, dtype=np.int64)
    return raw_points[indices]


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


def horizontal_alignment(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    direction_valid: bool,
) -> float | None:
    if not direction_valid:
        return None
    point_direction_2d, point_valid = normalize_vector(xz(point) - xz(origin))
    heading_2d, heading_valid = normalize_vector(xz(direction))
    if not point_valid or not heading_valid:
        return None
    return float(np.dot(point_direction_2d[:2], heading_2d[:2]))


def horizontal_line_distance(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    direction_valid: bool,
) -> float | None:
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


def finite_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def vector_payload(vector: np.ndarray, digits: int) -> list[float]:
    return [round(float(component), digits) for component in vector.tolist()]


def mean_non_null(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def build_motion_cues(
    joints: np.ndarray,
    poses: np.ndarray,
    args: argparse.Namespace,
) -> MotionCues:
    last_joints = joints[-1]
    first_joints = joints[0]
    num_joints = last_joints.shape[0]

    def joint(index: int) -> np.ndarray:
        if index >= num_joints:
            return np.zeros(3, dtype=np.float32)
        return last_joints[index].astype(np.float32)

    pelvis = joint(args.pelvis_joint_idx)
    head = joint(args.head_joint_idx)
    neck = joint(args.neck_joint_idx)
    left_shoulder = joint(args.left_shoulder_joint_idx)
    right_shoulder = joint(args.right_shoulder_joint_idx)
    nose = joint(args.nose_joint_idx)
    left_eye = joint(args.left_eye_joint_idx)
    right_eye = joint(args.right_eye_joint_idx)
    eye_mid = (0.5 * (left_eye + right_eye)).astype(np.float32)

    root_motion_direction, root_motion_valid = normalize_vector(
        last_joints[args.pelvis_joint_idx] - first_joints[args.pelvis_joint_idx]
        if args.pelvis_joint_idx < num_joints
        else np.zeros(3, dtype=np.float32)
    )

    root_pose_direction, root_pose_valid = normalize_vector(
        apply_rotvec(poses[-1, :3], np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        if poses.ndim == 2 and poses.shape[1] >= 3
        else root_motion_direction
    )
    root_direction = root_pose_direction if root_pose_valid else root_motion_direction
    root_valid = root_pose_valid or root_motion_valid

    torso_direction_raw = np.cross(right_shoulder - left_shoulder, neck - pelvis)
    torso_direction, torso_valid = normalize_vector(torso_direction_raw)
    torso_direction = orient_direction(torso_direction, root_direction, root_valid)

    eye_axis = right_eye - left_eye
    nose_axis = nose - eye_mid
    face_normal_raw_direction, face_raw_valid = normalize_vector(np.cross(nose_axis, eye_axis))
    face_reference = {
        "none": (np.zeros(3, dtype=np.float32), False),
        "root": (root_direction, root_valid),
        "torso": (torso_direction, torso_valid),
    }[args.face_orientation]
    face_normal_direction = orient_direction(
        face_normal_raw_direction,
        face_reference[0],
        face_reference[1],
    )

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


def human_payload(cues: MotionCues, args: argparse.Namespace) -> dict[str, Any]:
    digits = args.round_digits
    return {
        "last_input_frame_index": -1,
        "joint_indices": {
            "pelvis": args.pelvis_joint_idx,
            "neck": args.neck_joint_idx,
            "head": args.head_joint_idx,
            "left_shoulder": args.left_shoulder_joint_idx,
            "right_shoulder": args.right_shoulder_joint_idx,
            "nose": args.nose_joint_idx,
            "right_eye": args.right_eye_joint_idx,
            "left_eye": args.left_eye_joint_idx,
        },
        "face_orientation_mode": args.face_orientation,
        "pelvis": vector_payload(cues.pelvis, digits),
        "head": vector_payload(cues.head, digits),
        "neck": vector_payload(cues.neck, digits),
        "eye_mid": vector_payload(cues.eye_mid, digits),
        "root_motion_direction": vector_payload(cues.root_motion_direction, digits),
        "root_pose_direction": vector_payload(cues.root_pose_direction, digits),
        "root_direction": vector_payload(cues.root_direction, digits),
        "torso_direction": vector_payload(cues.torso_direction, digits),
        "face_normal_raw_direction": vector_payload(cues.face_normal_raw_direction, digits),
        "face_normal_direction": vector_payload(cues.face_normal_direction, digits),
        "valid": cues.valid,
    }


def object_metric_payload(
    scene_object: SceneObject,
    transform_norm: np.ndarray,
    scale: float,
    cues: MotionCues,
    args: argparse.Namespace,
) -> dict[str, Any]:
    digits = args.round_digits
    centroid = raw_to_model_points(scene_object.raw_centroid[None, :], transform_norm, scale)[0]
    sampled_points = raw_to_model_points(
        sample_object_points(scene_object.raw_points, args.point_sample),
        transform_norm,
        scale,
    )
    bbox_min = sampled_points.min(axis=0) if sampled_points.size else centroid
    bbox_max = sampled_points.max(axis=0) if sampled_points.size else centroid

    root_distance_3d = float(np.linalg.norm(centroid - cues.pelvis))
    root_distance_xz = float(np.linalg.norm(xz(centroid) - xz(cues.pelvis)))
    head_distance_3d = float(np.linalg.norm(centroid - cues.head))
    head_distance_xz = float(np.linalg.norm(xz(centroid) - xz(cues.head)))

    face_alignment = alignment_3d(
        centroid,
        cues.eye_mid,
        cues.face_normal_direction,
        cues.valid["face_normal_direction"],
    )
    face_horizontal_alignment = horizontal_alignment(
        centroid,
        cues.eye_mid,
        cues.face_normal_direction,
        cues.valid["face_normal_direction"],
    )
    face_raw_alignment = alignment_3d(
        centroid,
        cues.eye_mid,
        cues.face_normal_raw_direction,
        cues.valid["face_normal_raw_direction"],
    )
    face_line_distance_xz = horizontal_line_distance(
        centroid,
        cues.eye_mid,
        cues.face_normal_direction,
        cues.valid["face_normal_direction"],
    )
    root_alignment = alignment_3d(centroid, cues.pelvis, cues.root_direction, cues.valid["root_direction"])
    root_horizontal_alignment = horizontal_alignment(
        centroid,
        cues.pelvis,
        cues.root_direction,
        cues.valid["root_direction"],
    )
    torso_alignment = alignment_3d(centroid, cues.pelvis, cues.torso_direction, cues.valid["torso_direction"])
    torso_horizontal_alignment = horizontal_alignment(
        centroid,
        cues.pelvis,
        cues.torso_direction,
        cues.valid["torso_direction"],
    )
    root_line_distance_xz = horizontal_line_distance(
        centroid,
        cues.pelvis,
        cues.root_direction,
        cues.valid["root_direction"],
    )
    torso_line_distance_xz = horizontal_line_distance(
        centroid,
        cues.pelvis,
        cues.torso_direction,
        cues.valid["torso_direction"],
    )
    root_torso_alignment = mean_non_null([root_alignment, torso_alignment])
    root_torso_horizontal_alignment = mean_non_null([root_horizontal_alignment, torso_horizontal_alignment])

    metrics = {
        "distance": finite_or_none(root_distance_3d, digits),
        "root_centroid_distance_3d": finite_or_none(root_distance_3d, digits),
        "root_centroid_distance_xz": finite_or_none(root_distance_xz, digits),
        "head_centroid_distance_3d": finite_or_none(head_distance_3d, digits),
        "head_centroid_distance_xz": finite_or_none(head_distance_xz, digits),
        "root_min_point_distance_3d": finite_or_none(min_distance(sampled_points, cues.pelvis), digits),
        "head_min_point_distance_3d": finite_or_none(min_distance(sampled_points, cues.head), digits),
        "face_normal_alignment": finite_or_none(face_alignment, digits),
        "face_normal_horizontal_alignment": finite_or_none(face_horizontal_alignment, digits),
        "face_normal_raw_alignment": finite_or_none(face_raw_alignment, digits),
        "face_normal_line_distance_xz": finite_or_none(face_line_distance_xz, digits),
        "root_alignment": finite_or_none(root_alignment, digits),
        "root_horizontal_alignment": finite_or_none(root_horizontal_alignment, digits),
        "root_line_distance_xz": finite_or_none(root_line_distance_xz, digits),
        "torso_alignment": finite_or_none(torso_alignment, digits),
        "torso_horizontal_alignment": finite_or_none(torso_horizontal_alignment, digits),
        "torso_line_distance_xz": finite_or_none(torso_line_distance_xz, digits),
        "root_torso_alignment": finite_or_none(root_torso_alignment, digits),
        "root_torso_horizontal_alignment": finite_or_none(root_torso_horizontal_alignment, digits),
    }

    return {
        "object_id": scene_object.object_id,
        "label": scene_object.label,
        "semantic_label_id": scene_object.semantic_label_id,
        "source_ply_path": str(scene_object.source_ply_path),
        "raw_num_points": scene_object.raw_num_points,
        "sampled_num_points_for_metrics": int(sampled_points.shape[0]),
        "centroid": vector_payload(centroid, digits),
        "bbox_min": vector_payload(bbox_min, digits),
        "bbox_max": vector_payload(bbox_max, digits),
        "metrics": metrics,
    }


def static_object_catalog(objects: list[SceneObject]) -> list[dict[str, Any]]:
    return [
        {
            "object_id": obj.object_id,
            "label": obj.label,
            "semantic_label_id": obj.semantic_label_id,
            "source_ply_path": str(obj.source_ply_path),
            "raw_num_points": obj.raw_num_points,
        }
        for obj in objects
    ]


def build_sample_graph(
    sample: DatasetSample,
    scene_objects: list[SceneObject],
    joints: np.ndarray,
    poses: np.ndarray,
    transform_norm: np.ndarray,
    scale: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cues = build_motion_cues(joints, poses, args)
    objects = [
        object_metric_payload(scene_object, transform_norm, scale, cues, args)
        for scene_object in scene_objects
    ]
    objects.sort(key=lambda row: row["object_id"])

    return {
        "schema_version": SCHEMA_VERSION,
        "split": sample.split,
        "sample_index": sample.split_index,
        "sample_id": sample.sample_id,
        "sample_uid": sample.sample_uid,
        "scene_name": sample.scene_name,
        "sequence_name": sample.sequence_name,
        "start_frame": sample.start_frame,
        "end_frame": sample.end_frame,
        "coordinate_frame": "SLICES_888s_model_frame",
        "source_coordinate_frame": "OpenIns3D_raw_scene",
        "transform": {
            "scale": finite_or_none(scale, args.round_digits),
            "transform_norm_applied": True,
        },
        "human_final_frame": human_payload(cues, args),
        "objects": objects,
    }


def ensure_tensor_lengths(samples: list[DatasetSample], tensors_by_split: dict[str, dict[str, torch.Tensor]]) -> None:
    max_index_by_split = {"train": -1, "test": -1}
    for sample in samples:
        max_index_by_split[sample.split] = max(max_index_by_split[sample.split], sample.split_index)
    for split, max_index in max_index_by_split.items():
        if max_index < 0:
            continue
        tensors = tensors_by_split[split]
        for name, tensor in tensors.items():
            if tensor.shape[0] <= max_index:
                raise ValueError(
                    f"{split}/{name} has {tensor.shape[0]} samples, but dataset csv needs index {max_index}."
                )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def process_split(
    split: str,
    samples: list[DatasetSample],
    tensors_by_split: dict[str, dict[str, torch.Tensor]],
    inventory: SceneInventory,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_dir = args.output_dir / split
    by_scene_dir = split_dir / "by_scene"
    split_dir.mkdir(parents=True, exist_ok=True)
    by_scene_dir.mkdir(parents=True, exist_ok=True)

    scene_payloads: dict[str, dict[str, Any]] = {}
    sample_summaries: list[dict[str, Any]] = []
    sample_count = 0
    total_objects = 0
    missing_inventory: list[str] = []

    jsonl_path = split_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for sample in samples:
            if sample.split != split:
                continue
            if args.scene and sample.scene_name != args.scene:
                continue
            if args.limit >= 0 and sample_count >= args.limit:
                break

            scene_objects = inventory.load(
                sample.scene_name, sample.sample_uid
            )
            if not scene_objects and sample.sample_id not in missing_inventory:
                missing_inventory.append(sample.sample_id)
            transform_norm, scale = load_sample_transform(args.gimo_root, sample)
            tensors = tensors_by_split[split]
            graph = build_sample_graph(
                sample=sample,
                scene_objects=scene_objects,
                joints=tensors["joints_input"][sample.split_index].numpy(),
                poses=tensors["poses_input"][sample.split_index].numpy(),
                transform_norm=transform_norm,
                scale=scale,
                args=args,
            )

            jsonl_file.write(json.dumps(graph, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            jsonl_file.write("\n")

            scene_payload = scene_payloads.setdefault(
                sample.scene_name,
                {
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "scene_name": sample.scene_name,
                    "coordinate_frame": "SLICES_888s_model_frame",
                    "source_coordinate_frame": "OpenIns3D_raw_scene",
                    "object_inventory_mode": (
                        "sample_specific"
                        if inventory.is_sample_specific
                        else "scene_static"
                    ),
                    "object_inventory": (
                        []
                        if inventory.is_sample_specific
                        else static_object_catalog(scene_objects)
                    ),
                    "sequences": [],
                },
            )
            scene_payload["sequences"].append(graph)

            sample_summaries.append(
                {
                    "sample_index": sample.split_index,
                    "sample_id": sample.sample_id,
                    "sample_uid": sample.sample_uid,
                    "scene_name": sample.scene_name,
                    "sequence_name": sample.sequence_name,
                    "num_objects": len(scene_objects),
                }
            )
            sample_count += 1
            total_objects += len(scene_objects)

    for scene_name, payload in sorted(scene_payloads.items()):
        write_json(by_scene_dir / f"{scene_name}.json", payload)

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "num_samples": sample_count,
        "num_scenes": len(scene_payloads),
        "total_object_entries": total_objects,
        "samples_jsonl": str(jsonl_path),
        "by_scene_dir": str(by_scene_dir),
        "missing_inventory_scenes": missing_inventory,
        "samples": sample_summaries,
    }
    write_json(split_dir / "index.json", index_payload)
    return index_payload


def main() -> None:
    args = parse_args()
    selected_splits = ["train", "test"] if args.split == "all" else [args.split]
    samples = load_samples(args.dataset_csv, args.split)

    tensors_by_split: dict[str, dict[str, torch.Tensor]] = {}
    for split in selected_splits:
        split_root = args.slice_root / split
        tensors_by_split[split] = {
            "joints_input": load_tensor(split_root / "joints_input.pth"),
            "poses_input": load_tensor(split_root / "poses_input.pth"),
        }
    ensure_tensor_lengths(samples, tensors_by_split)

    inventory = SceneInventory(
        args.inventory_root,
        args.sample_inventory_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        process_split(split, samples, tensors_by_split, inventory, args)
        for split in selected_splits
    ]
    write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "gimo_root": str(args.gimo_root),
            "slice_root": str(args.slice_root),
            "inventory_root": str(args.inventory_root),
            "sample_inventory_manifest": (
                str(args.sample_inventory_manifest)
                if args.sample_inventory_manifest is not None
                else None
            ),
            "dataset_csv": str(args.dataset_csv),
            "output_dir": str(args.output_dir),
            "splits": summaries,
            "metric_notes": {
                "distance": "Alias of root_centroid_distance_3d in SLICES_888s model coordinates.",
                "face_normal_alignment": "3D cosine from final eye_mid to object centroid against face_normal_direction.",
                "root_alignment": "3D cosine from final pelvis to object centroid against final SMPL-X root orientation.",
                "torso_alignment": "3D cosine from final pelvis to object centroid against shoulder/neck torso direction.",
            },
        },
    )
    for summary in summaries:
        print(
            f"[{summary['split']}] wrote {summary['num_samples']} samples, "
            f"{summary['num_scenes']} scenes -> {summary['samples_jsonl']}"
        )
    print(f"summary -> {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
