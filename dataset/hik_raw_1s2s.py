"""Read-only raw HIK adapter for the native H25F50 model smoke test.

The official event frame is the first future frame.  HIK stays at 25 Hz and
the 75 contiguous offsets are::

    observed: [-25, -24, ..., -1]
    future:   [0, 1, ..., 49]

Only the last observed frame of each non-target person is used.  Its 29 HIK
joints form a sparse point set that is merged into the 1,000-point scene.
Future non-target motion is never read while constructing the scene input.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
import torch


HIK_TO_MODEL = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

# Triangulated faces of HIK's eight-corner box representation.
BOX_TRIANGLES = np.asarray(
    [
        [0, 1, 2],
        [0, 2, 3],
        [1, 5, 6],
        [1, 6, 2],
        [5, 4, 7],
        [5, 7, 6],
        [1, 0, 4],
        [1, 4, 5],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=np.int64,
)


def _round_half_up(value: Fraction) -> int:
    """Round rational frame locations deterministically, including negatives."""

    numerator = value.numerator
    denominator = value.denominator
    return (2 * numerator + denominator) // (2 * denominator)


@dataclass(frozen=True)
class HIK1s2sTemporalSpec:
    source_fps: int = 25
    target_fps: int = 25
    input_frames: int = 25
    future_frames: int = 50

    @property
    def offsets(self) -> tuple[int, ...]:
        indices = range(-self.input_frames, self.future_frames)
        return tuple(
            _round_half_up(
                Fraction(index * self.source_fps, self.target_fps)
            )
            for index in indices
        )

    @property
    def input_offsets(self) -> tuple[int, ...]:
        return self.offsets[: self.input_frames]

    @property
    def future_offsets(self) -> tuple[int, ...]:
        return self.offsets[self.input_frames :]

    def frames(self, future_anchor: int) -> tuple[int, ...]:
        return tuple(future_anchor + offset for offset in self.offsets)

    def validate(self) -> None:
        expected = (25, 25, 25, 50)
        actual = (
            self.source_fps,
            self.target_fps,
            self.input_frames,
            self.future_frames,
        )
        if actual != expected:
            raise ValueError(
                "This adapter is the fixed HIK 1s2s experiment; "
                f"expected {expected}, got {actual}"
            )


@dataclass(frozen=True)
class TrackRef:
    kitchen: str
    pid: int
    path: Path
    frames: np.ndarray


class RawHIKIndex:
    """Small metadata index over the raw per-track NPZ files."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()
        self.poses_root = self.data_root / "poses"
        self.scenes_root = self.data_root / "scenes"
        self.events_path = self.data_root / "hik_test.json"
        required = (self.poses_root, self.scenes_root, self.events_path)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "HIK raw data is incomplete. Missing: " + ", ".join(missing)
            )

        self.by_kitchen: dict[str, list[TrackRef]] = {
            key: [] for key in "ABCD"
        }
        self.by_person: dict[tuple[str, int], list[TrackRef]] = {}
        for path in sorted(self.poses_root.glob("[ABCD]_*_*.npz")):
            kitchen, pid_text, _ = path.stem.split("_")
            with np.load(path) as payload:
                frames = np.asarray(payload["frames"], dtype=np.int64)
            if frames.ndim != 1 or len(frames) == 0:
                raise ValueError(f"Invalid frames in {path}")
            if np.any(frames[1:] <= frames[:-1]):
                raise ValueError(f"Frames must be strictly increasing: {path}")
            ref = TrackRef(
                kitchen=kitchen,
                pid=int(pid_text),
                path=path,
                frames=frames,
            )
            self.by_kitchen[kitchen].append(ref)
            self.by_person.setdefault((kitchen, ref.pid), []).append(ref)

    def load_events(self) -> dict:
        with self.events_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _contains(ref: TrackRef, frame: int) -> bool:
        index = int(np.searchsorted(ref.frames, frame))
        return index < len(ref.frames) and int(ref.frames[index]) == frame

    def find_complete_track(
        self,
        kitchen: str,
        pid: int,
        frames: Iterable[int],
    ) -> TrackRef | None:
        wanted = tuple(int(frame) for frame in frames)
        for ref in self.by_person.get((kitchen, int(pid)), []):
            if all(self._contains(ref, frame) for frame in wanted):
                return ref
        return None

    def load_rows(
        self,
        ref: TrackRef,
        frames: Iterable[int],
    ) -> dict[str, np.ndarray]:
        wanted = np.asarray(tuple(int(frame) for frame in frames), dtype=np.int64)
        indices = np.searchsorted(ref.frames, wanted)
        if np.any(indices >= len(ref.frames)) or not np.array_equal(
            ref.frames[indices],
            wanted,
        ):
            raise KeyError(
                f"{ref.path.name} does not contain every requested frame "
                f"{wanted.tolist()}"
            )
        with np.load(ref.path) as payload:
            result = {
                key: np.asarray(payload[key][indices])
                for key in ("transforms", "smpl", "poses3d", "act")
            }
            result["betas"] = np.asarray(payload["betas"], dtype=np.float32)
        result["frames"] = wanted
        return result

    def other_observed_rows(
        self,
        kitchen: str,
        target_pid: int,
        input_frames: Iterable[int],
    ) -> list[tuple[int, TrackRef, dict[str, np.ndarray]]]:
        """Return other people at observed frames only.

        A person can be present in one or both sampled frames.  Missing frames
        are omitted instead of copying a future pose.
        """

        wanted = tuple(int(frame) for frame in input_frames)
        output = []
        seen_pids = set()
        for ref in self.by_kitchen[kitchen]:
            if ref.pid == int(target_pid) or ref.pid in seen_pids:
                continue
            available = tuple(
                frame for frame in wanted if self._contains(ref, frame)
            )
            if not available:
                continue
            output.append((ref.pid, ref, self.load_rows(ref, available)))
            seen_pids.add(ref.pid)
        return output

    def other_observed_pids(
        self,
        kitchen: str,
        target_pid: int,
        input_frames: Iterable[int],
    ) -> list[int]:
        """List non-target people without loading any pose payload."""

        wanted = tuple(int(frame) for frame in input_frames)
        output = []
        for (track_kitchen, pid), refs in self.by_person.items():
            if track_kitchen != kitchen or pid == int(target_pid):
                continue
            if any(
                self._contains(ref, frame)
                for ref in refs
                for frame in wanted
            ):
                output.append(pid)
        return sorted(output)


def hik_points_to_model(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.einsum("ij,...j->...i", HIK_TO_MODEL, points).astype(
        np.float32,
        copy=False,
    )


def hik_orientations_to_model(rotvecs: np.ndarray) -> np.ndarray:
    """Convert HIK-world root rotations to the model's y-up world."""

    rotvecs = np.asarray(rotvecs, dtype=np.float32).reshape(-1, 3)
    hik_rotations = Rotation.from_rotvec(rotvecs).as_matrix()
    model_rotations = np.einsum(
        "ij,bjk->bik",
        HIK_TO_MODEL.astype(np.float64),
        hik_rotations,
    )
    return Rotation.from_matrix(model_rotations).as_rotvec().astype(np.float32)


def _smplx_zeros(
    body_model,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "left_hand_pose": torch.zeros(
            batch_size,
            getattr(body_model, "num_pca_comps", 12),
            dtype=dtype,
            device=device,
        ),
        "right_hand_pose": torch.zeros(
            batch_size,
            getattr(body_model, "num_pca_comps", 12),
            dtype=dtype,
            device=device,
        ),
        "expression": torch.zeros(
            batch_size,
            getattr(body_model, "num_expression_coeffs", 10),
            dtype=dtype,
            device=device,
        ),
        "jaw_pose": torch.zeros(
            batch_size,
            3,
            dtype=dtype,
            device=device,
        ),
        "leye_pose": torch.zeros(
            batch_size,
            3,
            dtype=dtype,
            device=device,
        ),
        "reye_pose": torch.zeros(
            batch_size,
            3,
            dtype=dtype,
            device=device,
        ),
    }


@torch.no_grad()
def primary_motion_and_joints(
    index: RawHIKIndex,
    kitchen: str,
    pid: int,
    frames: Iterable[int],
    vposer,
    body_model,
    device: torch.device,
    input_frames: int = 25,
) -> tuple[torch.Tensor, torch.Tensor, TrackRef, np.ndarray]:
    """Create root-centred 38-D motion and 23 SMPL-X joints.

    HIK/HUMOF centres every sample at the target's root in the last observed
    frame.  The returned origin is in the model's y-up coordinate system and
    must also be subtracted from environment and non-target person points.
    """

    wanted = tuple(int(frame) for frame in frames)
    ref = index.find_complete_track(kitchen, pid, wanted)
    if ref is None:
        raise KeyError(
            f"No complete target track for kitchen={kitchen}, pid={pid}, "
            f"frames={wanted}"
        )
    rows = index.load_rows(ref, wanted)
    if input_frames <= 0 or input_frames >= len(wanted):
        raise ValueError(
            f"input_frames must be within the sequence, got {input_frames}"
        )
    scene_origin = hik_points_to_model(
        rows["poses3d"][input_frames - 1, 0]
    ).reshape(3)
    body_pose = torch.as_tensor(
        rows["smpl"].reshape(len(wanted), 63),
        dtype=torch.float32,
        device=device,
    )
    encoded = vposer.encode(body_pose).mean

    transforms = np.asarray(rows["transforms"], dtype=np.float32)
    global_orient = torch.as_tensor(
        hik_orientations_to_model(transforms[:, 3:6]),
        dtype=torch.float32,
        device=device,
    )
    translation = torch.as_tensor(
        hik_points_to_model(transforms[:, :3]) - scene_origin[None],
        dtype=torch.float32,
        device=device,
    )
    betas = torch.as_tensor(
        np.repeat(rows["betas"][None], len(wanted), axis=0),
        dtype=torch.float32,
        device=device,
    )

    body_model.batch_size = len(wanted)
    body_output = body_model(
        return_verts=True,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=translation,
        betas=betas,
        **_smplx_zeros(
            body_model,
            len(wanted),
            body_pose.dtype,
            device,
        ),
    )
    motion = torch.cat([global_orient, translation, encoded], dim=-1)
    joints = body_output.joints[:, :23]
    return (
        motion.detach().cpu(),
        joints.detach().cpu(),
        ref,
        scene_origin.astype(np.float32),
    )


def other_people_last_frame_joint_points(
    index: RawHIKIndex,
    kitchen: str,
    target_pid: int,
    last_observed_frame: int,
    scene_origin: np.ndarray,
    max_root_distance: float = 2.5,
    max_other_people: int = 15,
) -> tuple[np.ndarray, dict]:
    """Represent nearby non-target people as last-frame 29-joint point sets."""

    scene_origin = np.asarray(scene_origin, dtype=np.float32).reshape(3)
    candidates = []
    for (track_kitchen, pid), refs in index.by_person.items():
        if track_kitchen != kitchen or pid == int(target_pid):
            continue
        active_ref = next(
            (
                ref
                for ref in refs
                if index._contains(ref, int(last_observed_frame))
            ),
            None,
        )
        if active_ref is None:
            continue
        rows = index.load_rows(active_ref, (int(last_observed_frame),))
        points = (
            hik_points_to_model(rows["poses3d"][0])
            - scene_origin[None]
        )
        root_distance = float(np.linalg.norm(points[0]))
        if root_distance > float(max_root_distance):
            continue
        candidates.append(
            {
                "pid": int(pid),
                "distance": root_distance,
                "points": points.astype(np.float32),
            }
        )

    candidates.sort(key=lambda item: (item["distance"], item["pid"]))
    selected = candidates[: int(max_other_people)]
    if selected:
        points = np.concatenate(
            [item["points"] for item in selected],
            axis=0,
        )
    else:
        points = np.zeros((0, 3), dtype=np.float32)
    metadata = {
        "representation": "last_observed_29_joint_point_set",
        "last_observed_frame": int(last_observed_frame),
        "other_pids": [item["pid"] for item in selected],
        "root_distances_m": [item["distance"] for item in selected],
        "num_people": len(selected),
        "joints_per_person": 29,
        "points": int(len(points)),
        "max_root_distance_m": float(max_root_distance),
        "future_person_data_used": False,
    }
    return points, metadata


@torch.no_grad()
def other_people_surface_points(
    index: RawHIKIndex,
    kitchen: str,
    target_pid: int,
    input_frames: Iterable[int],
    body_model,
    device: torch.device,
    points_per_person_frame: int = 128,
) -> tuple[np.ndarray, dict]:
    """Turn every observed non-target body into sampled SMPL-X vertices."""

    if points_per_person_frame <= 0:
        raise ValueError("points_per_person_frame must be positive")
    people = index.other_observed_rows(
        kitchen,
        target_pid,
        input_frames,
    )
    records = []
    for pid, _, rows in people:
        for row_index, frame in enumerate(rows["frames"]):
            records.append(
                {
                    "pid": int(pid),
                    "frame": int(frame),
                    "transform": rows["transforms"][row_index],
                    "body_pose": rows["smpl"][row_index].reshape(63),
                    "betas": rows["betas"],
                }
            )
    if not records:
        return np.zeros((0, 3), dtype=np.float32), {
            "other_pids": [],
            "person_frame_records": 0,
            "points": 0,
        }

    body_pose = torch.as_tensor(
        np.stack([record["body_pose"] for record in records]),
        dtype=torch.float32,
        device=device,
    )
    transforms = np.stack([record["transform"] for record in records])
    global_orient = torch.as_tensor(
        hik_orientations_to_model(transforms[:, 3:6]),
        dtype=torch.float32,
        device=device,
    )
    translation = torch.as_tensor(
        hik_points_to_model(transforms[:, :3]),
        dtype=torch.float32,
        device=device,
    )
    betas = torch.as_tensor(
        np.stack([record["betas"] for record in records]),
        dtype=torch.float32,
        device=device,
    )

    body_model.batch_size = len(records)
    output = body_model(
        return_verts=True,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=translation,
        betas=betas,
        **_smplx_zeros(
            body_model,
            len(records),
            body_pose.dtype,
            device,
        ),
    )
    vertices = output.vertices.detach().cpu().numpy()
    vertex_indices = np.linspace(
        0,
        vertices.shape[1] - 1,
        num=points_per_person_frame,
        dtype=np.int64,
    )
    points = vertices[:, vertex_indices].reshape(-1, 3).astype(np.float32)
    metadata = {
        "other_pids": sorted({record["pid"] for record in records}),
        "person_frame_records": len(records),
        "points": int(len(points)),
        "observed_frames_only": True,
    }
    return points, metadata


def _box_triangles_for_frame(
    npy_path: Path,
    frame: int,
    frame_multiplier: int,
) -> np.ndarray:
    boxes = np.load(npy_path, mmap_mode="r")
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError(f"Expected [T, 8, 3] boxes in {npy_path}: {boxes.shape}")
    if frame_multiplier == 0:
        box_index = 0
    else:
        box_index = min(max(frame // frame_multiplier, 0), len(boxes) - 1)
    corners = np.asarray(boxes[box_index], dtype=np.float32)
    return corners[BOX_TRIANGLES]


def sample_hik_environment(
    data_root: str | Path,
    kitchen: str,
    frame: int,
    num_points: int,
    rng: np.random.Generator,
    scene_origin: np.ndarray | None = None,
    max_distance: float | None = None,
) -> np.ndarray:
    """Area-sample HIK box surfaces and optionally apply the HUMOF crop."""

    scene_root = Path(data_root) / "scenes" / f"{kitchen}_scene"
    triangles = []
    for json_path in sorted(scene_root.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        if info.get("shape") != "box":
            continue
        triangles.append(
            _box_triangles_for_frame(
                json_path.with_suffix(".npy"),
                int(frame),
                int(info.get("frame_multiplier", 0)),
            )
        )
    if not triangles:
        raise RuntimeError(f"No box geometry found in {scene_root}")
    triangles = np.concatenate(triangles, axis=0)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-10)
    triangles = triangles[valid]
    areas = areas[valid]
    probabilities = areas / areas.sum()

    if scene_origin is None:
        origin = np.zeros((3,), dtype=np.float32)
    else:
        origin = np.asarray(scene_origin, dtype=np.float32).reshape(3)

    def draw_points(count: int) -> np.ndarray:
        triangle_indices = rng.choice(
            len(triangles),
            size=int(count),
            replace=True,
            p=probabilities,
        )
        chosen = triangles[triangle_indices]
        u = rng.random(count, dtype=np.float32)
        v = rng.random(count, dtype=np.float32)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        sampled = (
            chosen[:, 0]
            + u[:, None] * (chosen[:, 1] - chosen[:, 0])
            + v[:, None] * (chosen[:, 2] - chosen[:, 0])
        )
        return hik_points_to_model(sampled) - origin[None]

    if max_distance is None:
        return draw_points(int(num_points)).astype(np.float32)

    accepted = []
    candidate_count = max(int(num_points) * 8, 4096)
    for _ in range(8):
        candidates = draw_points(candidate_count)
        distances = np.linalg.norm(candidates, axis=-1)
        nearby = candidates[distances <= float(max_distance)]
        if len(nearby):
            accepted.append(nearby)
        if sum(len(points) for points in accepted) >= int(num_points):
            break
    if not accepted:
        raise RuntimeError(
            f"No HIK environment points found within {max_distance} m"
        )
    accepted = np.concatenate(accepted, axis=0)
    replace = len(accepted) < int(num_points)
    keep = rng.choice(
        len(accepted),
        size=int(num_points),
        replace=replace,
    )
    return accepted[keep].astype(np.float32)


def merge_environment_and_people(
    environment_points: np.ndarray,
    people_points: np.ndarray,
    total_points: int,
    max_people_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Create one fixed-size scene while preserving a bounded human budget."""

    people_points = np.asarray(people_points, dtype=np.float32).reshape(-1, 3)
    max_people_points = min(int(max_people_points), int(total_points) - 1)
    if len(people_points) > max_people_points:
        keep = rng.choice(
            len(people_points),
            size=max_people_points,
            replace=False,
        )
        people_points = people_points[keep]

    environment_points = np.asarray(
        environment_points,
        dtype=np.float32,
    ).reshape(-1, 3)
    environment_budget = int(total_points) - len(people_points)
    if len(environment_points) < environment_budget:
        keep = rng.choice(
            len(environment_points),
            size=environment_budget,
            replace=True,
        )
    else:
        keep = rng.choice(
            len(environment_points),
            size=environment_budget,
            replace=False,
        )
    scene = np.concatenate(
        [environment_points[keep], people_points],
        axis=0,
    )
    scene = scene[rng.permutation(len(scene))]
    metadata = {
        "total_points": int(len(scene)),
        "environment_points": int(environment_budget),
        "people_points": int(len(people_points)),
    }
    return scene.astype(np.float32, copy=False), metadata


def flatten_test_events(events: dict) -> list[dict]:
    output = []
    for kitchen, actions in events.items():
        for action, samples in actions.items():
            for action_index, sample in enumerate(samples):
                output.append(
                    {
                        "kitchen": kitchen,
                        "action": action,
                        "action_index": action_index,
                        "pid": int(sample["pid"]),
                        "future_anchor": int(sample["frame"]),
                    }
                )
    return output


def audit_test_windows(
    index: RawHIKIndex,
    spec: HIK1s2sTemporalSpec,
) -> tuple[list[dict], dict]:
    """Audit all official HIK test anchors without materializing pose arrays."""

    spec.validate()
    manifest = []
    invalid = 0
    other_histogram: dict[int, int] = {}
    for sample in flatten_test_events(index.load_events()):
        frames = spec.frames(sample["future_anchor"])
        target = index.find_complete_track(
            sample["kitchen"],
            sample["pid"],
            frames,
        )
        if target is None:
            invalid += 1
            continue
        other_pids = index.other_observed_pids(
            sample["kitchen"],
            sample["pid"],
            frames[: spec.input_frames],
        )
        count = len(other_pids)
        other_histogram[count] = other_histogram.get(count, 0) + 1
        manifest.append(
            {
                **sample,
                "frames": list(frames),
                "input_frames": list(frames[: spec.input_frames]),
                "future_frames": list(frames[spec.input_frames :]),
                "track": target.path.name,
                "other_pids": other_pids,
            }
        )
    summary = {
        "source_fps": spec.source_fps,
        "target_fps": spec.target_fps,
        "input_frames": spec.input_frames,
        "future_frames": spec.future_frames,
        "offsets": list(spec.offsets),
        "total_official_events": len(manifest) + invalid,
        "valid_events": len(manifest),
        "invalid_incomplete_events": invalid,
        "other_people_histogram": {
            str(key): value for key, value in sorted(other_histogram.items())
        },
    }
    return manifest, summary
