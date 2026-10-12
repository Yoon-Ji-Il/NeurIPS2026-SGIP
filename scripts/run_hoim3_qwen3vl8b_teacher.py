#!/usr/bin/env python3
"""Qwen3-VL-8B privileged intent/object teacher for HOI-M3 HIK-scale.

The teacher sees the aligned full 3-second video (1 second history + 2 seconds
future), a target-marked last-observed RGB frame, and the observed-only HIK
geometry scene graph.  It returns an intent and a rubric score for every
candidate object.  The resulting probabilities are privileged distillation
targets and are never direct student/test inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import glob
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.hoim3_hikscale import (  # noqa: E402
    HOIM3Store,
    INPUT_FRAMES,
    OBJECT_POINTS,
    PROTOCOL,
    read_manifest,
    sample_key,
)


SCHEMA_VERSION = "hoim3-qwen3vl8b-intent-object-teacher-v6"
DEFAULT_DATA_ROOT = Path("data/HOIm3")
DEFAULT_CACHE_ROOT = (
    DEFAULT_DATA_ROOT / "ours_hikscale_direct24_h30f60"
)
DEFAULT_VIDEO_ROOT = DEFAULT_DATA_ROOT / "video_hikscale_view0"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs_hoim3" / "qwen3vl8b_teacher"
DEFAULT_MODEL = Path("pretrained_models/Qwen3-VL-8B-Instruct")
DEFAULT_TEACHER_OBJECT_NAME = (
    "final_object_qwen3_vl_8b_hoim3_hikscale_points_1024.pth"
)
SMPL24_EDGES = (
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
)


class TargetNotVisibleError(RuntimeError):
    """Raised when view 0 never shows enough target joints in the window."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
    os.replace(temporary, path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_command(args: argparse.Namespace) -> int:
    windows = load_gzip_jsonl(args.video_root / "windows.jsonl.gz")
    video_rows = load_gzip_jsonl(
        args.video_root
        / "manifests"
        / args.protocol
        / f"{args.split}.jsonl.gz"
    )
    video_by_key = {row["sample_key"]: row for row in video_rows}
    graphs = read_jsonl(
        args.cache_root / "scene_graphs" / args.split / "samples.jsonl"
    )
    video_summary = json.loads(
        (args.video_root / "summary.json").read_text(encoding="utf-8")
    )
    sequence_info = video_summary["sequence_video_info"]
    output_rows = []
    for graph in graphs:
        key = str(graph["sample_id"])
        video_sample = video_by_key.get(key)
        if video_sample is None:
            raise KeyError(f"No video join for {key}")
        window = windows[int(video_sample["window_index"])]
        if window["window_key"] != video_sample["window_key"]:
            raise ValueError(f"Window join mismatch for {key}")
        sequence = str(graph["sequence"])
        info = sequence_info[sequence]
        marker_video_frame = (
            int(window["video_start_frame"])
            + 2 * (
                int(graph["last_observed_frame"])
                - int(graph["start_frame"])
            )
        )
        output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_index": int(graph["sample_index"]),
                "sample_id": key,
                "split": args.split,
                "sequence": sequence,
                "target_person_index": int(graph["target_person_index"]),
                "start_frame": int(graph["start_frame"]),
                "last_observed_frame": int(graph["last_observed_frame"]),
                "video_path": str(window["video_paths_by_view"]["0"]),
                "video_paths_by_view": {
                    str(view): str(path)
                    for view, path in window["video_paths_by_view"].items()
                },
                "camera_view": 0,
                "window_key": str(window["window_key"]),
                "video_start_seconds": float(window["video_start_seconds"]),
                "video_end_seconds": float(window["video_end_seconds"]),
                "video_start_frame": int(window["video_start_frame"]),
                "video_end_frame_exclusive": int(
                    window["video_end_frame_exclusive"]
                ),
                "marker_video_frame": marker_video_frame,
                "marker_time_seconds": (
                    marker_video_frame / float(window["video_fps"])
                ),
                "marker_time_from_window_start_seconds": (
                    (
                        marker_video_frame
                        - int(window["video_start_frame"])
                    )
                    / float(window["video_fps"])
                ),
                "video_fps": float(window["video_fps"]),
                "calibration_path": str(info["calibration_path"]),
                "marker_path": str(
                    (
                        args.output_root
                        / "markers"
                        / args.split
                        / f"{int(graph['sample_index']):05d}.jpg"
                    ).resolve()
                ),
                "human_final_frame": graph["human_final_frame"],
                "other_people": graph["other_people"],
                "candidate_policy": graph["candidate_policy"],
                "objects": graph["objects"],
                "teacher_visibility": (
                    "privileged_full_history_plus_future_3s"
                ),
                "student_input_safe": False,
                "future_annotation_fields_exposed": [],
            }
        )
    if args.limit >= 0:
        output_rows = output_rows[: args.limit]
    write_jsonl(args.output_jsonl, output_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": args.protocol,
        "split": args.split,
        "samples": len(output_rows),
        "motion_scene_graph": str(
            (
                args.cache_root
                / "scene_graphs"
                / args.split
                / "samples.jsonl"
            ).resolve()
        ),
        "video_manifest": str(
            (
                args.video_root
                / "manifests"
                / args.protocol
                / f"{args.split}.jsonl.gz"
            ).resolve()
        ),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "teacher_visibility": "privileged_full_history_plus_future_3s",
        "student_input_safe": False,
        "future_annotation_fields_exposed": [],
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


def materialized_clip_name(row: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["window_key"]))
    return f"{safe}.mp4"


def materialize_one(
    row: dict[str, Any],
    clip_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    if clip_path.is_file() and clip_path.stat().st_size > 0:
        return str(row["window_key"]), "existing"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip_path.with_name(clip_path.stem + ".tmp.mp4")
    duration = (
        float(row["video_end_seconds"])
        - float(row["video_start_seconds"])
    )
    command = [
        str(args.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(row['video_start_seconds']):.9f}",
        "-i",
        str(row["video_path"]),
        "-t",
        f"{duration:.9f}",
        "-vf",
        f"fps={int(args.clip_fps)},scale={int(args.clip_width)}:-2",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        str(int(args.crf)),
        "-threads",
        "1",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.replace(temporary, clip_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return str(row["window_key"]), "created"


def select_best_camera_rows(
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    camera_views: tuple[int, ...],
) -> tuple[list[dict[str, Any]], Counter]:
    store = HOIM3Store(data_root)
    calibration_cache: dict[str, dict[str, Any]] = {}
    view_counts: Counter = Counter()
    output_rows = []
    for source in rows:
        row = dict(source)
        arrays = store.sequence(str(row["sequence"]))
        joints = np.asarray(
            arrays.joints_by_person[int(row["target_person_index"])][
                int(row["last_observed_frame"])
            ],
            dtype=np.float32,
        )
        calibration_path = str(row["calibration_path"])
        if calibration_path not in calibration_cache:
            calibration_cache[calibration_path] = json.loads(
                Path(calibration_path).read_text(encoding="utf-8")
            )
        calibration = calibration_cache[calibration_path]
        paths_by_view = {
            str(view): str(path)
            for view, path in row.get("video_paths_by_view", {}).items()
        }
        source_video = Path(paths_by_view.get("0", row["video_path"]))
        for view in camera_views:
            paths_by_view.setdefault(
                str(view),
                str(source_video.with_name(f"{view}.mp4")),
            )
        visibility = {}
        for view in camera_views:
            video_path = Path(paths_by_view.get(str(view), ""))
            if str(view) not in calibration or not video_path.is_file():
                continue
            visibility[view] = int(
                project_points(joints, calibration[str(view)])[1].sum()
            )
        if not visibility:
            raise FileNotFoundError(
                f"{row['sample_id']}: no synchronized camera video among "
                f"{camera_views}"
            )
        best_view = max(
            visibility,
            key=lambda view: (
                visibility[view],
                int(view == 0),
                -camera_views.index(view),
            ),
        )
        original_window_key = str(row["window_key"])
        row["camera_view"] = int(best_view)
        row["camera_visibility_counts"] = {
            str(view): int(count)
            for view, count in sorted(visibility.items())
        }
        row["target_visible_joints_at_last_observed"] = int(
            visibility[best_view]
        )
        row["video_path"] = paths_by_view[str(best_view)]
        if best_view != 0:
            row["window_key"] = f"{original_window_key}|view{best_view}"
        requested_marker = Path(row["marker_path"])
        row["marker_path"] = str(
            requested_marker.with_name(
                f"{requested_marker.stem}_view{best_view}.jpg"
            )
        )
        view_counts[str(best_view)] += 1
        output_rows.append(row)
    return output_rows, view_counts


def materialize_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input_jsonl)
    camera_views = tuple(
        int(value.strip())
        for value in args.camera_views.split(",")
        if value.strip()
    )
    if not camera_views:
        raise ValueError("--camera-views must contain at least one view")
    rows, selected_view_counts = select_best_camera_rows(
        rows,
        data_root=args.data_root,
        camera_views=camera_views,
    )
    unique = {}
    for row in rows:
        unique.setdefault(str(row["window_key"]), row)
    clip_by_window = {
        key: (args.clip_root / materialized_clip_name(row)).resolve()
        for key, row in unique.items()
    }
    status_counts = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                materialize_one,
                row,
                clip_by_window[key],
                args,
            ): key
            for key, row in unique.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                _, status = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to materialize {key}: {type(exc).__name__}: {exc}"
                ) from exc
            status_counts[status] += 1
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"clips {completed}/{len(futures)} "
                    f"created={status_counts['created']} "
                    f"existing={status_counts['existing']}",
                    flush=True,
                )
    output_rows = []
    for row in rows:
        copied = dict(row)
        copied["schema_version"] = SCHEMA_VERSION
        copied["teacher_video_path"] = str(
            clip_by_window[str(row["window_key"])]
        )
        copied["teacher_video_start_seconds"] = 0.0
        copied["teacher_video_end_seconds"] = (
            float(row["video_end_seconds"])
            - float(row["video_start_seconds"])
        )
        copied["teacher_marker_time_seconds"] = float(
            row["marker_time_from_window_start_seconds"]
        )
        copied["materialized_clip_fps"] = int(args.clip_fps)
        copied["materialized_clip_width"] = int(args.clip_width)
        output_rows.append(copied)
    write_jsonl(args.output_jsonl, output_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "samples": len(output_rows),
        "unique_windows": len(unique),
        "status_counts": dict(sorted(status_counts.items())),
        "clip_root": str(args.clip_root.resolve()),
        "clip_fps": int(args.clip_fps),
        "clip_width": int(args.clip_width),
        "camera_policy": "maximum_target_joint_visibility_at_last_observed",
        "camera_views": list(camera_views),
        "selected_view_counts": dict(sorted(selected_view_counts.items())),
        "minimum_selected_visible_joints": min(
            int(row["target_visible_joints_at_last_observed"])
            for row in output_rows
        ),
        "output_jsonl": str(args.output_jsonl.resolve()),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


def project_points(
    points: np.ndarray,
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    intrinsic = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(camera["RT"], dtype=np.float64).reshape(3, 4)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    camera_points = homogeneous @ extrinsic.T
    valid = camera_points[:, 2] > 1.0e-6
    pixels_h = camera_points @ intrinsic.T
    pixels = pixels_h[:, :2] / np.maximum(
        pixels_h[:, 2:3],
        1.0e-6,
    )
    width, height = camera["imgSize"]
    valid &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels.astype(np.float32), valid


def decode_frame(video_path: str, time_seconds: float) -> Image.Image:
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        seek_value = int(max(time_seconds - 0.25, 0.0) / float(stream.time_base))
        container.seek(
            seek_value,
            stream=stream,
            any_frame=False,
            backward=True,
        )
        selected = None
        selected_delta = float("inf")
        for frame in container.decode(stream):
            frame_time = float(frame.time or 0.0)
            delta = abs(frame_time - time_seconds)
            if delta < selected_delta:
                selected = frame
                selected_delta = delta
            if frame_time >= time_seconds:
                break
        if selected is None:
            raise RuntimeError(
                f"Could not decode {video_path} at {time_seconds:.6f}s"
            )
        return selected.to_image().convert("RGB")


def render_marker(row: dict[str, Any], data_root: Path) -> Path:
    requested_output = Path(row["marker_path"])
    output = (
        requested_output.parents[2]
        / "markers_v4"
        / str(row["split"])
        / requested_output.name
    )
    if output.is_file():
        return output
    store = HOIM3Store(data_root)
    arrays = store.sequence(str(row["sequence"]))
    calibration = json.loads(
        Path(row["calibration_path"]).read_text(encoding="utf-8")
    )
    camera_view = int(row.get("camera_view", 0))
    camera = calibration[str(camera_view)]
    start = int(row["start_frame"])
    last_observed = int(row["last_observed_frame"])
    end = start + INPUT_FRAMES + 60
    target_motion = np.asarray(
        arrays.joints_by_person[int(row["target_person_index"])][start:end],
        dtype=np.float32,
    )
    projected = [
        project_points(joints, camera)
        for joints in target_motion
    ]
    valid_counts = np.asarray(
        [int(valid.sum()) for _, valid in projected],
        dtype=np.int64,
    )
    preferred_offset = last_observed - start
    if valid_counts[preferred_offset] >= 4:
        marker_offset = preferred_offset
    else:
        candidates = np.flatnonzero(valid_counts >= 4)
        if len(candidates) == 0:
            raise TargetNotVisibleError(
                f"{row['sample_id']}: view{camera_view} maximum visible joints="
                f"{int(valid_counts.max(initial=0))}"
            )
        # Prefer the most visible frame, then the one closest to the
        # observation boundary.  The full window is allowed because this is
        # explicitly a privileged teacher input.
        marker_offset = int(
            max(
                candidates.tolist(),
                key=lambda index: (
                    int(valid_counts[index]),
                    -abs(index - preferred_offset),
                ),
            )
        )
    pixels, valid = projected[marker_offset]
    marker_video = row.get("teacher_video_path", row["video_path"])
    last_observed_time = float(row.get(
        "teacher_marker_time_seconds",
        row["marker_time_seconds"],
    ))
    marker_time = (
        last_observed_time
        + (marker_offset - preferred_offset) / 30.0
    )
    image = decode_frame(marker_video, marker_time)
    calibration_width, calibration_height = camera["imgSize"]
    pixels[:, 0] *= image.width / float(calibration_width)
    pixels[:, 1] *= image.height / float(calibration_height)
    target_width = min(1280, image.width)
    scale = target_width / image.width
    target_height = int(round(image.height * scale))
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    pixels *= scale
    draw = ImageDraw.Draw(image)
    for start, end in SMPL24_EDGES:
        if valid[start] and valid[end]:
            draw.line(
                [
                    tuple(pixels[start].tolist()),
                    tuple(pixels[end].tolist()),
                ],
                fill=(255, 30, 30),
                width=max(3, int(round(6 * scale))),
            )
    for pixel, is_valid in zip(pixels, valid, strict=True):
        if not is_valid:
            continue
        x, y = pixel.tolist()
        radius = max(3, int(round(8 * scale)))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 40, 40),
            outline=(255, 255, 255),
            width=2,
        )
    visible = pixels[valid]
    margin = max(8, int(round(28 * scale)))
    x0, y0 = visible.min(axis=0) - margin
    x1, y1 = visible.max(axis=0) + margin
    draw.rectangle(
        (float(x0), float(y0), float(x1), float(y1)),
        outline=(255, 0, 0),
        width=max(4, int(round(10 * scale))),
    )
    label = (
        f"TARGET PERSON p{int(row['target_person_index'])} "
        f"@ {marker_time:.1f}s"
    )
    font = ImageFont.load_default(size=max(16, int(round(28 * scale))))
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_x = max(float(x0), 0.0)
    label_y = max(float(y0) - text_height - 12, 0.0)
    draw.rectangle(
        (
            label_x,
            label_y,
            label_x + text_width + 12,
            label_y + text_height + 8,
        ),
        fill=(255, 0, 0),
    )
    draw.text(
        (label_x + 6, label_y + 3),
        label,
        fill=(255, 255, 255),
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jpg.tmp")
    image.save(temporary, format="JPEG", quality=90)
    os.replace(temporary, output)
    return output


def compact_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for candidate in row["objects"]:
        metrics = candidate["metrics"]
        output.append(
            {
                "object_id": candidate["object_id"],
                "name": candidate["name"],
                "candidate_reason": candidate["candidate_reason"],
                "geometry_score_no_cone": metrics[
                    "geometry_score_no_cone"
                ],
                "root_bbox_distance_m": metrics.get(
                    "root_bbox_distance_3d",
                    0.0,
                ),
                "hand_bbox_distance_m": metrics.get(
                    "hand_bbox_distance_3d",
                    0.0,
                ),
                "root_alignment": metrics.get(
                    "root_horizontal_alignment",
                    0.0,
                ),
                "torso_alignment": metrics.get(
                    "torso_horizontal_alignment",
                    0.0,
                ),
                "face_alignment": metrics.get(
                    "face_horizontal_alignment",
                    0.0,
                ),
            }
        )
    return output


def build_prompt(row: dict[str, Any]) -> str:
    candidates = compact_candidates(row)
    return f"""You are a privileged human-object interaction teacher.

The video is exactly 3 seconds: the first 1 second is observed history and the
next 2 seconds is future. The separate reference image is sampled from this
same aligned window, and its timestamp is printed in the red label. The target
person is marked in RED in that image. Track that same person through the
video. Ignore the actions of other people except when deciding that the target
interaction is human-human.

Every physical object present in the scene at the last observed frame is
included. There is no cone filtering and cone values are not used for ranking.
The provided geometry uses only observed distance, hand proximity, and
alignment. Candidate `__none__` means no physical object, including a purely
human-human interaction.

Candidates:
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

First infer the target person's short verb-object intent. Then score EVERY
candidate with four integer rubric components, using this exact order:

1. intent_affordance (1-40): how directly the object affords the inferred
   intent; 35-40 is prototypical and 1-4 is incompatible.
2. state_viability (1-28): whether the visible state permits the interaction;
   23-28 is immediately usable and 1-7 is blocked.
3. action_priority (1-15): how naturally the target person would choose this
   candidate first in the visible context.
4. scene_grounded_support (1-17): support from observed geometry; 14-17 is
   strong and 1-4 is negative.

The rubric total is the sum (4-100), with intent-related evidence weighted
much more than geometry. Use the full video as privileged visual evidence.
Do not invent or rename candidate IDs. The selected object must be the
candidate with the highest total. Use `__none__` only for no physical object
or a clearly visible, direct human-human interaction such as handing,
handshaking, hugging, or touching another person. Merely walking, standing
near another person, or not yet touching an object is NOT sufficient for
`__none__`: infer the interaction goal from the full future video. If the
intent phrase names a physical candidate, `__none__` must not be selected.

Return exactly one MINIFIED single-line JSON object with no markdown, no
explanation, no confidence, and no rationale. Use exactly these three keys:
{{"intent":"short verb-object phrase","scores":{{"candidate_id":[1,1,1,1]}},"selected_object_id":"candidate_id"}}
"""


def extract_json(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model JSON is not an object")
    return value


def normalized_result(
    parsed: dict[str, Any],
    row: dict[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    candidate_ids = [str(item["object_id"]) for item in row["objects"]]
    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("scores must be an object")
    notes = []

    def canonical_id(value: Any) -> str:
        object_id = str(value).strip()
        compact = re.sub(r"[^a-z0-9]+", "", object_id.lower())
        if compact == "none" and "__none__" in candidate_ids:
            return "__none__"
        return object_id

    canonical_raw: dict[str, Any] = {}
    unknown_ids = []
    for raw_id, value in raw_scores.items():
        object_id = canonical_id(raw_id)
        if object_id not in candidate_ids:
            unknown_ids.append(str(raw_id))
            continue
        canonical_raw[object_id] = value
    if unknown_ids:
        notes.append(f"ignored_unknown_score_ids={unknown_ids}")
    if not canonical_raw:
        raise ValueError("No recognized candidate score IDs")

    ranges = ((1, 40), (1, 28), (1, 15), (1, 17))
    component_names = (
        "intent_affordance",
        "state_viability",
        "action_priority",
        "scene_grounded_support",
    )
    components: dict[str, dict[str, int]] = {}
    rubric_scores: dict[str, float] = {}
    missing_ids = []
    for object_id in candidate_ids:
        raw_value = canonical_raw.get(object_id)
        if raw_value is None:
            missing_ids.append(object_id)
            values = [1, 1, 1, 1]
        elif isinstance(raw_value, dict):
            values = [
                raw_value.get("intent_affordance"),
                raw_value.get("state_viability"),
                raw_value.get("action_priority"),
                raw_value.get("scene_grounded_support"),
            ]
        elif isinstance(raw_value, (list, tuple)) and len(raw_value) == 4:
            values = list(raw_value)
        else:
            raise ValueError(
                f"{object_id}: score must be four-item list or component object"
            )
        parsed_values = []
        for name, value, (low, high) in zip(
            component_names,
            values,
            ranges,
            strict=True,
        ):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"{object_id}.{name} must be finite, got {value!r}"
                )
            integer = int(round(numeric))
            clamped = min(max(integer, low), high)
            if abs(numeric - integer) > 1.0e-6 or clamped != integer:
                notes.append(
                    f"normalized_component={object_id}.{name}:"
                    f"{value!r}->{clamped}"
                )
            parsed_values.append(clamped)
        components[object_id] = dict(
            zip(component_names, parsed_values, strict=True)
        )
        rubric_scores[object_id] = float(sum(parsed_values))
    if missing_ids:
        notes.append(f"filled_missing_score_ids={missing_ids}")

    selected_raw = canonical_id(parsed.get("selected_object_id", ""))
    rubric_top = max(
        candidate_ids,
        key=lambda object_id: (
            rubric_scores[object_id],
            -candidate_ids.index(object_id),
        ),
    )
    if selected_raw not in candidate_ids:
        notes.append(f"invalid_selected_id={selected_raw!r};used_rubric_argmax")
        selected_raw = rubric_top
    if selected_raw == "__none__":
        intent_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(parsed.get("intent", "")).lower().replace("_", " "),
            )
        )
        mentioned_physical_ids = []
        for candidate in row["objects"]:
            object_id = str(candidate["object_id"])
            if object_id == "__none__":
                continue
            label_tokens = {
                token
                for token in re.findall(
                    r"[a-z0-9]+",
                    (
                        str(candidate.get("name", ""))
                        + " "
                        + str(candidate.get("label", ""))
                    ).lower().replace("_", " "),
                )
                if len(token) >= 3
            }
            if label_tokens & intent_tokens:
                mentioned_physical_ids.append(object_id)
        if len(set(mentioned_physical_ids)) == 1:
            selected_raw = mentioned_physical_ids[0]
            notes.append(
                "none_overridden_by_unique_physical_object_named_in_intent"
            )

    # Match the established HIK soft-teacher policy: preserve the LLM hard
    # selection as posterior argmax, but retain the other rubric scores as a
    # soft target.  A one-point logit margin is only applied when required.
    teacher_logits = dict(rubric_scores)
    hard_anchor_applied = False
    max_other = max(
        (
            teacher_logits[object_id]
            for object_id in candidate_ids
            if object_id != selected_raw
        ),
        default=-float("inf"),
    )
    if teacher_logits[selected_raw] <= max_other:
        teacher_logits[selected_raw] = max_other + 1.0
        hard_anchor_applied = True
        notes.append("hard_selection_anchored_with_margin=1")

    values = np.asarray(
        [teacher_logits[key] for key in candidate_ids],
        dtype=np.float64,
    )
    logits = values / max(float(temperature), 1.0e-6)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return {
        "intent": str(parsed.get("intent", "")).strip(),
        "rubric_components": components,
        "rubric_scores": rubric_scores,
        "teacher_logits": teacher_logits,
        "soft_distribution": {
            key: float(probability)
            for key, probability in zip(
                candidate_ids,
                probabilities,
                strict=True,
            )
        },
        "selected_object_id": selected_raw,
        "confidence": float(parsed.get("confidence", 0.0)),
        "rationale_short": str(parsed.get("rationale_short", "")).strip(),
        "hard_anchor_applied": hard_anchor_applied,
        "normalization_notes": notes,
    }


def geometry_fallback(
    row: dict[str, Any],
    *,
    temperature: float,
    error: str,
) -> dict[str, Any]:
    candidate_ids = [str(item["object_id"]) for item in row["objects"]]
    selected_object_id = max(
        row["objects"],
        key=lambda item: float(
            item["metrics"].get(
                "geometry_score_no_cone",
                item["metrics"].get("geometry_score", 0.0),
            )
        ),
    )["object_id"]
    parsed = {
        "intent": "unknown",
        "scores": {
            item["object_id"]: [
                1,
                1,
                1,
                1 + int(
                    round(
                        16.0
                        * max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    item["metrics"].get(
                                        "geometry_score_no_cone",
                                        item["metrics"].get(
                                            "geometry_score",
                                            0.0,
                                        ),
                                    )
                                ),
                            ),
                        )
                    )
                ),
            ]
            for item in row["objects"]
        },
        "selected_object_id": selected_object_id,
        "confidence": 0.0,
        "rationale_short": "technical fallback to observed geometry",
    }
    result = normalized_result(parsed, row, temperature=temperature)
    result["status"] = "geometry_fallback"
    result["error"] = error
    return result


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    rows = read_jsonl(path)
    kept = [
        row for row in rows
        if (
            str(row.get("status"))
            in {"ok", "target_not_visible_geometry_fallback"}
            and str(row.get("schema_version")) == SCHEMA_VERSION
        )
    ]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
        print(
            f"removed {len(rows) - len(kept)} fallback rows for retry",
            flush=True,
        )
    return {str(row["sample_id"]) for row in kept}


def build_messages(
    row: dict[str, Any],
    marker_path: Path,
    prompt: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This aligned reference image marks the target person "
                        "in red; its timestamp is printed in the label."
                    ),
                },
                {
                    "type": "image",
                    "image": f"file://{marker_path.resolve()}",
                    "min_pixels": args.image_min_pixels,
                    "max_pixels": args.image_max_pixels,
                },
                {
                    "type": "text",
                    "text": (
                        "Now inspect the full aligned 3-second privileged video."
                    ),
                },
                {
                    "type": "video",
                    "video": (
                        "file://"
                        f"{Path(row.get('teacher_video_path', row['video_path'])).resolve()}"
                    ),
                    "video_start": float(
                        row.get(
                            "teacher_video_start_seconds",
                            row["video_start_seconds"],
                        )
                    ),
                    "video_end": float(
                        row.get(
                            "teacher_video_end_seconds",
                            row["video_end_seconds"],
                        )
                    ),
                    "nframes": args.video_frames,
                    "min_pixels": args.video_min_pixels,
                    "max_pixels": args.video_max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def infer_command(args: argparse.Namespace) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
    rows = read_jsonl(args.input_jsonl)
    assigned = [
        row
        for row in rows
        if int(row["sample_index"]) % args.num_shards == args.shard_id
    ]
    seen = existing_ids(args.output_jsonl) if args.resume else set()
    pending = [
        row for row in assigned if str(row["sample_id"]) not in seen
    ]
    if args.limit >= 0:
        pending = pending[: args.limit]
    print(
        f"shard={args.shard_id}/{args.num_shards} assigned={len(assigned)} "
        f"existing={len(seen)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return 0

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    model.generation_config.do_sample = False
    for attribute in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attribute):
            setattr(model.generation_config, attribute, None)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output_jsonl.is_file() else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as output:
        for position, row in enumerate(pending, start=1):
            error = None
            raw = ""
            result = None
            prompt = build_prompt(row)
            prompt_hash = sha256_text(prompt)
            marker = None
            try:
                marker = render_marker(row, args.data_root)
            except TargetNotVisibleError as exc:
                error = f"{type(exc).__name__}:{exc}"
                result = geometry_fallback(
                    row,
                    temperature=args.teacher_temperature,
                    error=error,
                )
                result["status"] = "target_not_visible_geometry_fallback"
            if result is None:
                for attempt in range(max(args.retries, 1)):
                    try:
                        messages = build_messages(row, marker, prompt, args)
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        image_inputs, video_inputs, video_kwargs = process_vision_info(
                            messages,
                            return_video_kwargs=True,
                            image_patch_size=processor.image_processor.patch_size,
                        )
                        # qwen-vl-utils returns one fps value per video, while
                        # Transformers 5.x validates the single-video batch field
                        # as a scalar.  This pipeline always supplies exactly one
                        # video for each inference sample.
                        fps_value = video_kwargs.get("fps")
                        if isinstance(fps_value, list) and len(fps_value) == 1:
                            video_kwargs["fps"] = fps_value[0]
                        if video_inputs:
                            clip_fps = float(
                                row.get(
                                    "materialized_clip_fps",
                                    video_kwargs.get("fps", 4.0),
                                )
                            )
                            frame_count = int(video_inputs[0].shape[0])
                            video_kwargs["video_metadata"] = [
                                {
                                    "total_num_frames": frame_count,
                                    "fps": clip_fps,
                                    "width": int(video_inputs[0].shape[-1]),
                                    "height": int(video_inputs[0].shape[-2]),
                                    "duration": frame_count / clip_fps,
                                    "video_backend": "torchvision",
                                    "frames_indices": list(range(frame_count)),
                                }
                            ]
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                            **(video_kwargs if video_inputs else {}),
                        ).to(args.device)
                        with torch.inference_mode():
                            generated = model.generate(
                                **inputs,
                                do_sample=False,
                                max_new_tokens=args.max_new_tokens,
                            )
                        trimmed = [
                            output_ids[len(input_ids) :]
                            for input_ids, output_ids in zip(
                                inputs.input_ids,
                                generated,
                                strict=True,
                            )
                        ]
                        raw = processor.batch_decode(
                            trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                        result = normalized_result(
                            extract_json(raw),
                            row,
                            temperature=args.teacher_temperature,
                        )
                        result["status"] = "ok"
                        result["error"] = None
                        break
                    except Exception as exc:
                        error = (
                            f"attempt={attempt + 1}:"
                            f"{type(exc).__name__}:{exc}"
                        )
            if result is None:
                if not args.allow_geometry_fallback:
                    raise RuntimeError(
                        f"{row['sample_id']}: Qwen failed: {error}"
                    )
                result = geometry_fallback(
                    row,
                    temperature=args.teacher_temperature,
                    error=str(error),
                )
            selected = next(
                item
                for item in row["objects"]
                if item["object_id"] == result["selected_object_id"]
            )
            output_row = {
                "schema_version": SCHEMA_VERSION,
                "sample_index": int(row["sample_index"]),
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "sequence": str(row["sequence"]),
                "target_person_index": int(row["target_person_index"]),
                "start_frame": int(row["start_frame"]),
                "last_observed_frame": int(row["last_observed_frame"]),
                **result,
                "selected_object_index": int(selected["object_index"]),
                "selected_object_name": str(selected["name"]),
                "raw_model_output": raw,
                "model": str(Path(args.model).resolve()),
                "dtype": args.dtype,
                "teacher_temperature": args.teacher_temperature,
                "prompt_sha256": prompt_hash,
                "marker_path": str(marker) if marker is not None else None,
                "teacher_visibility": row["teacher_visibility"],
                "student_input_safe": False,
            }
            output.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            output.flush()
            print(
                f"[{position}/{len(pending)}] {row['sample_id']} "
                f"intent={result['intent']!r} "
                f"object={result['selected_object_id']} "
                f"status={result['status']}",
                flush=True,
            )
    return 0


def merge_command(args: argparse.Namespace) -> int:
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_glob}")
    by_id = {}
    for path in paths:
        for row in read_jsonl(path):
            key = str(row["sample_id"])
            if key in by_id:
                raise ValueError(f"Duplicate sample {key}")
            by_id[key] = row
    rows = sorted(by_id.values(), key=lambda row: int(row["sample_index"]))
    if args.expected_count >= 0 and len(rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count}, found {len(rows)}"
        )
    write_jsonl(args.output_jsonl, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "samples": len(rows),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in rows).items())
        ),
        "selected_object_counts": dict(
            sorted(
                Counter(
                    row["selected_object_name"] for row in rows
                ).items()
            )
        ),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "input_shards": [str(path.resolve()) for path in paths],
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


def export_split(
    split: str,
    args: argparse.Namespace,
    store: HOIM3Store,
) -> dict:
    source_rows = read_manifest(args.data_root, split, args.protocol)
    selections = read_jsonl(args.selection_root / split / "selections.jsonl")
    if len(source_rows) != len(selections):
        raise ValueError(
            f"{split}: manifest={len(source_rows)} selections={len(selections)}"
        )
    tensor = torch.zeros(
        (len(source_rows), OBJECT_POINTS, 3),
        dtype=torch.float32,
    )
    mask = torch.zeros((len(source_rows),), dtype=torch.bool)
    manifest = []
    for index, (record, selection) in enumerate(
        zip(source_rows, selections, strict=True)
    ):
        key = sample_key(record)
        if key != selection["sample_id"]:
            raise ValueError(
                f"{split}[{index}] order mismatch: {key} vs "
                f"{selection['sample_id']}"
            )
        object_index = int(selection["selected_object_index"])
        entry = {
            "index": index,
            "sample_id": key,
            "selected_object_id": selection["selected_object_id"],
            "selected_object_index": object_index,
            "selected_object_name": selection["selected_object_name"],
            "intent": selection["intent"],
            "rubric_scores": selection["rubric_scores"],
            "soft_distribution": selection["soft_distribution"],
            "selector_status": selection["status"],
        }
        if object_index < 0:
            entry["status"] = "no_physical_object"
        else:
            arrays = store.sequence(str(record["sequence"]))
            last = int(record["last_observed_frame"])
            target = int(record["target_person_index"])
            origin = np.asarray(
                arrays.joints_by_person[target][last, 0],
                dtype=np.float32,
            )
            points = (
                store.object_world_points(arrays, last, object_index)
                - origin[None, :]
            )
            tensor[index] = torch.from_numpy(points)
            mask[index] = True
            entry["status"] = "ok"
        manifest.append(entry)
    split_dir = args.cache_root / "SLICES_888s" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    points_path = split_dir / args.output_name
    mask_path = split_dir / args.output_name.replace(".pth", "_mask.pth")
    temporary = points_path.with_suffix(".pth.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, points_path)
    temporary = mask_path.with_suffix(".pth.tmp")
    torch.save(mask, temporary)
    os.replace(temporary, mask_path)
    write_json(
        split_dir / args.output_name.replace(".pth", "_manifest.json"),
        manifest,
    )
    return {
        "split": split,
        "samples": len(source_rows),
        "shape": list(tensor.shape),
        "valid_objects": int(mask.sum().item()),
        "none_objects": int((~mask).sum().item()),
        "points_path": str(points_path),
        "mask_path": str(mask_path),
        "coordinate_frame": "target_last_observed_root_centered_y_up",
        "teacher_visibility": "privileged_full_history_plus_future_3s",
    }


def export_command(args: argparse.Namespace) -> int:
    store = HOIM3Store(args.data_root)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    summaries = {
        split: export_split(split, args, store)
        for split in splits
    }
    write_json(args.selection_root / "teacher_object_cache_summary.json", summaries)
    print(json.dumps(summaries, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prepare.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    prepare.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare.add_argument("--protocol", default=PROTOCOL)
    prepare.add_argument("--split", choices=("train", "test"), required=True)
    prepare.add_argument("--limit", type=int, default=-1)
    prepare.add_argument("--output-jsonl", type=Path, required=True)
    prepare.add_argument("--summary-json", type=Path, required=True)
    prepare.set_defaults(func=prepare_command)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--input-jsonl", type=Path, required=True)
    materialize.add_argument("--output-jsonl", type=Path, required=True)
    materialize.add_argument("--summary-json", type=Path, required=True)
    materialize.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    materialize.add_argument(
        "--camera-views",
        default="0,14,28",
        help="Synchronized camera views considered for target visibility.",
    )
    materialize.add_argument(
        "--clip-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "clips",
    )
    materialize.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("ffmpeg"),
    )
    materialize.add_argument("--workers", type=int, default=16)
    materialize.add_argument("--clip-fps", type=int, default=4)
    materialize.add_argument("--clip-width", type=int, default=448)
    materialize.add_argument(
        "--crf",
        type=int,
        default=5,
        help="MPEG-4 qscale (kept as --crf for launcher compatibility).",
    )
    materialize.set_defaults(func=materialize_command)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--input-jsonl", type=Path, required=True)
    infer.add_argument("--output-jsonl", type=Path, required=True)
    infer.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    infer.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    infer.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    infer.add_argument("--image-min-pixels", type=int, default=256 * 28 * 28)
    infer.add_argument("--image-max-pixels", type=int, default=640 * 28 * 28)
    infer.add_argument("--video-min-pixels", type=int, default=128 * 28 * 28)
    infer.add_argument("--video-max-pixels", type=int, default=256 * 28 * 28)
    infer.add_argument("--video-frames", type=int, default=12)
    infer.add_argument("--max-new-tokens", type=int, default=384)
    infer.add_argument("--teacher-temperature", type=float, default=20.0)
    infer.add_argument("--retries", type=int, default=2)
    infer.add_argument("--num-shards", type=int, default=1)
    infer.add_argument("--shard-id", type=int, default=0)
    infer.add_argument("--limit", type=int, default=-1)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--allow-geometry-fallback", action="store_true")
    infer.set_defaults(func=infer_command)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input-glob", required=True)
    merge.add_argument("--expected-count", type=int, default=-1)
    merge.add_argument("--output-jsonl", type=Path, required=True)
    merge.add_argument("--summary-json", type=Path, required=True)
    merge.set_defaults(func=merge_command)

    export = subparsers.add_parser("export")
    export.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    export.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    export.add_argument("--selection-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    export.add_argument("--protocol", default=PROTOCOL)
    export.add_argument("--split", choices=("train", "test", "all"), default="all")
    export.add_argument("--output-name", default=DEFAULT_TEACHER_OBJECT_NAME)
    export.set_defaults(func=export_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if hasattr(args, "output_name") and not args.output_name.endswith(".pth"):
        raise ValueError("--output-name must end with .pth")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
