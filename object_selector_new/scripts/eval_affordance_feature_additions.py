#!/usr/bin/env python3
"""Evaluate point-cloud affordance feature additions over a student base set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "runs"
    / "analysis"
    / "intrinsic_feature_additions_v1"
    / "candidates_with_intrinsic_features.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

AFFORDANCE_COLUMNS = (
    "affordance_flat_top_coverage",
    "affordance_flat_top_surface_score",
    "affordance_sit_surface_score",
    "affordance_sleep_surface_score",
    "affordance_grasp_near_hand_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate point-cloud affordance feature additions on top of a base set.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-features",
        type=str,
        default=(
            "root_horizontal_alignment,face_normal_alignment,"
            "intrinsic_bbox_top_above_pelvis,object_shape_bottom_half_fraction"
        ),
    )
    parser.add_argument(
        "--extra-features",
        type=str,
        default=",".join(AFFORDANCE_COLUMNS),
    )
    parser.add_argument("--teacher-temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), [clean_row(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def parse_feature_list(raw: str) -> tuple[str, ...]:
    cols = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not cols:
        raise ValueError("feature list cannot be empty")
    return cols


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def score_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row["best_gt_exact_count"]),
        int(row["best_teacher_top1_count"]),
        -float(row["best_kl"]),
    )


def as_tensor3(values: Any, *, name: str, key: tuple[str, str, str]) -> torch.Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector for key={key}, got {values!r}")
    return torch.tensor([float(values[0]), float(values[1]), float(values[2])], dtype=torch.float32)


def as_array3(values: Any, *, name: str, key: tuple[str, str, str]) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector for key={key}, got {values!r}")
    return np.asarray([float(values[0]), float(values[1]), float(values[2])], dtype=np.float64)


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
                if not sample_id:
                    raise ValueError(f"Missing sample_id in {path}")
                if sample_id in output[split]:
                    # The combined training set already drops a small number of duplicated sample_ids.
                    # Keep the first occurrence here to stay aligned with that deduped view.
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
            raise TypeError(f"Expected joints_input [N,T,J,3] in {path}, got {type(tensor)!r}")
        output[split] = tensor.float()
    return output


def load_points(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise ValueError(f"No vertices in {path}")
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N,3] point cloud in {path}, got {points.shape}")
    return points


def bbox_distance(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    low = torch.minimum(bbox_min, bbox_max)
    high = torch.maximum(bbox_min, bbox_max)
    below = torch.clamp(low - points, min=0.0)
    above = torch.clamp(points - high, min=0.0)
    return torch.linalg.norm(below + above, dim=-1)


def safe_indices(num_joints: int, requested: tuple[int, ...]) -> list[int]:
    return [index for index in requested if 0 <= index < num_joints]


def gaussian(value: float, center: float, sigma: float) -> float:
    sigma = max(float(sigma), 1.0e-6)
    z = (float(value) - float(center)) / sigma
    return float(math.exp(-0.5 * z * z))


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def grid_occupancy_ratio(points_x: np.ndarray, points_z: np.ndarray, *, x_min: float, x_max: float, z_min: float, z_max: float, bin_size: float) -> tuple[float, float]:
    x_extent = max(float(x_max - x_min), 1.0e-6)
    z_extent = max(float(z_max - z_min), 1.0e-6)
    step = max(float(bin_size), 1.0e-4)
    x_bins = max(int(math.ceil(x_extent / step)), 1)
    z_bins = max(int(math.ceil(z_extent / step)), 1)
    total_bins = x_bins * z_bins
    x_index = np.clip(((points_x - x_min) / step).astype(np.int64), 0, x_bins - 1)
    z_index = np.clip(((points_z - z_min) / step).astype(np.int64), 0, z_bins - 1)
    occupied = np.unique(x_index * z_bins + z_index).size
    coverage = float(occupied) / float(total_bins)
    area = float(occupied) * (step * step)
    return coverage, area


def compute_affordance_features(
    *,
    key: tuple[str, str, str],
    scene_object: dict[str, Any],
    joints_input: torch.Tensor,
    points: np.ndarray,
) -> dict[str, float]:
    bbox_min = as_array3(scene_object.get("bbox_min"), name="bbox_min", key=key)
    bbox_max = as_array3(scene_object.get("bbox_max"), name="bbox_max", key=key)
    low = np.minimum(bbox_min, bbox_max)
    high = np.maximum(bbox_min, bbox_max)
    extents = np.maximum(high - low, 1.0e-6)
    x_extent, y_extent, z_extent = [float(x) for x in extents]
    xz_area = max(x_extent * z_extent, 1.0e-6)

    top_band = max(min(0.18 * y_extent, 0.12), 0.04)
    top_threshold = float(high[1] - top_band)
    top_mask = points[:, 1] >= top_threshold
    top_points = points[top_mask]
    if top_points.shape[0] == 0:
        top_points = points[np.argmax(points[:, 1]) : np.argmax(points[:, 1]) + 1]

    bin_size = max(min(math.sqrt(xz_area) / 12.0, 0.08), 0.02)
    top_coverage, top_area = grid_occupancy_ratio(
        top_points[:, 0],
        top_points[:, 2],
        x_min=float(low[0]),
        x_max=float(high[0]),
        z_min=float(low[2]),
        z_max=float(high[2]),
        bin_size=bin_size,
    )
    top_y_std = float(top_points[:, 1].std()) if top_points.shape[0] > 1 else 0.0
    top_flatness = math.exp(-top_y_std / max(0.03, 0.12 * y_extent))
    flat_top_surface_score = float(top_coverage * top_flatness)

    top_surface_y = float(np.median(top_points[:, 1]))
    sit_height_weight = gaussian(top_surface_y, center=0.48, sigma=0.16)
    sit_area_weight = clamp01(top_area / 0.20)
    sit_surface_score = float(flat_top_surface_score * sit_height_weight * sit_area_weight)

    sleep_height_weight = gaussian(top_surface_y, center=0.50, sigma=0.22)
    sleep_area_weight = clamp01(top_area / 0.75)
    sleep_surface_score = float(flat_top_surface_score * sleep_height_weight * sleep_area_weight)

    num_joints = int(joints_input.shape[1])
    hand_indices = safe_indices(num_joints, (20, 21)) or safe_indices(num_joints, (18, 19))
    if not hand_indices:
        raise ValueError(f"No hand joints available for key={key}")
    last_hands = joints_input[-1, hand_indices, :]
    bbox_min_t = as_tensor3(scene_object.get("bbox_min"), name="bbox_min", key=key)
    bbox_max_t = as_tensor3(scene_object.get("bbox_max"), name="bbox_max", key=key)
    min_bbox_dist = float(bbox_distance(last_hands, bbox_min_t, bbox_max_t).min().item())

    hand_points = last_hands.detach().cpu().numpy()
    hand_to_points = np.linalg.norm(points[None, :, :] - hand_points[:, None, :], axis=2)
    best_hand_index, _ = np.unravel_index(int(np.argmin(hand_to_points)), hand_to_points.shape)
    center_hand = hand_points[best_hand_index]
    dist_to_best_hand = np.linalg.norm(points - center_hand[None, :], axis=1)
    radius = max(0.12, 0.20 * max(x_extent, y_extent, z_extent))
    local_points = points[dist_to_best_hand <= radius]
    if local_points.shape[0] < 32:
        nearest_k = min(64, points.shape[0])
        nearest_indices = np.argpartition(dist_to_best_hand, nearest_k - 1)[:nearest_k]
        local_points = points[nearest_indices]
    local_low = local_points.min(axis=0)
    local_high = local_points.max(axis=0)
    local_extents = np.maximum(local_high - local_low, 1.0e-6)
    sorted_local = np.sort(local_extents)
    local_cross_section = float(sorted_local[0] * sorted_local[1])
    local_long_extent = float(sorted_local[2])
    proximity = math.exp(-min_bbox_dist / 0.12)
    thinness = math.exp(-local_cross_section / 0.01)
    support_length = clamp01(local_long_extent / 0.08)
    grasp_near_hand_score = float(proximity * thinness * support_length)

    return {
        "affordance_flat_top_coverage": float(top_coverage),
        "affordance_flat_top_surface_score": float(flat_top_surface_score),
        "affordance_sit_surface_score": float(sit_surface_score),
        "affordance_sleep_surface_score": float(sleep_surface_score),
        "affordance_grasp_near_hand_score": float(grasp_near_hand_score),
    }


def build_affordance_lookup(
    *,
    scene_graph_root: Path,
    slice_root: Path,
    extra_features: tuple[str, ...],
) -> dict[tuple[str, str, str], dict[str, float]]:
    requested = tuple(feature for feature in extra_features if feature in AFFORDANCE_COLUMNS)
    if not requested:
        return {}
    scene_graphs = load_scene_graphs(scene_graph_root)
    input_joints = load_input_joints(slice_root)
    points_cache: dict[Path, np.ndarray] = {}
    feature_cache: dict[tuple[str, str, str], dict[str, float]] = {}

    for split, split_samples in scene_graphs.items():
        for sample_id, scene_sample in split_samples.items():
            sample_index = int(scene_sample.get("sample_index", -1))
            if sample_index < 0:
                raise ValueError(f"Missing sample_index for split={split}, sample_id={sample_id}")
            joints = input_joints[split][sample_index]
            objects = scene_sample.get("objects", [])
            for scene_object in objects:
                object_id = str(scene_object.get("object_id", "")).strip()
                if not object_id:
                    raise ValueError(f"Missing object_id for split={split}, sample_id={sample_id}")
                key = (split, sample_id, object_id)
                source_ply = str(scene_object.get("source_ply_path", "")).strip()
                if not source_ply:
                    raise ValueError(f"Missing source_ply_path for key={key}")
                ply_path = Path(source_ply)
                if not ply_path.exists():
                    raise FileNotFoundError(f"Missing PLY for key={key}: {ply_path}")
                if ply_path not in points_cache:
                    points_cache[ply_path] = load_points(ply_path)
                computed = compute_affordance_features(
                    key=key,
                    scene_object=scene_object,
                    joints_input=joints,
                    points=points_cache[ply_path],
                )
                feature_cache[key] = {column: computed[column] for column in requested}
    return feature_cache


def write_augmented_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    extra_features: tuple[str, ...],
    affordance_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> None:
    fieldnames, rows = read_csv(input_csv)
    missing_columns = [column for column in extra_features if column not in AFFORDANCE_COLUMNS]
    if missing_columns:
        raise ValueError(f"Unknown affordance columns requested: {missing_columns}")

    out_fieldnames = list(fieldnames)
    for column in extra_features:
        if column not in out_fieldnames:
            out_fieldnames.append(column)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        split = row.get("split", "")
        sample_id = row.get("sample_id", "")
        object_id = row.get("object_id", "")
        key = (split, sample_id, object_id)
        payload = affordance_lookup.get(key)
        if payload is None:
            raise KeyError(f"Missing affordance features for key={key}")
        enriched = dict(row)
        for column in extra_features:
            if column not in payload:
                raise KeyError(f"Missing affordance column={column!r} for key={key}")
            enriched[column] = f"{float(payload[column]):.9g}"
        output_rows.append(enriched)

    write_csv(output_csv, output_rows, out_fieldnames)


def run_student(
    *,
    input_csv: Path,
    output_dir: Path,
    feature_columns: tuple[str, ...],
    teacher_temperature: float,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "object_selector_new.scripts.train_student",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--feature-columns",
        ",".join(feature_columns),
        "--teacher-temperature",
        str(teacher_temperature),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    print(f"[run] {output_dir.name}: {', '.join(feature_columns)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    summary_path = output_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    total = int(summary["num_test_samples"])
    return {
        "run_dir": str(output_dir),
        "summary_path": str(summary_path),
        "feature_columns": list(feature_columns),
        "best_epoch": int(summary["best_epoch"]),
        "best_teacher_top1_count": int(summary["best_teacher_top1_count"]),
        "best_teacher_top1_pct": float(summary["best_teacher_top1_count"]) / max(total, 1),
        "best_gt_exact_count": int(summary["best_gt_exact_count"]),
        "best_gt_exact_pct": float(summary["best_gt_exact_count"]) / max(total, 1),
        "best_kl": float(summary["best_kl"]),
        "num_test_samples": total,
    }


def main() -> None:
    args = parse_args()
    base_features = parse_feature_list(args.base_features)
    extra_features = parse_feature_list(args.extra_features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    affordance_lookup = build_affordance_lookup(
        scene_graph_root=args.scene_graph_root,
        slice_root=args.slice_root,
        extra_features=extra_features,
    )
    augmented_csv = args.output_dir / "candidates_with_affordance_features.csv"
    write_augmented_csv(
        input_csv=args.input_csv,
        output_csv=augmented_csv,
        extra_features=extra_features,
        affordance_lookup=affordance_lookup,
    )
    print(f"[info] wrote augmented csv: {augmented_csv}")

    results: list[dict[str, Any]] = []
    baseline = run_student(
        input_csv=augmented_csv,
        output_dir=args.output_dir / "runs" / "base_affordance",
        feature_columns=base_features,
        teacher_temperature=args.teacher_temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    baseline["extra_feature"] = ""
    results.append(baseline)

    for feature in extra_features:
        feature_columns = tuple(base_features) + (feature,)
        result = run_student(
            input_csv=augmented_csv,
            output_dir=args.output_dir / "runs" / f"plus_{sanitize_tag(feature)}",
            feature_columns=feature_columns,
            teacher_temperature=args.teacher_temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["extra_feature"] = feature
        results.append(result)

    ranked = sorted(results, key=score_key, reverse=True)
    write_json(args.output_dir / "results.json", ranked)
    write_json(
        args.output_dir / "summary.json",
        {
            "input_csv": str(args.input_csv),
            "scene_graph_root": str(args.scene_graph_root),
            "slice_root": str(args.slice_root),
            "augmented_csv": str(augmented_csv),
            "base_features": list(base_features),
            "extra_features": list(extra_features),
            "epochs": args.epochs,
            "teacher_temperature": args.teacher_temperature,
            "baseline_result": baseline,
            "best_result": ranked[0] if ranked else None,
        },
    )

    print("[done] ranked results")
    for index, row in enumerate(ranked, start=1):
        label = row["extra_feature"] or "BASE"
        print(
            f"{index}. {label}: gt={row['best_gt_exact_count']}/{row['num_test_samples']}, "
            f"teacher={row['best_teacher_top1_count']}/{row['num_test_samples']}, "
            f"kl={row['best_kl']:.6f}"
        )


if __name__ == "__main__":
    main()
