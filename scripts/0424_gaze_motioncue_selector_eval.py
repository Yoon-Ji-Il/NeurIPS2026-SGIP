#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_GIMO_ROOT = Path("data/GIMO")
DEFAULT_SLICE_ROOT = DEFAULT_GIMO_ROOT / "SLICES_888s"
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_METADATA_CSV = REPO_ROOT / "dataset" / "combined_dataset.csv"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "object_selector" / "gaze_runs" / "gaze_motion_cue_fusion_v1"


@dataclass(frozen=True)
class CandidateRecord:
    object_id: str
    label: str
    semantic_label_id: str
    features: dict[str, float]


@dataclass(frozen=True)
class SampleRecord:
    split: str
    split_index: int
    sample_id: str
    scene_name: str
    sequence_name: str
    start_frame: str
    end_frame: str
    gt_object_id: str
    gt_label: str
    final_gaze_point: np.ndarray
    candidates: tuple[CandidateRecord, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a deterministic object selector that combines gaze proximity, "
            "gaze direction, and teacher-style motion cues without using an LLM."
        )
    )
    parser.add_argument("--gimo-root", type=Path, default=DEFAULT_GIMO_ROOT)
    parser.add_argument("--gimo-dataset-csv", type=Path, default=DEFAULT_GIMO_ROOT / "dataset.csv")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--bbox-margin", type=float, default=0.0)
    parser.add_argument("--gaze-bbox-scale", type=float, default=0.20)
    parser.add_argument("--gaze-centroid-scale", type=float, default=0.75)
    parser.add_argument("--gaze-ray-scale", type=float, default=0.35)
    parser.add_argument("--face-line-scale", type=float, default=0.50)
    parser.add_argument("--gaze-centroid-weight", type=float, default=0.35)
    parser.add_argument("--face-alignment-weight", type=float, default=0.50)
    parser.add_argument("--body-alignment-weight", type=float, default=2.00)
    parser.add_argument("--face-line-weight", type=float, default=0.50)
    parser.add_argument("--motion-scale-grid", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--gaze-distance-grid", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--gaze-alignment-grid", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--gaze-ray-grid", type=str, default="0.25,0.5,1.0")
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        row[clean_key] = value.strip() if isinstance(value, str) else str(value)
    return row


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                rows = [clean_row(row) for row in csv.DictReader(file)]
            if not rows:
                raise RuntimeError(f"No rows found in {path}")
            return rows
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Could not decode {path}; tried encodings: {errors}",
    )


def norm_value(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_float_grid(raw: str) -> list[float]:
    values = [safe_float(part, default=float("nan")) for part in str(raw).split(",")]
    output = [value for value in values if math.isfinite(value)]
    if not output:
        raise ValueError(f"Expected at least one finite float in grid spec: {raw!r}")
    return output


def build_sample_id(scene_name: str, sequence_name: str, start_frame: str, end_frame: str) -> str:
    return "|".join(
        (
            norm_value(scene_name),
            norm_value(sequence_name),
            norm_value(start_frame),
            norm_value(end_frame),
        )
    )


def split_name_from_row(row: dict[str, str]) -> str:
    training = norm_value(row.get("training", "0")).lower()
    if training in {"1", "true", "train"}:
        return "train"
    return "test"


def load_scene_graphs(scene_graph_root: Path, split: str) -> dict[str, dict[str, Any]]:
    jsonl_path = scene_graph_root / split / "samples.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing scene-graph jsonl: {jsonl_path}")
    lookup: dict[str, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            graph = json.loads(line)
            sample_id = str(graph.get("sample_id", "")).strip()
            if sample_id:
                lookup[sample_id] = graph
    return lookup


def load_gaze_tensor(slice_root: Path, split: str) -> torch.Tensor:
    path = slice_root / split / "gazes.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing gaze tensor: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)!r}")
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected gaze tensor [N, T, G, 3], got {tuple(tensor.shape)}")
    return tensor.detach().cpu().float()


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for row in load_csv_rows(path):
        if str(row.get("source", "data")).strip() not in {"", "data"}:
            continue
        if not str(row.get("sequence_path", "")).strip():
            continue
        sample_id = build_sample_id(
            row.get("scene", ""),
            row.get("sequence_path", ""),
            row.get("start_frame", ""),
            row.get("end_frame", ""),
        )
        metadata[sample_id] = {key: str(value).strip() for key, value in row.items() if key is not None}
    return metadata


def gt_instance_to_object_id(value: Any) -> str:
    text = norm_value(value)
    if not text:
        return ""
    try:
        return f"instance_{int(float(text)):04d}"
    except ValueError:
        return text


def normalize_label(text: Any) -> str:
    value = str(text).strip().lower().replace("_", " ").replace("-", " ")
    value = " ".join(value.split())
    if value.endswith("es") and len(value) > 4:
        singular = value[:-2]
        if singular.endswith(("ch", "sh", "x", "s", "z")):
            return singular
    if value.endswith("s") and len(value) > 3 and not value.endswith("ss"):
        return value[:-1]
    return value


def bbox_contains(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float) -> bool:
    lower = bbox_min - float(margin)
    upper = bbox_max + float(margin)
    return bool(np.all(point >= lower) and np.all(point <= upper))


def bbox_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    clamped = np.minimum(np.maximum(point, bbox_min), bbox_max)
    return float(np.linalg.norm(point - clamped))


def exp_score(distance: float, scale: float) -> float:
    safe_scale = max(float(scale), 1.0e-6)
    return float(math.exp(-max(float(distance), 0.0) / safe_scale))


def alignment_to_unit(value: Any, default: float = 0.5) -> float:
    raw = safe_float(value, default=float("nan"))
    if not math.isfinite(raw):
        return default
    clipped = max(-1.0, min(1.0, raw))
    return float((clipped + 1.0) * 0.5)


def point_to_ray_distance(origin: np.ndarray, direction: np.ndarray, point: np.ndarray) -> float:
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-8)
    offset = point - origin
    projection = float(np.dot(offset, unit))
    if projection <= 0.0:
        return float(np.linalg.norm(offset))
    closest = origin + projection * unit
    return float(np.linalg.norm(point - closest))


def gaze_alignment(origin: np.ndarray, gaze_point: np.ndarray, object_point: np.ndarray) -> float:
    gaze_dir = gaze_point - origin
    object_dir = object_point - origin
    gaze_norm = float(np.linalg.norm(gaze_dir))
    object_norm = float(np.linalg.norm(object_dir))
    if gaze_norm <= 1.0e-8 or object_norm <= 1.0e-8:
        return 0.5
    cosine = float(np.dot(gaze_dir, object_dir) / max(gaze_norm * object_norm, 1.0e-8))
    return alignment_to_unit(cosine)


def body_alignment_norm(metrics: dict[str, Any]) -> float:
    root = metrics.get("root_alignment")
    torso = metrics.get("torso_alignment")
    if root is None and torso is None:
        return alignment_to_unit(metrics.get("root_torso_alignment"))
    values = [alignment_to_unit(value) for value in (root, torso) if value is not None]
    if not values:
        return alignment_to_unit(metrics.get("root_torso_alignment"))
    return float(sum(values) / len(values))


def candidate_features(
    *,
    object_row: dict[str, Any],
    eye_mid: np.ndarray,
    final_gaze_point: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    centroid = np.asarray(object_row.get("centroid", []), dtype=np.float32)
    bbox_min = np.asarray(object_row.get("bbox_min", []), dtype=np.float32)
    bbox_max = np.asarray(object_row.get("bbox_max", []), dtype=np.float32)
    if centroid.shape != (3,) or bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError("centroid/bbox must each have shape (3,)")

    metrics = object_row.get("metrics", {})
    is_inside_bbox = float(bbox_contains(final_gaze_point, bbox_min, bbox_max, args.bbox_margin))
    bbox_dist = bbox_distance(final_gaze_point, bbox_min, bbox_max)
    centroid_dist = float(np.linalg.norm(final_gaze_point - centroid))
    gaze_bbox_score = exp_score(bbox_dist, args.gaze_bbox_scale)
    gaze_centroid_score = exp_score(centroid_dist, args.gaze_centroid_scale)
    gaze_alignment_norm = gaze_alignment(eye_mid, final_gaze_point, centroid)
    gaze_ray_dist = point_to_ray_distance(eye_mid, final_gaze_point - eye_mid, centroid)
    gaze_ray_score = exp_score(gaze_ray_dist, args.gaze_ray_scale)
    face_alignment = alignment_to_unit(metrics.get("face_normal_alignment"))
    body_alignment = body_alignment_norm(metrics)
    face_line = exp_score(safe_float(metrics.get("face_normal_line_distance_xz"), default=1.0e6), args.face_line_scale)

    return {
        "is_inside_bbox": is_inside_bbox,
        "bbox_distance": float(bbox_dist),
        "centroid_distance": float(centroid_dist),
        "gaze_bbox_score": float(gaze_bbox_score),
        "gaze_centroid_score": float(gaze_centroid_score),
        "gaze_alignment_norm": float(gaze_alignment_norm),
        "gaze_ray_distance": float(gaze_ray_dist),
        "gaze_ray_score": float(gaze_ray_score),
        "face_alignment_norm": float(face_alignment),
        "body_alignment_norm": float(body_alignment),
        "face_line_score": float(face_line),
        "root_alignment": safe_float(metrics.get("root_alignment"), default=0.0),
        "torso_alignment": safe_float(metrics.get("torso_alignment"), default=0.0),
        "root_torso_alignment": safe_float(metrics.get("root_torso_alignment"), default=0.0),
        "face_normal_alignment": safe_float(metrics.get("face_normal_alignment"), default=0.0),
        "root_distance": safe_float(metrics.get("distance"), default=0.0),
    }


def build_sample_records(args: argparse.Namespace) -> list[SampleRecord]:
    metadata_lookup = load_metadata(args.metadata_csv)
    dataset_rows = load_csv_rows(args.gimo_dataset_csv)
    splits = ["train", "test"] if args.split == "all" else [args.split]
    samples: list[SampleRecord] = []

    for split in splits:
        split_rows = [row for row in dataset_rows if split_name_from_row(row) == split]
        scene_graph_lookup = load_scene_graphs(args.scene_graph_root, split)
        gaze_tensor = load_gaze_tensor(args.slice_root, split)
        if gaze_tensor.shape[0] != len(split_rows):
            raise ValueError(
                f"{split} gaze tensor length {gaze_tensor.shape[0]} does not match dataset rows {len(split_rows)}"
            )

        limit = len(split_rows) if args.limit < 0 else min(len(split_rows), args.limit)
        for split_index, row in enumerate(split_rows[:limit]):
            sample_id = build_sample_id(
                row["scene"],
                row["sequence_path"],
                row["start_frame"],
                row["end_frame"],
            )
            graph = scene_graph_lookup.get(sample_id)
            if graph is None:
                continue

            metadata = metadata_lookup.get(sample_id, {})
            gt_object_id = gt_instance_to_object_id(metadata.get("GT_instance", ""))
            gt_label = metadata.get("object_norm", "")

            gaze_sequence = gaze_tensor[split_index].mean(dim=1).numpy()
            final_gaze_point = np.asarray(gaze_sequence[-1], dtype=np.float32)
            eye_mid = np.asarray(graph.get("human_final_frame", {}).get("eye_mid", []), dtype=np.float32)
            if eye_mid.shape != (3,):
                eye_mid = final_gaze_point.copy()

            candidates: list[CandidateRecord] = []
            for object_row in graph.get("objects", []):
                object_id = str(object_row.get("object_id", "")).strip()
                if not object_id:
                    continue
                try:
                    features = candidate_features(
                        object_row=object_row,
                        eye_mid=eye_mid,
                        final_gaze_point=final_gaze_point,
                        args=args,
                    )
                except ValueError:
                    continue

                candidates.append(
                    CandidateRecord(
                        object_id=object_id,
                        label=str(object_row.get("label", "")).strip(),
                        semantic_label_id=str(object_row.get("semantic_label_id", "")).strip(),
                        features=features,
                    )
                )

            if not candidates:
                continue

            samples.append(
                SampleRecord(
                    split=split,
                    split_index=split_index,
                    sample_id=sample_id,
                    scene_name=norm_value(row.get("scene", "")),
                    sequence_name=norm_value(row.get("sequence_path", "")),
                    start_frame=norm_value(row.get("start_frame", "")),
                    end_frame=norm_value(row.get("end_frame", "")),
                    gt_object_id=gt_object_id,
                    gt_label=gt_label,
                    final_gaze_point=final_gaze_point,
                    candidates=tuple(candidates),
                )
            )

    return samples


def nearest_baseline_key(candidate: CandidateRecord) -> tuple[float, float, float, str]:
    features = candidate.features
    return (
        -float(features["is_inside_bbox"]),
        float(features["bbox_distance"]),
        float(features["centroid_distance"]),
        candidate.object_id,
    )


def config_from_values(
    *,
    gaze_distance_weight: float,
    gaze_alignment_weight: float,
    gaze_ray_weight: float,
    motion_scale: float,
) -> dict[str, float]:
    return {
        "gaze_distance_weight": float(gaze_distance_weight),
        "gaze_alignment_weight": float(gaze_alignment_weight),
        "gaze_ray_weight": float(gaze_ray_weight),
        "motion_scale": float(motion_scale),
    }


def combined_score(
    candidate: CandidateRecord,
    *,
    config: dict[str, float],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    features = candidate.features
    motion_face = args.face_alignment_weight * features["face_alignment_norm"]
    motion_body = args.body_alignment_weight * features["body_alignment_norm"]
    motion_line = args.face_line_weight * features["face_line_score"]
    motion_total = config["motion_scale"] * (motion_face + motion_body + motion_line)

    gaze_bbox = config["gaze_distance_weight"] * features["gaze_bbox_score"]
    gaze_centroid = args.gaze_centroid_weight * features["gaze_centroid_score"]
    gaze_align = config["gaze_alignment_weight"] * features["gaze_alignment_norm"]
    gaze_ray = config["gaze_ray_weight"] * features["gaze_ray_score"]

    total = motion_total + gaze_bbox + gaze_centroid + gaze_align + gaze_ray
    breakdown = {
        "score_total": float(total),
        "score_motion_total": float(motion_total),
        "score_gaze_bbox": float(gaze_bbox),
        "score_gaze_centroid": float(gaze_centroid),
        "score_gaze_alignment": float(gaze_align),
        "score_gaze_ray": float(gaze_ray),
        "score_motion_face": float(config["motion_scale"] * motion_face),
        "score_motion_body": float(config["motion_scale"] * motion_body),
        "score_motion_face_line": float(config["motion_scale"] * motion_line),
    }
    return float(total), breakdown


def predict_with_config(
    sample: SampleRecord,
    *,
    config: dict[str, float],
    args: argparse.Namespace,
) -> tuple[CandidateRecord, list[tuple[CandidateRecord, dict[str, float]]]]:
    scored: list[tuple[CandidateRecord, dict[str, float]]] = []
    for candidate in sample.candidates:
        total, breakdown = combined_score(candidate, config=config, args=args)
        payload = dict(candidate.features)
        payload.update(breakdown)
        scored.append((candidate, payload))
    scored.sort(
        key=lambda item: (
            -float(item[1]["score_total"]),
            float(item[0].features["bbox_distance"]),
            float(item[0].features["centroid_distance"]),
            item[0].object_id,
        )
    )
    return scored[0][0], scored


def predict_motion_only(sample: SampleRecord, *, args: argparse.Namespace) -> CandidateRecord:
    config = config_from_values(
        gaze_distance_weight=0.0,
        gaze_alignment_weight=0.0,
        gaze_ray_weight=0.0,
        motion_scale=1.0,
    )
    return predict_with_config(sample, config=config, args=args)[0]


def predict_nearest_baseline(sample: SampleRecord) -> CandidateRecord:
    ordered = sorted(sample.candidates, key=nearest_baseline_key)
    return ordered[0]


def evaluate_predictions(
    samples: list[SampleRecord],
    predictions: dict[str, str],
) -> dict[str, Any]:
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "test", "all"):
        if split_name == "all":
            split_samples = list(samples)
        else:
            split_samples = [sample for sample in samples if sample.split == split_name]
        total = len(split_samples)
        gt_in_pool = 0
        exact_instance = 0
        strict_label = 0
        relaxed_label = 0
        for sample in split_samples:
            pred_object_id = predictions.get(sample.sample_id, "")
            pred_candidate = next((candidate for candidate in sample.candidates if candidate.object_id == pred_object_id), None)
            gt_candidate = next((candidate for candidate in sample.candidates if candidate.object_id == sample.gt_object_id), None)
            if gt_candidate is not None:
                gt_in_pool += 1
            if sample.gt_object_id and pred_object_id == sample.gt_object_id:
                exact_instance += 1
            pred_label = pred_candidate.label if pred_candidate is not None else ""
            if pred_label and sample.gt_label and pred_label == sample.gt_label:
                strict_label += 1
            if normalize_label(pred_label) and normalize_label(pred_label) == normalize_label(sample.gt_label):
                relaxed_label += 1

        metrics_by_split[split_name] = {
            "samples": total,
            "gt_in_pool": gt_in_pool,
            "exact_instance_match": exact_instance,
            "exact_instance_acc": (exact_instance / total) if total else 0.0,
            "strict_label_match": strict_label,
            "strict_label_acc": (strict_label / total) if total else 0.0,
            "relaxed_label_match": relaxed_label,
            "relaxed_label_acc": (relaxed_label / total) if total else 0.0,
        }
    return metrics_by_split


def evaluate_named_run(
    *,
    name: str,
    samples: list[SampleRecord],
    predictor: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    predictions: dict[str, str] = {}
    for sample in samples:
        selected = predictor(sample)
        predictions[sample.sample_id] = selected.object_id
    metrics = evaluate_predictions(samples, predictions)
    return {"name": name, "metrics": metrics}, predictions


def build_best_config_outputs(
    *,
    samples: list[SampleRecord],
    config: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for sample in samples:
        selected, ranked = predict_with_config(sample, config=config, args=args)
        gt_in_pool = any(candidate.object_id == sample.gt_object_id for candidate in sample.candidates)
        selected_payload = next(
            payload for candidate, payload in ranked if candidate.object_id == selected.object_id
        )
        for rank, (candidate, payload) in enumerate(ranked, start=1):
            candidate_rows.append(
                {
                    "split": sample.split,
                    "split_index": sample.split_index,
                    "sample_id": sample.sample_id,
                    "scene_name": sample.scene_name,
                    "sequence_name": sample.sequence_name,
                    "object_id": candidate.object_id,
                    "label": candidate.label,
                    "semantic_label_id": candidate.semantic_label_id,
                    "gt_object_id": sample.gt_object_id,
                    "gt_label": sample.gt_label,
                    "is_gt": int(candidate.object_id == sample.gt_object_id),
                    "is_selected": int(candidate.object_id == selected.object_id),
                    "rank": rank,
                    "final_gaze_point": json.dumps([float(value) for value in sample.final_gaze_point.tolist()]),
                    **payload,
                }
            )
        sample_rows.append(
            {
                "split": sample.split,
                "split_index": sample.split_index,
                "sample_id": sample.sample_id,
                "scene_name": sample.scene_name,
                "sequence_name": sample.sequence_name,
                "start_frame": sample.start_frame,
                "end_frame": sample.end_frame,
                "selected_object_id": selected.object_id,
                "selected_label": selected.label,
                "gt_object_id": sample.gt_object_id,
                "gt_label": sample.gt_label,
                "exact_instance_match": int(selected.object_id == sample.gt_object_id),
                "strict_label_match": int(selected.label == sample.gt_label),
                "relaxed_label_match": int(normalize_label(selected.label) == normalize_label(sample.gt_label)),
                "gt_in_pool": int(gt_in_pool),
                "num_candidates": len(sample.candidates),
                "selected_score_total": float(selected_payload["score_total"]),
            }
        )
    return candidate_rows, sample_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer fieldnames for empty csv: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    samples = build_sample_records(args)
    if not samples:
        raise RuntimeError("No valid samples were built for evaluation")

    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    nearest_result, nearest_predictions = evaluate_named_run(
        name="gaze_nearest_baseline",
        samples=samples,
        predictor=predict_nearest_baseline,
    )
    motion_result, _ = evaluate_named_run(
        name="motion_only_teacher_like",
        samples=samples,
        predictor=lambda sample: predict_motion_only(sample, args=args),
    )

    default_config = config_from_values(
        gaze_distance_weight=1.5,
        gaze_alignment_weight=1.0,
        gaze_ray_weight=0.5,
        motion_scale=1.0,
    )
    default_result, _ = evaluate_named_run(
        name="gaze_motion_default",
        samples=samples,
        predictor=lambda sample: predict_with_config(sample, config=default_config, args=args)[0],
    )

    sweep_rows: list[dict[str, Any]] = []
    best_config: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for motion_scale, gaze_distance_weight, gaze_alignment_weight, gaze_ray_weight in itertools.product(
        parse_float_grid(args.motion_scale_grid),
        parse_float_grid(args.gaze_distance_grid),
        parse_float_grid(args.gaze_alignment_grid),
        parse_float_grid(args.gaze_ray_grid),
    ):
        config = config_from_values(
            gaze_distance_weight=gaze_distance_weight,
            gaze_alignment_weight=gaze_alignment_weight,
            gaze_ray_weight=gaze_ray_weight,
            motion_scale=motion_scale,
        )
        predictions = {
            sample.sample_id: predict_with_config(sample, config=config, args=args)[0].object_id
            for sample in samples
        }
        metrics = evaluate_predictions(samples, predictions)
        row = {
            **config,
            "train_exact_instance_acc": metrics["train"]["exact_instance_acc"],
            "train_exact_instance_match": metrics["train"]["exact_instance_match"],
            "train_samples": metrics["train"]["samples"],
            "test_exact_instance_acc": metrics["test"]["exact_instance_acc"],
            "test_exact_instance_match": metrics["test"]["exact_instance_match"],
            "test_samples": metrics["test"]["samples"],
            "all_exact_instance_acc": metrics["all"]["exact_instance_acc"],
            "all_exact_instance_match": metrics["all"]["exact_instance_match"],
            "all_samples": metrics["all"]["samples"],
            "all_relaxed_label_acc": metrics["all"]["relaxed_label_acc"],
            "all_relaxed_label_match": metrics["all"]["relaxed_label_match"],
        }
        sweep_rows.append(row)

        current_key = (
            metrics["train"]["exact_instance_acc"],
            metrics["train"]["relaxed_label_acc"],
            metrics["test"]["exact_instance_acc"],
            metrics["all"]["exact_instance_acc"],
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_config = config

    if best_config is None:
        raise RuntimeError("Failed to select a best configuration from the sweep")

    best_result, best_predictions = evaluate_named_run(
        name="best_train_sweep",
        samples=samples,
        predictor=lambda sample: predict_with_config(sample, config=best_config, args=args)[0],
    )
    candidate_rows, sample_rows = build_best_config_outputs(samples=samples, config=best_config, args=args)

    write_csv(args.analysis_dir / "sweep_results.csv", sweep_rows)
    write_csv(args.analysis_dir / "best_train_config_candidate_scores.csv", candidate_rows)
    write_csv(args.analysis_dir / "best_train_config_sample_summary.csv", sample_rows)

    top_configs = sorted(
        sweep_rows,
        key=lambda row: (
            float(row["train_exact_instance_acc"]),
            float(row["train_exact_instance_match"]),
            float(row["test_exact_instance_acc"]),
            float(row["all_exact_instance_acc"]),
        ),
        reverse=True,
    )[:20]

    summary = {
        "analysis_dir": str(args.analysis_dir),
        "metadata_csv": str(args.metadata_csv),
        "gimo_dataset_csv": str(args.gimo_dataset_csv),
        "scene_graph_root": str(args.scene_graph_root),
        "slice_root": str(args.slice_root),
        "num_samples": len(samples),
        "feature_formulae": {
            "face_alignment_norm": "(face_normal_alignment + 1) / 2",
            "body_alignment_norm": "mean((root_alignment + 1) / 2, (torso_alignment + 1) / 2)",
            "face_line_score": f"exp(-face_normal_line_distance_xz / {args.face_line_scale})",
            "gaze_bbox_score": f"exp(-bbox_distance / {args.gaze_bbox_scale})",
            "gaze_centroid_score": f"exp(-centroid_distance / {args.gaze_centroid_scale})",
            "gaze_alignment_norm": "((cos(eye_mid->final_gaze, eye_mid->object_centroid) + 1) / 2)",
            "gaze_ray_score": f"exp(-distance(object_centroid, gaze_ray_from_eye_mid) / {args.gaze_ray_scale})",
        },
        "selection_notes": {
            "bbox_bonus_removed": True,
            "bbox_used_only_for_diagnostics_and_nearest_baseline": True,
            "gaze_ray_definition": "ray origin is eye_mid and direction is eye_mid -> final_gaze_point",
        },
        "teacher_motion_prior_weights": {
            "face_alignment_weight": args.face_alignment_weight,
            "body_alignment_weight": args.body_alignment_weight,
            "face_line_weight": args.face_line_weight,
        },
        "baselines": [nearest_result, motion_result, default_result],
        "best_train_config": {
            "config": best_config,
            "metrics": best_result["metrics"],
        },
        "top_sweep_configs": top_configs,
        "artifacts": {
            "sweep_csv": str(args.analysis_dir / "sweep_results.csv"),
            "best_candidate_csv": str(args.analysis_dir / "best_train_config_candidate_scores.csv"),
            "best_sample_csv": str(args.analysis_dir / "best_train_config_sample_summary.csv"),
        },
        "nearest_baseline_predictions_file": str(args.analysis_dir / "nearest_baseline_predictions.json"),
        "best_train_predictions_file": str(args.analysis_dir / "best_train_predictions.json"),
    }
    write_json(args.analysis_dir / "summary.json", summary)
    write_json(args.analysis_dir / "nearest_baseline_predictions.json", nearest_predictions)
    write_json(args.analysis_dir / "best_train_predictions.json", best_predictions)

    print(json.dumps(summary["best_train_config"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
