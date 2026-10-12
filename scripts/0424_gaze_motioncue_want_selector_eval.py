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
DEFAULT_TEACHER_CANDIDATE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "teacher_candidate_scores.csv"
)
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "object_selector" / "gaze_runs" / "gaze_motioncue_want_v1"
ARTIFACT_PREFIX = "0424_gaze_motioncue_want"
NEG_INF_THRESHOLD = -1.0e8


@dataclass(frozen=True)
class CandidateRecord:
    split: str
    sample_id: str
    object_id: str
    label: str
    want: str
    gt_object_id: str
    gt_label: str
    is_gt: bool
    in_candidate_pool: bool
    original_is_final: bool
    original_score: float
    semantic_total: float
    posterior: float
    features: dict[str, float]


@dataclass(frozen=True)
class SampleRecord:
    row_uid: str
    split: str
    split_index: int
    sample_id: str
    scene_name: str
    sequence_name: str
    start_frame: str
    end_frame: str
    want: str
    gt_object_id: str
    gt_label: str
    original_object_id: str
    final_gaze_point: np.ndarray
    candidates: tuple[CandidateRecord, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add gaze distance/alignment scores to the original want-aware LLM+motion "
            "teacher fusion score and evaluate object-selection accuracy."
        )
    )
    parser.add_argument("--teacher-candidate-csv", type=Path, default=DEFAULT_TEACHER_CANDIDATE_CSV)
    parser.add_argument("--gimo-dataset-csv", type=Path, default=DEFAULT_GIMO_ROOT / "dataset.csv")
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--base-score-column", type=str, default="masked_fusion_total")
    parser.add_argument("--gaze-bbox-scale", type=float, default=0.20)
    parser.add_argument("--gaze-centroid-scale", type=float, default=0.75)
    parser.add_argument("--gaze-centroid-weight", type=float, default=0.35)
    parser.add_argument("--gaze-distance-grid", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0")
    parser.add_argument("--gaze-alignment-grid", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0")
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


def build_dataset_entries(path: Path, split_filter: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    counters = {"train": 0, "test": 0}
    for row in load_csv_rows(path):
        split = split_name_from_row(row)
        split_index = counters[split]
        counters[split] += 1
        if split_filter != "all" and split != split_filter:
            continue
        sample_id = build_sample_id(
            row.get("scene", ""),
            row.get("sequence_path", ""),
            row.get("start_frame", ""),
            row.get("end_frame", ""),
        )
        entries.append(
            {
                "row_uid": f"{split}|{split_index}|{sample_id}",
                "split": split,
                "split_index": split_index,
                "sample_id": sample_id,
                "scene_name": norm_value(row.get("scene", "")),
                "sequence_name": norm_value(row.get("sequence_path", "")),
                "start_frame": norm_value(row.get("start_frame", "")),
                "end_frame": norm_value(row.get("end_frame", "")),
            }
        )
    return entries


def exp_score(distance: float, scale: float) -> float:
    safe_scale = max(float(scale), 1.0e-6)
    return float(math.exp(-max(float(distance), 0.0) / safe_scale))


def alignment_to_unit(value: Any, default: float = 0.5) -> float:
    raw = safe_float(value, default=float("nan"))
    if not math.isfinite(raw):
        return default
    clipped = max(-1.0, min(1.0, raw))
    return float((clipped + 1.0) * 0.5)


def bbox_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    clamped = np.minimum(np.maximum(point, bbox_min), bbox_max)
    return float(np.linalg.norm(point - clamped))


def gaze_alignment(origin: np.ndarray, gaze_point: np.ndarray, object_point: np.ndarray) -> float:
    gaze_dir = gaze_point - origin
    object_dir = object_point - origin
    gaze_norm = float(np.linalg.norm(gaze_dir))
    object_norm = float(np.linalg.norm(object_dir))
    if gaze_norm <= 1.0e-8 or object_norm <= 1.0e-8:
        return 0.5
    cosine = float(np.dot(gaze_dir, object_dir) / max(gaze_norm * object_norm, 1.0e-8))
    return alignment_to_unit(cosine)


def candidate_gaze_features(
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

    bbox_dist = bbox_distance(final_gaze_point, bbox_min, bbox_max)
    centroid_dist = float(np.linalg.norm(final_gaze_point - centroid))
    bbox_score = exp_score(bbox_dist, args.gaze_bbox_scale)
    centroid_score = exp_score(centroid_dist, args.gaze_centroid_scale)
    distance_score = bbox_score + args.gaze_centroid_weight * centroid_score
    alignment_score = gaze_alignment(eye_mid, final_gaze_point, centroid)
    return {
        "gaze_bbox_distance": float(bbox_dist),
        "gaze_centroid_distance": float(centroid_dist),
        "gaze_bbox_score": float(bbox_score),
        "gaze_centroid_score": float(centroid_score),
        "gaze_distance_score": float(distance_score),
        "gaze_alignment_score": float(alignment_score),
    }


def load_teacher_candidate_groups(path: Path, split_filter: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in load_csv_rows(path):
        split = str(row.get("split", "")).strip()
        if split_filter != "all" and split != split_filter:
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not split or not sample_id:
            continue
        groups.setdefault((split, sample_id), []).append(row)
    return groups


def build_sample_records(args: argparse.Namespace) -> list[SampleRecord]:
    dataset_entries = build_dataset_entries(args.gimo_dataset_csv, args.split)
    teacher_groups = load_teacher_candidate_groups(args.teacher_candidate_csv, args.split)
    scene_graphs = {
        split: load_scene_graphs(args.scene_graph_root, split)
        for split in ("train", "test")
        if args.split in {"all", split}
    }
    gaze_tensors = {
        split: load_gaze_tensor(args.slice_root, split)
        for split in ("train", "test")
        if args.split in {"all", split}
    }

    samples: list[SampleRecord] = []
    sample_cache: dict[tuple[str, str], SampleRecord] = {}
    for dataset_info in dataset_entries:
        if args.limit >= 0 and len(samples) >= args.limit:
            break

        split = str(dataset_info["split"])
        sample_id = str(dataset_info["sample_id"])
        sample_key = (split, sample_id)
        cached = sample_cache.get(sample_key)
        if cached is not None:
            samples.append(
                SampleRecord(
                    row_uid=str(dataset_info["row_uid"]),
                    split=split,
                    split_index=int(dataset_info["split_index"]),
                    sample_id=sample_id,
                    scene_name=str(dataset_info["scene_name"]),
                    sequence_name=str(dataset_info["sequence_name"]),
                    start_frame=str(dataset_info["start_frame"]),
                    end_frame=str(dataset_info["end_frame"]),
                    want=cached.want,
                    gt_object_id=cached.gt_object_id,
                    gt_label=cached.gt_label,
                    original_object_id=cached.original_object_id,
                    final_gaze_point=cached.final_gaze_point,
                    candidates=cached.candidates,
                )
            )
            continue

        rows = teacher_groups.get(sample_key)
        if rows is None:
            continue
        graph = scene_graphs[split].get(sample_id)
        if graph is None:
            continue
        split_index = int(dataset_info["split_index"])
        gaze_tensor = gaze_tensors[split]
        if split_index >= gaze_tensor.shape[0]:
            continue
        gaze_sequence = gaze_tensor[split_index].mean(dim=1).numpy()
        final_gaze_point = np.asarray(gaze_sequence[-1], dtype=np.float32)
        eye_mid = np.asarray(graph.get("human_final_frame", {}).get("eye_mid", []), dtype=np.float32)
        if eye_mid.shape != (3,):
            eye_mid = final_gaze_point.copy()

        object_lookup = {
            str(object_row.get("object_id", "")).strip(): object_row
            for object_row in graph.get("objects", [])
        }
        gt_row = next((row for row in rows if str(row.get("is_gt", "")).strip() == "1"), None)
        final_row = next((row for row in rows if str(row.get("is_final", "")).strip() == "1"), None)
        gt_object_id = str(gt_row.get("object_id", "")).strip() if gt_row else ""
        gt_label = str(gt_row.get("name", "")).strip() if gt_row else ""
        original_object_id = str(final_row.get("object_id", "")).strip() if final_row else ""
        want = str(rows[0].get("want", "")).strip()

        candidates: list[CandidateRecord] = []
        for row in rows:
            object_id = str(row.get("object_id", "")).strip()
            object_row = object_lookup.get(object_id)
            if object_row is None:
                continue
            try:
                gaze_features = candidate_gaze_features(
                    object_row=object_row,
                    eye_mid=eye_mid,
                    final_gaze_point=final_gaze_point,
                    args=args,
                )
            except ValueError:
                continue

            original_score = safe_float(row.get(args.base_score_column), default=NEG_INF_THRESHOLD)
            in_candidate_pool = str(row.get("in_candidate_pool", "")).strip() == "1"
            if original_score <= NEG_INF_THRESHOLD:
                in_candidate_pool = False

            candidates.append(
                CandidateRecord(
                    split=split,
                    sample_id=sample_id,
                    object_id=object_id,
                    label=str(row.get("name", "")).strip(),
                    want=want,
                    gt_object_id=gt_object_id,
                    gt_label=gt_label,
                    is_gt=str(row.get("is_gt", "")).strip() == "1",
                    in_candidate_pool=in_candidate_pool,
                    original_is_final=str(row.get("is_final", "")).strip() == "1",
                    original_score=original_score,
                    semantic_total=safe_float(row.get("semantic_total"), default=0.0),
                    posterior=safe_float(row.get("posterior"), default=0.0),
                    features=gaze_features,
                )
            )

        if not candidates:
            continue
        sample = SampleRecord(
            row_uid=str(dataset_info["row_uid"]),
            split=split,
            split_index=split_index,
            sample_id=sample_id,
            scene_name=str(dataset_info["scene_name"]),
            sequence_name=str(dataset_info["sequence_name"]),
            start_frame=str(dataset_info["start_frame"]),
            end_frame=str(dataset_info["end_frame"]),
            want=want,
            gt_object_id=gt_object_id,
            gt_label=gt_label,
            original_object_id=original_object_id,
            final_gaze_point=final_gaze_point,
            candidates=tuple(candidates),
        )
        sample_cache[sample_key] = sample
        samples.append(sample)
    return samples


def config_from_values(gaze_distance_weight: float, gaze_alignment_weight: float) -> dict[str, float]:
    return {
        "gaze_distance_weight": float(gaze_distance_weight),
        "gaze_alignment_weight": float(gaze_alignment_weight),
    }


def score_candidate(candidate: CandidateRecord, *, config: dict[str, float]) -> dict[str, float]:
    if not candidate.in_candidate_pool:
        return {
            "score_total": NEG_INF_THRESHOLD,
            "score_base": candidate.original_score,
            "score_gaze_distance": 0.0,
            "score_gaze_alignment": 0.0,
        }
    gaze_distance = config["gaze_distance_weight"] * candidate.features["gaze_distance_score"]
    gaze_alignment_score = config["gaze_alignment_weight"] * candidate.features["gaze_alignment_score"]
    total = candidate.original_score + gaze_distance + gaze_alignment_score
    return {
        "score_total": float(total),
        "score_base": float(candidate.original_score),
        "score_gaze_distance": float(gaze_distance),
        "score_gaze_alignment": float(gaze_alignment_score),
    }


def predict_with_config(
    sample: SampleRecord,
    *,
    config: dict[str, float],
) -> tuple[CandidateRecord, list[tuple[CandidateRecord, dict[str, float]]]]:
    scored: list[tuple[CandidateRecord, dict[str, float]]] = []
    for candidate in sample.candidates:
        payload = dict(candidate.features)
        payload.update(score_candidate(candidate, config=config))
        scored.append((candidate, payload))
    scored.sort(
        key=lambda item: (
            -float(item[1]["score_total"]),
            -float(item[1]["score_base"]),
            -float(item[0].posterior),
            item[0].object_id,
        )
    )
    return scored[0][0], scored


def predict_original(sample: SampleRecord) -> CandidateRecord:
    if sample.original_object_id:
        selected = next((candidate for candidate in sample.candidates if candidate.object_id == sample.original_object_id), None)
        if selected is not None:
            return selected
    return predict_with_config(sample, config=config_from_values(0.0, 0.0))[0]


def prediction_key(sample: SampleRecord) -> str:
    return sample.row_uid


def evaluate_predictions(samples: list[SampleRecord], predictions: dict[str, str]) -> dict[str, Any]:
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "test", "all"):
        split_samples = samples if split_name == "all" else [sample for sample in samples if sample.split == split_name]
        total = len(split_samples)
        gt_in_pool = 0
        exact_instance = 0
        strict_label = 0
        relaxed_label = 0
        original_exact = 0
        rescued = 0
        broken = 0
        changed = 0
        same_label_wrong_instance = 0
        for sample in split_samples:
            pred_object_id = predictions.get(prediction_key(sample), "")
            pred_candidate = next((candidate for candidate in sample.candidates if candidate.object_id == pred_object_id), None)
            gt_candidate = next((candidate for candidate in sample.candidates if candidate.object_id == sample.gt_object_id), None)
            original_hit = bool(sample.gt_object_id and sample.original_object_id == sample.gt_object_id)
            pred_hit = bool(sample.gt_object_id and pred_object_id == sample.gt_object_id)
            if gt_candidate is not None and gt_candidate.in_candidate_pool:
                gt_in_pool += 1
            if original_hit:
                original_exact += 1
            if pred_hit:
                exact_instance += 1
            if pred_object_id != sample.original_object_id:
                changed += 1
            if pred_hit and not original_hit:
                rescued += 1
            if original_hit and not pred_hit:
                broken += 1
            pred_label = pred_candidate.label if pred_candidate is not None else ""
            if pred_label and sample.gt_label and pred_label == sample.gt_label:
                strict_label += 1
                if not pred_hit:
                    same_label_wrong_instance += 1
            if normalize_label(pred_label) and normalize_label(pred_label) == normalize_label(sample.gt_label):
                relaxed_label += 1

        metrics_by_split[split_name] = {
            "samples": total,
            "gt_in_candidate_pool": gt_in_pool,
            "exact_instance_match": exact_instance,
            "exact_instance_acc": (exact_instance / total) if total else 0.0,
            "strict_label_match": strict_label,
            "strict_label_acc": (strict_label / total) if total else 0.0,
            "relaxed_label_match": relaxed_label,
            "relaxed_label_acc": (relaxed_label / total) if total else 0.0,
            "original_exact_instance_match": original_exact,
            "changed_from_original": changed,
            "rescued_original_miss": rescued,
            "broke_original_hit": broken,
            "same_label_wrong_instance": same_label_wrong_instance,
        }
    return metrics_by_split


def run_predictions(samples: list[SampleRecord], config: dict[str, float]) -> dict[str, str]:
    return {
        prediction_key(sample): predict_with_config(sample, config=config)[0].object_id
        for sample in samples
    }


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


def build_output_rows(
    samples: list[SampleRecord],
    *,
    config: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for sample in samples:
        selected, ranked = predict_with_config(sample, config=config)
        selected_payload = next(payload for candidate, payload in ranked if candidate.object_id == selected.object_id)
        original_hit = int(sample.original_object_id == sample.gt_object_id)
        selected_hit = int(selected.object_id == sample.gt_object_id)
        for rank, (candidate, payload) in enumerate(ranked, start=1):
            candidate_rows.append(
                {
                    "row_uid": sample.row_uid,
                    "split": sample.split,
                    "split_index": sample.split_index,
                    "sample_id": sample.sample_id,
                    "want": sample.want,
                    "object_id": candidate.object_id,
                    "name": candidate.label,
                    "gt_object_id": sample.gt_object_id,
                    "gt_label": sample.gt_label,
                    "is_gt": int(candidate.is_gt),
                    "in_candidate_pool": int(candidate.in_candidate_pool),
                    "original_is_final": int(candidate.original_is_final),
                    "is_selected": int(candidate.object_id == selected.object_id),
                    "rank": rank,
                    "semantic_total": candidate.semantic_total,
                    "original_score": candidate.original_score,
                    "posterior": candidate.posterior,
                    **payload,
                }
            )
        sample_rows.append(
            {
                "status": "ok",
                "row_uid": sample.row_uid,
                "split": sample.split,
                "split_index": sample.split_index,
                "sample_id": sample.sample_id,
                "scene_name": sample.scene_name,
                "sequence_name": sample.sequence_name,
                "start_frame": sample.start_frame,
                "end_frame": sample.end_frame,
                "want_text": sample.want,
                "num_candidates": len(sample.candidates),
                "selected_object_id": selected.object_id,
                "selected_label": selected.label,
                "gt_object_id": sample.gt_object_id,
                "gt_label": sample.gt_label,
                "original_object_id": sample.original_object_id,
                "exact_instance_match": selected_hit,
                "strict_label_match": int(selected.label == sample.gt_label),
                "relaxed_label_match": int(normalize_label(selected.label) == normalize_label(sample.gt_label)),
                "original_exact_instance_match": original_hit,
                "rescued_original_miss": int(selected_hit and not original_hit),
                "broke_original_hit": int(original_hit and not selected_hit),
                "changed_from_original": int(selected.object_id != sample.original_object_id),
                "selected_score_total": selected_payload["score_total"],
                "selected_score_base": selected_payload["score_base"],
                "selected_score_gaze_distance": selected_payload["score_gaze_distance"],
                "selected_score_gaze_alignment": selected_payload["score_gaze_alignment"],
            }
        )
    return sample_rows, candidate_rows


def main() -> None:
    args = parse_args()
    samples = build_sample_records(args)
    if not samples:
        raise RuntimeError("No valid samples were built for evaluation")
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    original_predictions = {prediction_key(sample): predict_original(sample).object_id for sample in samples}
    original_metrics = evaluate_predictions(samples, original_predictions)

    sweep_rows: list[dict[str, Any]] = []
    best_train_config: dict[str, float] | None = None
    best_train_key: tuple[float, float, float, float, float] | None = None
    best_test_config: dict[str, float] | None = None
    best_test_key: tuple[float, float, float, float, float] | None = None

    for gaze_distance_weight, gaze_alignment_weight in itertools.product(
        parse_float_grid(args.gaze_distance_grid),
        parse_float_grid(args.gaze_alignment_grid),
    ):
        config = config_from_values(gaze_distance_weight, gaze_alignment_weight)
        predictions = run_predictions(samples, config)
        metrics = evaluate_predictions(samples, predictions)
        row = {
            **config,
            "train_exact_instance_match": metrics["train"]["exact_instance_match"],
            "train_exact_instance_acc": metrics["train"]["exact_instance_acc"],
            "train_samples": metrics["train"]["samples"],
            "train_rescued_original_miss": metrics["train"]["rescued_original_miss"],
            "train_broke_original_hit": metrics["train"]["broke_original_hit"],
            "test_exact_instance_match": metrics["test"]["exact_instance_match"],
            "test_exact_instance_acc": metrics["test"]["exact_instance_acc"],
            "test_samples": metrics["test"]["samples"],
            "test_rescued_original_miss": metrics["test"]["rescued_original_miss"],
            "test_broke_original_hit": metrics["test"]["broke_original_hit"],
            "all_exact_instance_match": metrics["all"]["exact_instance_match"],
            "all_exact_instance_acc": metrics["all"]["exact_instance_acc"],
            "all_samples": metrics["all"]["samples"],
            "all_rescued_original_miss": metrics["all"]["rescued_original_miss"],
            "all_broke_original_hit": metrics["all"]["broke_original_hit"],
            "all_changed_from_original": metrics["all"]["changed_from_original"],
            "all_relaxed_label_match": metrics["all"]["relaxed_label_match"],
            "all_relaxed_label_acc": metrics["all"]["relaxed_label_acc"],
        }
        sweep_rows.append(row)

        train_key = (
            metrics["train"]["exact_instance_acc"],
            metrics["train"]["rescued_original_miss"] - metrics["train"]["broke_original_hit"],
            metrics["test"]["exact_instance_acc"],
            metrics["all"]["exact_instance_acc"],
            -gaze_distance_weight - gaze_alignment_weight,
        )
        test_key = (
            metrics["test"]["exact_instance_acc"],
            metrics["test"]["rescued_original_miss"] - metrics["test"]["broke_original_hit"],
            metrics["train"]["exact_instance_acc"],
            metrics["all"]["exact_instance_acc"],
            -gaze_distance_weight - gaze_alignment_weight,
        )
        if best_train_key is None or train_key > best_train_key:
            best_train_key = train_key
            best_train_config = config
        if best_test_key is None or test_key > best_test_key:
            best_test_key = test_key
            best_test_config = config

    if best_train_config is None or best_test_config is None:
        raise RuntimeError("Failed to select best configs")

    best_train_predictions = run_predictions(samples, best_train_config)
    best_test_predictions = run_predictions(samples, best_test_config)
    best_train_metrics = evaluate_predictions(samples, best_train_predictions)
    best_test_metrics = evaluate_predictions(samples, best_test_predictions)
    sample_rows, candidate_rows = build_output_rows(samples, config=best_train_config)

    sweep_rows.sort(
        key=lambda row: (
            float(row["train_exact_instance_acc"]),
            int(row["train_rescued_original_miss"]) - int(row["train_broke_original_hit"]),
            float(row["test_exact_instance_acc"]),
            float(row["all_exact_instance_acc"]),
        ),
        reverse=True,
    )
    top_train_configs = sweep_rows[:20]
    top_test_configs = sorted(
        sweep_rows,
        key=lambda row: (
            float(row["test_exact_instance_acc"]),
            int(row["test_rescued_original_miss"]) - int(row["test_broke_original_hit"]),
            float(row["train_exact_instance_acc"]),
            float(row["all_exact_instance_acc"]),
        ),
        reverse=True,
    )[:20]

    sweep_csv = args.analysis_dir / f"{ARTIFACT_PREFIX}_sweep_results.csv"
    sample_csv = args.analysis_dir / f"{ARTIFACT_PREFIX}_best_train_config_sample_summary.csv"
    candidate_csv = args.analysis_dir / f"{ARTIFACT_PREFIX}_best_train_config_candidate_scores.csv"
    selector_csv = args.analysis_dir / f"{ARTIFACT_PREFIX}_selector_summary.csv"
    summary_path = args.analysis_dir / f"{ARTIFACT_PREFIX}_summary.json"
    write_csv(sweep_csv, sweep_rows)
    write_csv(sample_csv, sample_rows)
    write_csv(candidate_csv, candidate_rows)
    write_csv(selector_csv, sample_rows)

    summary = {
        "analysis_dir": str(args.analysis_dir),
        "teacher_candidate_csv": str(args.teacher_candidate_csv),
        "base_score_column": args.base_score_column,
        "num_samples": len(samples),
        "num_unique_split_sample_ids": len({(sample.split, sample.sample_id) for sample in samples}),
        "num_duplicate_dataset_rows": len(samples) - len({(sample.split, sample.sample_id) for sample in samples}),
        "feature_formulae": {
            "base_score": f"original teacher {args.base_score_column}",
            "gaze_bbox_score": f"exp(-bbox_distance(final_gaze_point, object_bbox) / {args.gaze_bbox_scale})",
            "gaze_centroid_score": f"exp(-distance(final_gaze_point, object_centroid) / {args.gaze_centroid_scale})",
            "gaze_distance_score": f"gaze_bbox_score + {args.gaze_centroid_weight} * gaze_centroid_score",
            "gaze_alignment_score": "((cos(eye_mid->final_gaze, eye_mid->object_centroid) + 1) / 2)",
            "new_score": "base_score + gaze_distance_weight * gaze_distance_score + gaze_alignment_weight * gaze_alignment_score",
        },
        "notes": {
            "llm_want_aware_score_reused": True,
            "motion_fusion_score_reused": True,
            "candidate_pool_mask_preserved": True,
            "bbox_bonus_used": False,
            "dataset_rows_preserved": True,
            "duplicate_sample_rows_reuse_first_selection": True,
            "prediction_key": "row_uid = split|split_index|sample_id",
        },
        "original_teacher": original_metrics,
        "best_train_config": {
            "config": best_train_config,
            "metrics": best_train_metrics,
        },
        "best_test_config_for_analysis_only": {
            "config": best_test_config,
            "metrics": best_test_metrics,
        },
        "top_train_configs": top_train_configs,
        "top_test_configs_for_analysis_only": top_test_configs,
        "artifacts": {
            "summary_json": str(summary_path),
            "sweep_csv": str(sweep_csv),
            "best_train_sample_csv": str(sample_csv),
            "best_train_candidate_csv": str(candidate_csv),
            "selector_summary_csv": str(selector_csv),
        },
    }
    write_json(summary_path, summary)
    write_json(args.analysis_dir / f"{ARTIFACT_PREFIX}_best_train_predictions.json", best_train_predictions)
    write_json(args.analysis_dir / f"{ARTIFACT_PREFIX}_best_test_predictions_for_analysis_only.json", best_test_predictions)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
