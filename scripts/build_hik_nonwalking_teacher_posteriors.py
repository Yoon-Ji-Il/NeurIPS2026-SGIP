#!/usr/bin/env python3
"""Build HIK teacher soft distributions with the legacy motion-C recipe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import iter_jsonl, load_selection_lookup  # noqa: E402


DEFAULT_ALIGNMENT_ROOT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking/SLICES_888s"
)
DEFAULT_DIRECT29_ROOT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking_direct29/SLICES_888s"
)
DEFAULT_FUSION_SUMMARY = Path(
    "./object_selector/"
    "fusion_weight_runs/qwen3_motion_c_selector_weights_final/summary.json"
)
DEFAULT_SEMANTIC_ROOT = (
    REPO_ROOT / "runs_hik" / "qwen3_14b_h25f50_nonwalking_candidate_semantics"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs_hik" / "hik_h25f50_nonwalking_teacher_soft_motion_c"
)

ACTION_TO_LEGACY_WANT = {
    "sitting_down": "sit",
    "whiteboard": "write",
    "sink": "use",
    "cupboard": "open",
    "coffee": "use",
}

STUDENT_FEATURE_COLUMNS = (
    "geometry_score",
    "root_bbox_distance_3d",
    "head_bbox_distance_3d",
    "hand_bbox_distance_3d",
    "root_horizontal_alignment",
    "torso_horizontal_alignment",
    "face_horizontal_alignment",
    "torso_cone_fraction",
    "face_cone_fraction",
    "root_heading_cone_fraction",
    "root_line_distance_xz",
    "torso_line_distance_xz",
    "face_line_distance_xz",
    "label_sittable",
    "label_whiteboard",
    "label_sink",
    "label_cupboard",
    "label_coffee_machine",
    "label_occluder",
    "label_table",
    "label_dishwasher",
    "label_microwave",
    "label_kettle",
    "label_trash",
    "label_drawer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument(
        "--hard-selection-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "qwen3_14b_action_geometry_h25f50_validated_v2",
    )
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--direct29-root", type=Path, default=DEFAULT_DIRECT29_ROOT)
    parser.add_argument("--fusion-summary-json", type=Path, default=DEFAULT_FUSION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--semantic-gap", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minimum-logit-margin", type=float, default=1.0)
    parser.add_argument("--body-sigma", type=float, default=0.6)
    parser.add_argument("--hand-sigma", type=float, default=0.6)
    parser.add_argument("--endpoint-sigma", type=float, default=0.8)
    parser.add_argument("--limit-per-split", type=int, default=-1)
    return parser.parse_args()


def as_tensor3(value: Any) -> torch.Tensor:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return torch.zeros(3, dtype=torch.float32)
    return torch.tensor([float(item) for item in value], dtype=torch.float32)


def bbox_distance(
    points: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> torch.Tensor:
    below = torch.clamp(bbox_min - points, min=0.0)
    above = torch.clamp(points - bbox_max, min=0.0)
    return torch.linalg.norm(below + above, dim=-1)


def score_from_distance(distance: float, sigma: float) -> float:
    return math.exp(-max(float(distance), 0.0) / max(float(sigma), 1.0e-8))


def future_motion_features(
    candidate: dict[str, Any],
    joints_label: torch.Tensor,
    *,
    body_sigma: float,
    hand_sigma: float,
    endpoint_sigma: float,
) -> dict[str, float]:
    bbox_min = as_tensor3(candidate.get("bbox_min"))
    bbox_max = as_tensor3(candidate.get("bbox_max"))
    body_indices = [index for index in (0, 12, 15) if index < joints_label.shape[1]]
    hand_indices = [index for index in (20, 21) if index < joints_label.shape[1]]
    if not hand_indices:
        hand_indices = [index for index in (18, 19) if index < joints_label.shape[1]]
    if not body_indices or not hand_indices:
        raise ValueError(f"Unsupported HIK joint count: {joints_label.shape[1]}")

    body_distance = float(
        bbox_distance(joints_label[:, body_indices], bbox_min, bbox_max).min().item()
    )
    hand_distance = float(
        bbox_distance(joints_label[:, hand_indices], bbox_min, bbox_max).min().item()
    )
    root_endpoint_distance = float(
        bbox_distance(joints_label[-1:, [0]], bbox_min, bbox_max).min().item()
    )
    return {
        "future_body_contact_distance": body_distance,
        "future_hand_contact_distance": hand_distance,
        "future_root_endpoint_distance": root_endpoint_distance,
        "future_body_contact_score": score_from_distance(body_distance, body_sigma),
        "future_hand_contact_score": score_from_distance(hand_distance, hand_sigma),
        "future_root_endpoint_score": score_from_distance(
            root_endpoint_distance,
            endpoint_sigma,
        ),
    }


def geometry_fusion_features(candidate: dict[str, Any]) -> dict[str, float]:
    metrics = candidate.get("metrics", {})
    face_alignment = float(metrics.get("face_horizontal_alignment", 0.0))
    body_alignment = float(metrics.get("torso_horizontal_alignment", 0.0))
    face_line_distance = float(metrics.get("face_line_distance_xz", 0.0))
    return {
        "face_alignment_norm": (max(-1.0, min(1.0, face_alignment)) + 1.0) * 0.5,
        "body_alignment_norm": (max(-1.0, min(1.0, body_alignment)) + 1.0) * 0.5,
        "face_line_score": math.exp(-max(face_line_distance, 0.0) / 0.8),
    }


def student_features(candidate: dict[str, Any]) -> dict[str, float]:
    metrics = candidate.get("metrics", {})
    label = str(candidate.get("label", "")).lower()
    output = {
        column: float(metrics.get(column, 0.0))
        for column in STUDENT_FEATURE_COLUMNS
        if not column.startswith("label_")
    }
    for column in STUDENT_FEATURE_COLUMNS:
        if column.startswith("label_"):
            output[column] = float(label == column.removeprefix("label_"))
    return output


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def read_semantic_lookup(root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = root / split / "semantic_scores.jsonl"
    rows = list(iter_jsonl(path))
    lookup = {str(row["sample_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate semantic rows in {path}")
    return lookup


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.minimum_logit_margin <= 0:
        raise ValueError("--minimum-logit-margin must be > 0")
    fusion_summary = json.loads(args.fusion_summary_json.read_text(encoding="utf-8"))
    legacy_weights = fusion_summary["weights_by_want"]

    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    anchored_count = 0

    for split in ("train", "test"):
        manifest_path = args.alignment_root / split / "hik_h25f50_cache_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.limit_per_split >= 0:
            manifest = manifest[: args.limit_per_split]
        if any(str(row.get("event_action", "")) == "walking" for row in manifest):
            raise ValueError(f"Walking row found in {manifest_path}")

        graph_lookup = {
            str(graph["sample_id"]): graph
            for graph in iter_jsonl(args.scene_graph_root / split / "samples.jsonl")
        }
        semantic_lookup = read_semantic_lookup(args.semantic_root, split)
        hard_lookup = load_selection_lookup(
            args.hard_selection_root / split / "selections.jsonl"
        )
        direct_manifest = json.loads(
            (
                args.direct29_root / split / "hik_direct29_cache_manifest.json"
            ).read_text(encoding="utf-8")
        )
        direct_ids = [str(row["sample_id"]) for row in direct_manifest]
        context_ids = [str(row["sample_id"]) for row in json.loads(manifest_path.read_text())]
        if direct_ids != context_ids:
            raise ValueError(f"Direct29/context manifest mismatch for {split}")
        joints_label = torch.load(
            args.direct29_root / split / "joints29_label.pth",
            map_location="cpu",
            weights_only=False,
        ).float()
        if joints_label.shape != (len(direct_ids), 50, 29, 3):
            raise ValueError(
                f"Unexpected joints29_label shape for {split}: {tuple(joints_label.shape)}"
            )
        direct_index = {sample_id: index for index, sample_id in enumerate(direct_ids)}

        for split_index, manifest_row in enumerate(manifest):
            sample_id = str(manifest_row["sample_id"])
            graph = graph_lookup.get(sample_id)
            semantic = semantic_lookup.get(sample_id)
            hard = hard_lookup.get(sample_id)
            if graph is None or semantic is None or hard is None:
                raise KeyError(
                    f"Missing aligned input for {split}/{sample_id}: "
                    f"graph={graph is not None}, semantic={semantic is not None}, "
                    f"hard={hard is not None}"
                )
            if str(graph.get("event_action", "")) == "walking":
                raise ValueError(f"Walking graph reached posterior builder: {sample_id}")
            action = str(graph["event_action"])
            legacy_want = ACTION_TO_LEGACY_WANT.get(action)
            if legacy_want is None:
                raise KeyError(f"No legacy weight mapping for HIK action {action!r}")
            weights = legacy_weights[legacy_want]
            semantic_by_id = {
                str(row["object_id"]): row
                for row in semantic.get("semantic_scores", [])
            }
            candidates = sorted(
                list(graph.get("objects", [])),
                key=lambda row: str(row.get("object_id", "")),
            )
            candidate_ids = [str(row.get("object_id", "")) for row in candidates]
            if set(candidate_ids) != set(semantic_by_id):
                raise ValueError(f"Semantic candidate mismatch for {sample_id}")
            hard_id = str(hard.get("selected_object_id", ""))
            if hard_id not in set(candidate_ids):
                raise ValueError(f"Hard teacher object absent for {sample_id}: {hard_id}")

            max_semantic = max(
                float(semantic_by_id[object_id]["semantic_total"])
                for object_id in candidate_ids
            )
            per_candidate: list[dict[str, Any]] = []
            for candidate in candidates:
                object_id = str(candidate["object_id"])
                sem = semantic_by_id[object_id]
                geometry = geometry_fusion_features(candidate)
                motion = future_motion_features(
                    candidate,
                    joints_label[direct_index[sample_id]],
                    body_sigma=args.body_sigma,
                    hand_sigma=args.hand_sigma,
                    endpoint_sigma=args.endpoint_sigma,
                )
                features = {
                    "semantic_total": float(sem["semantic_total"]),
                    **geometry,
                    "future_hand_contact_score": motion[
                        "future_hand_contact_score"
                    ],
                    "future_root_endpoint_score": motion[
                        "future_root_endpoint_score"
                    ],
                }
                components = {
                    key: float(features[key]) * float(weights[key])
                    for key in (
                        "semantic_total",
                        "face_alignment_norm",
                        "body_alignment_norm",
                        "face_line_score",
                        "future_hand_contact_score",
                        "future_root_endpoint_score",
                    )
                }
                in_pool = (
                    max_semantic - float(sem["semantic_total"])
                    <= args.semantic_gap + 1.0e-8
                ) or object_id == hard_id
                raw_logit = sum(components.values())
                per_candidate.append(
                    {
                        "candidate": candidate,
                        "semantic": sem,
                        "geometry": geometry,
                        "motion": motion,
                        "features": features,
                        "components": components,
                        "in_pool": in_pool,
                        "raw_logit": raw_logit,
                        "hard_logit": raw_logit if in_pool else -1.0e9,
                    }
                )

            hard_index = candidate_ids.index(hard_id)
            valid_other_logits = [
                item["hard_logit"]
                for index, item in enumerate(per_candidate)
                if index != hard_index and item["in_pool"]
            ]
            anchor_applied = False
            if valid_other_logits:
                max_other = max(valid_other_logits)
                if per_candidate[hard_index]["hard_logit"] <= max_other:
                    per_candidate[hard_index]["hard_logit"] = (
                        max_other + args.minimum_logit_margin
                    )
                    anchor_applied = True
                    anchored_count += 1
            pool_indices = [
                index for index, item in enumerate(per_candidate) if item["in_pool"]
            ]
            probabilities = softmax(
                [per_candidate[index]["hard_logit"] for index in pool_indices],
                args.temperature,
            )
            posterior = [0.0] * len(per_candidate)
            for index, probability in zip(pool_indices, probabilities, strict=True):
                posterior[index] = probability
            selected_index = max(range(len(posterior)), key=posterior.__getitem__)
            if candidate_ids[selected_index] != hard_id:
                raise AssertionError(
                    f"Hard-consistent posterior failed for {sample_id}: "
                    f"{candidate_ids[selected_index]} != {hard_id}"
                )

            for index, item in enumerate(per_candidate):
                candidate = item["candidate"]
                sem = item["semantic"]
                row = {
                    "split": split,
                    "split_index": split_index,
                    "sample_id": sample_id,
                    "event_action": action,
                    "legacy_want": legacy_want,
                    "object_id": candidate["object_id"],
                    "name": candidate.get("label", ""),
                    "category": candidate.get("label", ""),
                    "match": sem["match"],
                    "ease": sem["ease"],
                    "semantic_total": sem["semantic_total"],
                    "reasoning_trace": sem.get("reasoning_trace", ""),
                    **item["geometry"],
                    **item["motion"],
                    **student_features(candidate),
                    "raw_fusion_total": item["raw_logit"],
                    "masked_fusion_total": (
                        item["raw_logit"] if item["in_pool"] else -1.0e9
                    ),
                    "teacher_logit_hard_consistent": item["hard_logit"],
                    "posterior": posterior[index],
                    "in_candidate_pool": int(item["in_pool"]),
                    "hard_teacher_object_id": hard_id,
                    "is_hard_teacher_selected": int(index == hard_index),
                    "hard_anchor_applied": int(anchor_applied),
                    "is_gt": 0,
                }
                for key, value in item["components"].items():
                    row[f"component_{key}"] = value
                    row[f"weight_{key}"] = weights[key]
                candidate_rows.append(row)

            sample_rows.append(
                {
                    "split": split,
                    "split_index": split_index,
                    "sample_id": sample_id,
                    "event_action": action,
                    "num_candidates": len(candidates),
                    "hard_teacher_object_id": hard_id,
                    "posterior_top1_object_id": candidate_ids[selected_index],
                    "posterior_top1_probability": posterior[selected_index],
                    "hard_anchor_applied": int(anchor_applied),
                }
            )
        split_counts[split] = len(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "teacher_candidate_scores.csv"
    summary_path = args.output_dir / "teacher_selection_summary.csv"
    write_csv(candidate_path, candidate_rows)
    write_csv(summary_path, sample_rows)
    summary = {
        "method": "legacy_match_ease_motion_c_fusion_hard_consistent",
        "walking_excluded": True,
        "alignment_root": str(args.alignment_root.resolve()),
        "scene_graph_root": str(args.scene_graph_root.resolve()),
        "semantic_root": str(args.semantic_root.resolve()),
        "hard_selection_root": str(args.hard_selection_root.resolve()),
        "direct29_root": str(args.direct29_root.resolve()),
        "fusion_summary_json": str(args.fusion_summary_json.resolve()),
        "action_to_legacy_want": ACTION_TO_LEGACY_WANT,
        "semantic_gap": args.semantic_gap,
        "temperature": args.temperature,
        "minimum_logit_margin": args.minimum_logit_margin,
        "split_counts": split_counts,
        "candidate_rows": len(candidate_rows),
        "anchored_samples": anchored_count,
        "posterior_argmax_matches_hard_teacher": len(sample_rows),
        "student_feature_columns": list(STUDENT_FEATURE_COLUMNS),
        "student_future_features_used": False,
        "teacher_future_features": [
            "future_hand_contact_score",
            "future_root_endpoint_score",
        ],
        "candidate_csv": str(candidate_path.resolve()),
        "selection_summary_csv": str(summary_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
