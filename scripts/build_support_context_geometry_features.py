#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "candidate_scores.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_support_context_geometry.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"

SUPPORT_CONTEXT_COLUMNS = (
    "support_parent_log_count",
    "support_parent_overlap_sum",
    "support_parent_overlap_max",
    "support_parent_score_sum",
    "support_parent_score_max",
    "support_parent_gap_abs_min",
    "support_parent_size_ratio_log_max",
    "support_child_log_count",
    "support_child_overlap_sum",
    "support_child_overlap_max",
    "support_child_score_sum",
    "support_child_score_max",
    "support_child_gap_abs_min",
    "support_child_volume_ratio_sum",
    "support_role_parent_minus_child_score",
    "support_role_child_minus_parent_score",
    "support_role_has_parent",
    "support_role_has_child",
    "support_role_is_stack_middle",
    "support_role_is_isolated",
    "support_floor_gap",
    "support_floor_contact",
    "support_floor_contact_x_child_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append label-free support-context geometry features. These features use only "
            "static per-sample object bbox relations from scene_graphs: no want/action, "
            "semantic labels, category names, future frames, GT, or teacher outputs."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--vertical-gap-low", type=float, default=-0.12)
    parser.add_argument("--vertical-gap-high", type=float, default=0.30)
    parser.add_argument("--min-top-overlap", type=float, default=0.04)
    parser.add_argument("--min-bottom-overlap", type=float, default=0.005)
    parser.add_argument("--floor-contact-margin", type=float, default=0.08)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vector3(values: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def bbox_info(obj: dict[str, Any]) -> dict[str, float]:
    bbox_min = vector3(obj.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(obj.get("bbox_max"), name="bbox_max")
    low_x, low_y, low_z = (min(a, b) for a, b in zip(bbox_min, bbox_max, strict=True))
    high_x, high_y, high_z = (max(a, b) for a, b in zip(bbox_min, bbox_max, strict=True))
    size_x = max(high_x - low_x, 1.0e-6)
    size_y = max(high_y - low_y, 1.0e-6)
    size_z = max(high_z - low_z, 1.0e-6)
    return {
        "low_x": low_x,
        "low_y": low_y,
        "low_z": low_z,
        "high_x": high_x,
        "high_y": high_y,
        "high_z": high_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "xz_area": size_x * size_z,
        "volume": size_x * size_y * size_z,
    }


def xz_intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    overlap_x = max(0.0, min(a["high_x"], b["high_x"]) - max(a["low_x"], b["low_x"]))
    overlap_z = max(0.0, min(a["high_z"], b["high_z"]) - max(a["low_z"], b["low_z"]))
    return overlap_x * overlap_z


def maybe_support_edge(
    *,
    top: dict[str, float],
    bottom: dict[str, float],
    vertical_gap_low: float,
    vertical_gap_high: float,
    min_top_overlap: float,
    min_bottom_overlap: float,
) -> dict[str, float] | None:
    inter_area = xz_intersection_area(top, bottom)
    if inter_area <= 0.0:
        return None
    top_overlap = inter_area / max(top["xz_area"], 1.0e-8)
    bottom_overlap = inter_area / max(bottom["xz_area"], 1.0e-8)
    if top_overlap < min_top_overlap or bottom_overlap < min_bottom_overlap:
        return None
    vertical_gap = top["low_y"] - bottom["high_y"]
    if not (vertical_gap_low <= vertical_gap <= vertical_gap_high):
        return None
    gap_abs = abs(vertical_gap)
    size_ratio_log = math.log(bottom["volume"] / max(top["volume"], 1.0e-8))
    score = top_overlap * bottom_overlap * math.log1p(max(size_ratio_log, 0.0)) / (1.0 + gap_abs)
    return {
        "top_overlap": top_overlap,
        "bottom_overlap": bottom_overlap,
        "gap_abs": gap_abs,
        "size_ratio_log": size_ratio_log,
        "score": score,
        "top_to_bottom_volume_ratio": top["volume"] / max(bottom["volume"], 1.0e-8),
    }


def zero_features() -> dict[str, float]:
    return {column: 0.0 for column in SUPPORT_CONTEXT_COLUMNS}


def min_or_zero(values: list[float]) -> float:
    return min(values) if values else 0.0


def max_or_zero(values: list[float]) -> float:
    return max(values) if values else 0.0


def build_sample_features(
    objects: list[dict[str, Any]],
    *,
    vertical_gap_low: float,
    vertical_gap_high: float,
    min_top_overlap: float,
    min_bottom_overlap: float,
    floor_contact_margin: float,
) -> dict[str, dict[str, float]]:
    info_by_id = {
        str(obj.get("object_id", "")): bbox_info(obj)
        for obj in objects
        if str(obj.get("object_id", ""))
    }
    if not info_by_id:
        return {}

    sample_floor_y = min(info["low_y"] for info in info_by_id.values())
    parent_edges: dict[str, list[dict[str, float]]] = defaultdict(list)
    child_edges: dict[str, list[dict[str, float]]] = defaultdict(list)

    for top_id, top_info in info_by_id.items():
        for bottom_id, bottom_info in info_by_id.items():
            if top_id == bottom_id:
                continue
            edge = maybe_support_edge(
                top=top_info,
                bottom=bottom_info,
                vertical_gap_low=vertical_gap_low,
                vertical_gap_high=vertical_gap_high,
                min_top_overlap=min_top_overlap,
                min_bottom_overlap=min_bottom_overlap,
            )
            if edge is None:
                continue
            parent_edges[top_id].append(edge | {"bottom_id": bottom_id})
            child_edges[bottom_id].append(edge | {"top_id": top_id})

    output: dict[str, dict[str, float]] = {}
    for object_id, info in info_by_id.items():
        parents = parent_edges.get(object_id, [])
        children = child_edges.get(object_id, [])
        parent_scores = [edge["score"] for edge in parents]
        child_scores = [edge["score"] for edge in children]
        parent_score_sum = sum(parent_scores)
        child_score_sum = sum(child_scores)
        has_parent = 1.0 if parents else 0.0
        has_child = 1.0 if children else 0.0
        floor_gap = max(info["low_y"] - sample_floor_y, 0.0)
        floor_contact = 1.0 if floor_gap <= floor_contact_margin else 0.0

        output[object_id] = {
            "support_parent_log_count": math.log1p(len(parents)),
            "support_parent_overlap_sum": sum(edge["top_overlap"] for edge in parents),
            "support_parent_overlap_max": max_or_zero([edge["top_overlap"] for edge in parents]),
            "support_parent_score_sum": parent_score_sum,
            "support_parent_score_max": max_or_zero(parent_scores),
            "support_parent_gap_abs_min": min_or_zero([edge["gap_abs"] for edge in parents]),
            "support_parent_size_ratio_log_max": max_or_zero([edge["size_ratio_log"] for edge in parents]),
            "support_child_log_count": math.log1p(len(children)),
            "support_child_overlap_sum": sum(edge["bottom_overlap"] for edge in children),
            "support_child_overlap_max": max_or_zero([edge["bottom_overlap"] for edge in children]),
            "support_child_score_sum": child_score_sum,
            "support_child_score_max": max_or_zero(child_scores),
            "support_child_gap_abs_min": min_or_zero([edge["gap_abs"] for edge in children]),
            "support_child_volume_ratio_sum": sum(edge["top_to_bottom_volume_ratio"] for edge in children),
            "support_role_parent_minus_child_score": parent_score_sum - child_score_sum,
            "support_role_child_minus_parent_score": child_score_sum - parent_score_sum,
            "support_role_has_parent": has_parent,
            "support_role_has_child": has_child,
            "support_role_is_stack_middle": 1.0 if parents and children else 0.0,
            "support_role_is_isolated": 1.0 if not parents and not children else 0.0,
            "support_floor_gap": floor_gap,
            "support_floor_contact": floor_contact,
            "support_floor_contact_x_child_score": floor_contact * child_score_sum,
        }
    return output


def load_feature_lookup(
    *,
    scene_graph_root: Path,
    vertical_gap_low: float,
    vertical_gap_high: float,
    min_top_overlap: float,
    min_bottom_overlap: float,
    floor_contact_margin: float,
) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    duplicate_mismatches: list[dict[str, Any]] = []
    for split in ("train", "test"):
        path = scene_graph_root / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scene graph file: {path}")
        with path.open("r", encoding="utf-8") as file:
            for sample_index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("sample_id", ""))
                features_by_object = build_sample_features(
                    list(sample.get("objects", [])),
                    vertical_gap_low=vertical_gap_low,
                    vertical_gap_high=vertical_gap_high,
                    min_top_overlap=min_top_overlap,
                    min_bottom_overlap=min_bottom_overlap,
                    floor_contact_margin=floor_contact_margin,
                )
                for object_id, features in features_by_object.items():
                    key = (split, sample_id, object_id)
                    previous = lookup.get(key)
                    if previous is not None:
                        max_diff = max(abs(previous[name] - features[name]) for name in SUPPORT_CONTEXT_COLUMNS)
                        if max_diff > 1.0e-8:
                            duplicate_mismatches.append(
                                {
                                    "split": split,
                                    "sample_id": sample_id,
                                    "object_id": object_id,
                                    "sample_index": sample_index,
                                    "max_abs_diff": max_diff,
                                }
                            )
                        continue
                    lookup[key] = features
    if duplicate_mismatches:
        raise RuntimeError(
            "Duplicate sample ids produced different support-context features: "
            f"{duplicate_mismatches[:5]}"
        )
    return lookup


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    lookup = load_feature_lookup(
        scene_graph_root=args.scene_graph_root,
        vertical_gap_low=float(args.vertical_gap_low),
        vertical_gap_high=float(args.vertical_gap_high),
        min_top_overlap=float(args.min_top_overlap),
        min_bottom_overlap=float(args.min_bottom_overlap),
        floor_contact_margin=float(args.floor_contact_margin),
    )
    default_features = zero_features()
    output_rows: list[dict[str, Any]] = []
    missing_rows = 0
    for row in rows:
        output = dict(row)
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows += 1
            features = default_features
        output.update(features)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in SUPPORT_CONTEXT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"num_rows={len(output_rows)} missing_rows={missing_rows}")
    print("support_context_columns=" + ",".join(SUPPORT_CONTEXT_COLUMNS))


if __name__ == "__main__":
    main()
