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
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_bbox_aspect_extras.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_support_rank_features.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"

BASELINE_7D_COLUMNS = (
    "root_alignment",
    "torso_alignment",
    "root_torso_alignment",
    "face_normal_alignment",
    "face_normal_line_distance_xz",
    "scene32768_expanded_bbox_1p1_fraction",
    "object_shape_bbox_aspect_max_over_min",
)

LOWER_BETTER_COLUMNS = {
    "face_normal_line_distance_xz",
}

SUPPORT_COLUMNS = (
    "object_support_on_larger_overlap_max",
    "object_support_on_larger_size_ratio_log",
    "object_support_on_larger_vertical_gap_abs_min",
    "object_support_on_larger_score",
    "object_support_child_on_top_log_count",
    "object_support_child_on_top_overlap_sum",
    "object_support_child_on_top_volume_ratio_sum",
    "object_support_max_child_score",
)

RELATIVE_COLUMNS = tuple(
    output_column
    for column in BASELINE_7D_COLUMNS
    for output_column in (
        f"rel_{column}_rank_best1",
        f"rel_{column}_zscore",
    )
)

OUTPUT_COLUMNS = SUPPORT_COLUMNS + RELATIVE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair support/on-top bbox-relation features and per-sample relative "
            "rank/zscore versions of the 7D baseline features. The new features use only "
            "observed/static object geometry and already-allowed raw features."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--vertical-gap-low", type=float, default=-0.15)
    parser.add_argument("--vertical-gap-high", type=float, default=0.35)
    parser.add_argument("--larger-volume-ratio", type=float, default=1.5)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
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


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def vector3(values: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def bbox_info(obj: dict[str, Any]) -> dict[str, float]:
    bbox_min = vector3(obj.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(obj.get("bbox_max"), name="bbox_max")
    low_x, low_y, low_z = (min(a, b) for a, b in zip(bbox_min, bbox_max, strict=True))
    high_x, high_y, high_z = (max(a, b) for a, b in zip(bbox_min, bbox_max, strict=True))
    sx = max(high_x - low_x, 1.0e-6)
    sy = max(high_y - low_y, 1.0e-6)
    sz = max(high_z - low_z, 1.0e-6)
    return {
        "low_x": low_x,
        "low_y": low_y,
        "low_z": low_z,
        "high_x": high_x,
        "high_y": high_y,
        "high_z": high_z,
        "size_x": sx,
        "size_y": sy,
        "size_z": sz,
        "xz_area": sx * sz,
        "volume": sx * sy * sz,
    }


def xz_intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    overlap_x = max(0.0, min(a["high_x"], b["high_x"]) - max(a["low_x"], b["low_x"]))
    overlap_z = max(0.0, min(a["high_z"], b["high_z"]) - max(a["low_z"], b["low_z"]))
    return overlap_x * overlap_z


def support_edge_features(
    *,
    top: dict[str, float],
    bottom: dict[str, float],
    vertical_gap_low: float,
    vertical_gap_high: float,
    larger_volume_ratio: float,
    min_overlap_frac: float,
) -> dict[str, float] | None:
    if bottom["volume"] < top["volume"] * larger_volume_ratio:
        return None
    inter_area = xz_intersection_area(top, bottom)
    top_overlap = inter_area / max(top["xz_area"], 1.0e-8)
    bottom_overlap = inter_area / max(bottom["xz_area"], 1.0e-8)
    if top_overlap < min_overlap_frac:
        return None
    vertical_gap = top["low_y"] - bottom["high_y"]
    if not (vertical_gap_low <= vertical_gap <= vertical_gap_high):
        return None
    size_ratio_log = math.log(bottom["volume"] / max(top["volume"], 1.0e-8))
    gap_abs = abs(vertical_gap)
    score = top_overlap * math.log1p(max(size_ratio_log, 0.0)) / (1.0 + gap_abs)
    return {
        "top_overlap": top_overlap,
        "bottom_overlap": bottom_overlap,
        "size_ratio_log": size_ratio_log,
        "gap_abs": gap_abs,
        "score": score,
        "top_to_bottom_volume_ratio": top["volume"] / max(bottom["volume"], 1.0e-8),
    }


def load_scene_graph_objects(scene_graph_root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        path = scene_graph_root / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scene graph file: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                output[(split, str(payload["sample_id"]))] = list(payload.get("objects", []))
    return output


def build_support_lookup(
    scene_objects: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    vertical_gap_low: float,
    vertical_gap_high: float,
    larger_volume_ratio: float,
    min_overlap_frac: float,
) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for (split, sample_id), objects in scene_objects.items():
        info_by_id = {str(obj["object_id"]): bbox_info(obj) for obj in objects}
        rows_by_id: dict[str, dict[str, float]] = {
            object_id: {column: 0.0 for column in SUPPORT_COLUMNS}
            for object_id in info_by_id
        }
        rows_children: dict[str, list[dict[str, float]]] = defaultdict(list)

        for top_id, top_info in info_by_id.items():
            top_edges: list[dict[str, float]] = []
            for bottom_id, bottom_info in info_by_id.items():
                if top_id == bottom_id:
                    continue
                edge = support_edge_features(
                    top=top_info,
                    bottom=bottom_info,
                    vertical_gap_low=vertical_gap_low,
                    vertical_gap_high=vertical_gap_high,
                    larger_volume_ratio=larger_volume_ratio,
                    min_overlap_frac=min_overlap_frac,
                )
                if edge is None:
                    continue
                top_edges.append(edge | {"bottom_id": bottom_id})
                rows_children[bottom_id].append(edge | {"top_id": top_id})

            if top_edges:
                best_overlap = max(edge["top_overlap"] for edge in top_edges)
                best_size_ratio = max(edge["size_ratio_log"] for edge in top_edges)
                best_score = max(edge["score"] for edge in top_edges)
                min_gap = min(edge["gap_abs"] for edge in top_edges)
                rows_by_id[top_id].update(
                    {
                        "object_support_on_larger_overlap_max": best_overlap,
                        "object_support_on_larger_size_ratio_log": best_size_ratio,
                        "object_support_on_larger_vertical_gap_abs_min": min_gap,
                        "object_support_on_larger_score": best_score,
                    }
                )

        for bottom_id, edges in rows_children.items():
            if not edges:
                continue
            rows_by_id[bottom_id].update(
                {
                    "object_support_child_on_top_log_count": math.log1p(len(edges)),
                    "object_support_child_on_top_overlap_sum": sum(edge["bottom_overlap"] for edge in edges),
                    "object_support_child_on_top_volume_ratio_sum": sum(
                        edge["top_to_bottom_volume_ratio"] for edge in edges
                    ),
                    "object_support_max_child_score": max(edge["score"] for edge in edges),
                }
            )

        for object_id, features in rows_by_id.items():
            lookup[(split, sample_id, object_id)] = features
    return lookup


def sample_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("split", ""), row.get("sample_id", ""))


def build_relative_feature_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[sample_key(row)].append((index, row))

    for group in grouped.values():
        group_size = len(group)
        for column in BASELINE_7D_COLUMNS:
            values = [parse_float(row.get(column)) for _, row in group]
            mean = sum(values) / max(len(values), 1)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            std = math.sqrt(max(variance, 0.0))

            reverse = column not in LOWER_BETTER_COLUMNS
            ordered = sorted(
                range(group_size),
                key=lambda local_index: (values[local_index], -local_index if reverse else local_index),
                reverse=reverse,
            )
            rank_scores = [1.0 for _ in group]
            if group_size > 1:
                for rank, local_index in enumerate(ordered):
                    rank_scores[local_index] = 1.0 - rank / (group_size - 1)

            for local_index, (global_index, _) in enumerate(group):
                output.setdefault(global_index, {})
                output[global_index][f"rel_{column}_rank_best1"] = rank_scores[local_index]
                output[global_index][f"rel_{column}_zscore"] = (
                    (values[local_index] - mean) / max(std, 1.0e-6)
                )
    return output


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    scene_objects = load_scene_graph_objects(args.scene_graph_root)
    support_lookup = build_support_lookup(
        scene_objects,
        vertical_gap_low=float(args.vertical_gap_low),
        vertical_gap_high=float(args.vertical_gap_high),
        larger_volume_ratio=float(args.larger_volume_ratio),
        min_overlap_frac=float(args.min_overlap_frac),
    )
    relative_lookup = build_relative_feature_rows(rows)

    output_rows: list[dict[str, Any]] = []
    missing_support = 0
    for index, row in enumerate(rows):
        output = dict(row)
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        support_features = support_lookup.get(key)
        if support_features is None:
            missing_support += 1
            support_features = {column: 0.0 for column in SUPPORT_COLUMNS}
        for column in SUPPORT_COLUMNS:
            output[column] = support_features.get(column, 0.0)
        for column in RELATIVE_COLUMNS:
            output[column] = relative_lookup.get(index, {}).get(column, 0.0)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"num_rows={len(output_rows)} missing_support_rows={missing_support}")
    print("support_columns=" + ",".join(SUPPORT_COLUMNS))
    print("relative_columns=" + ",".join(RELATIVE_COLUMNS))


if __name__ == "__main__":
    main()
