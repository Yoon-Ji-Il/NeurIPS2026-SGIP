#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_geometry.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_bbox_neighborhood.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"

OUTPUT_COLUMNS = (
    "object_bbox_nearest_center_distance_xz",
    "object_bbox_neighbor_r050_log_count",
    "object_bbox_neighbor_r100_log_count",
    "object_bbox_neighbor_r150_log_count",
    "object_bbox_xz_overlap_log_count",
    "object_bbox_xz_overlap_area_ratio",
    "object_bbox_xz_iou_max",
    "object_bbox_xz_contained_by_larger_max_fraction",
    "object_bbox_xz_contains_smaller_log_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append label-free bbox neighborhood/clutter features. "
            "Uses only per-sample static object bbox geometry; no want, future, GT, teacher rank, "
            "category, or semantic label."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
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


def vector3(values: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def xz_box(obj: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox_min = vector3(obj.get("bbox_min"), name="bbox_min")
    bbox_max = vector3(obj.get("bbox_max"), name="bbox_max")
    x0 = min(bbox_min[0], bbox_max[0])
    x1 = max(bbox_min[0], bbox_max[0])
    z0 = min(bbox_min[2], bbox_max[2])
    z1 = max(bbox_min[2], bbox_max[2])
    return x0, z0, x1, z1


def box_area(box: tuple[float, float, float, float]) -> float:
    x0, z0, x1, z1 = box
    return max(x1 - x0, 0.0) * max(z1 - z0, 0.0)


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, z0, x1, z1 = box
    return (0.5 * (x0 + x1), 0.5 * (z0 + z1))


def intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, lz0, lx1, lz1 = left
    rx0, rz0, rx1, rz1 = right
    ix = max(min(lx1, rx1) - max(lx0, rx0), 0.0)
    iz = max(min(lz1, rz1) - max(lz0, rz0), 0.0)
    return ix * iz


def compute_sample_features(objects: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    boxes: list[tuple[str, tuple[float, float, float, float], float, tuple[float, float]]] = []
    for obj in objects:
        object_id = str(obj.get("object_id", ""))
        if not object_id:
            continue
        box = xz_box(obj)
        boxes.append((object_id, box, box_area(box), box_center(box)))

    output: dict[str, dict[str, float]] = {}
    for index, (object_id, box, area, center) in enumerate(boxes):
        nearest = math.inf
        neighbor_counts = {0.5: 0, 1.0: 0, 1.5: 0}
        overlap_count = 0
        overlap_area_sum = 0.0
        max_iou = 0.0
        contained_by_larger = 0.0
        contains_smaller = 0

        for other_index, (_, other_box, other_area, other_center) in enumerate(boxes):
            if other_index == index:
                continue
            distance = math.hypot(center[0] - other_center[0], center[1] - other_center[1])
            nearest = min(nearest, distance)
            for radius in neighbor_counts:
                if distance <= radius:
                    neighbor_counts[radius] += 1

            overlap = intersection_area(box, other_box)
            if overlap > 0.0:
                overlap_count += 1
                overlap_area_sum += overlap
                union = area + other_area - overlap
                max_iou = max(max_iou, overlap / max(union, 1.0e-8))
                own_fraction = overlap / max(area, 1.0e-8)
                other_fraction = overlap / max(other_area, 1.0e-8)
                if other_area >= area:
                    contained_by_larger = max(contained_by_larger, own_fraction)
                if area > other_area and other_fraction >= 0.5:
                    contains_smaller += 1

        if not math.isfinite(nearest):
            nearest = 0.0
        output[object_id] = {
            "object_bbox_nearest_center_distance_xz": nearest,
            "object_bbox_neighbor_r050_log_count": math.log1p(neighbor_counts[0.5]),
            "object_bbox_neighbor_r100_log_count": math.log1p(neighbor_counts[1.0]),
            "object_bbox_neighbor_r150_log_count": math.log1p(neighbor_counts[1.5]),
            "object_bbox_xz_overlap_log_count": math.log1p(overlap_count),
            "object_bbox_xz_overlap_area_ratio": overlap_area_sum / max(area, 1.0e-8),
            "object_bbox_xz_iou_max": max_iou,
            "object_bbox_xz_contained_by_larger_max_fraction": contained_by_larger,
            "object_bbox_xz_contains_smaller_log_count": math.log1p(contains_smaller),
        }
    return output


def load_feature_lookup(scene_graph_root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
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
                sample_features = compute_sample_features(sample.get("objects", []))
                for object_id, features in sample_features.items():
                    key = (split, sample_id, object_id)
                    previous = lookup.get(key)
                    if previous is not None:
                        max_abs_diff = max(abs(previous[name] - features[name]) for name in OUTPUT_COLUMNS)
                        if max_abs_diff > 1.0e-8:
                            duplicate_mismatches.append(
                                {
                                    "split": split,
                                    "sample_id": sample_id,
                                    "object_id": object_id,
                                    "sample_index": sample_index,
                                    "max_abs_diff": max_abs_diff,
                                }
                            )
                        continue
                    lookup[key] = features
    if duplicate_mismatches:
        raise RuntimeError(
            "Duplicate sample ids produced different bbox-neighborhood features: "
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

    lookup = load_feature_lookup(args.scene_graph_root)
    output_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    zero_features = {column: 0.0 for column in OUTPUT_COLUMNS}
    for row in rows:
        output = dict(row)
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows.append({"split": key[0], "sample_id": key[1], "object_id": key[2]})
            features = zero_features
        output.update(features)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"features={','.join(OUTPUT_COLUMNS)}")
    print(f"missing_rows={len(missing_rows)}")
    print(f"num_rows={len(output_rows)}")


if __name__ == "__main__":
    main()
