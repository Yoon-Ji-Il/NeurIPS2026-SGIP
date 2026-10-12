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
    / "teacher_candidate_scores_scene32768_logcount_human_relative.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair log-count and human-relative bbox features. "
            "Uses expanded bbox occupancy plus observed pelvis and candidate bbox only."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--num-scene-points", type=float, default=32768.0)
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


def parse_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vector3(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"Expected {name} to be a 3-vector, got {values!r}")
    return [float(values[0]), float(values[1]), float(values[2])]


def load_scene_graph_features(scene_graph_root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
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
                pelvis = vector3(sample.get("human_final_frame", {}).get("pelvis"), name="pelvis")
                pelvis_y = pelvis[1]
                for obj in sample.get("objects", []):
                    object_id = str(obj.get("object_id", ""))
                    if not sample_id or not object_id:
                        continue
                    bbox_min = vector3(obj.get("bbox_min"), name="bbox_min")
                    bbox_max = vector3(obj.get("bbox_max"), name="bbox_max")
                    bbox_low_y = min(bbox_min[1], bbox_max[1])
                    bbox_high_y = max(bbox_min[1], bbox_max[1])
                    bbox_center_y = 0.5 * (bbox_low_y + bbox_high_y)
                    features = {
                        "object_bbox_center_y_minus_pelvis_y": bbox_center_y - pelvis_y,
                    }
                    key = (split, sample_id, object_id)
                    previous = lookup.get(key)
                    if previous is not None:
                        max_abs_diff = max(abs(previous[name] - features[name]) for name in features)
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
            "Duplicate sample ids produced different human-relative features: "
            f"{duplicate_mismatches[:5]}"
        )
    return lookup


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id", "scene32768_expanded_bbox_1p1_fraction"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")

    lookup = load_scene_graph_features(args.scene_graph_root)
    feature_columns = [
        "scene32768_expanded_bbox_1p1_log_count",
        "object_bbox_center_y_minus_pelvis_y",
    ]

    output_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    for row in rows:
        output = dict(row)
        fraction = parse_float(row.get("scene32768_expanded_bbox_1p1_fraction"))
        output["scene32768_expanded_bbox_1p1_log_count"] = math.log1p(
            max(fraction, 0.0) * max(float(args.num_scene_points), 1.0)
        )
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        features = lookup.get(key)
        if features is None:
            missing_rows.append({"split": key[0], "sample_id": key[1], "object_id": key[2]})
            features = {"object_bbox_center_y_minus_pelvis_y": 0.0}
        output.update(features)
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in feature_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"features={','.join(feature_columns)}")
    print(f"missing_rows={len(missing_rows)}")


if __name__ == "__main__":
    main()
