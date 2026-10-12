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
    / "teacher_candidate_scores_scene32768_local.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_scene32768_local_interactions.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"

INTERACTION_COLUMNS = (
    "scene32768_root_align_x_bbox_fraction",
    "scene32768_torso_align_x_bbox_fraction",
    "scene32768_root_torso_align_x_bbox_fraction",
    "scene32768_face_align_x_bbox_fraction",
    "scene32768_face_line_exp_x_bbox_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fair multiplicative interaction features built only from raw5 "
            "observed-motion features and static scene32768 bbox_fraction."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
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


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_scene_graph_metrics(scene_graph_root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    output: dict[tuple[str, str, str], dict[str, float]] = {}
    for split in ("train", "test"):
        path = scene_graph_root / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scene graph file: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("sample_id", ""))
                for obj in sample.get("objects", []):
                    object_id = str(obj.get("object_id", ""))
                    metrics = obj.get("metrics") or {}
                    output[(split, sample_id, object_id)] = {
                        "root_alignment": parse_float(metrics.get("root_alignment")),
                        "torso_alignment": parse_float(metrics.get("torso_alignment")),
                        "root_torso_alignment": parse_float(metrics.get("root_torso_alignment")),
                        "face_normal_alignment": parse_float(metrics.get("face_normal_alignment")),
                        "face_normal_line_distance_xz": parse_float(metrics.get("face_normal_line_distance_xz")),
                    }
    return output


def feature_value(row: dict[str, str], metrics: dict[str, float], column: str) -> float:
    if column in row and str(row[column]).strip():
        return parse_float(row.get(column))
    return parse_float(metrics.get(column))


def build_interactions(row: dict[str, str], metrics: dict[str, float]) -> dict[str, float]:
    bbox_fraction = parse_float(row.get("scene32768_bbox_fraction"))
    face_line = max(feature_value(row, metrics, "face_normal_line_distance_xz"), 0.0)
    return {
        "scene32768_root_align_x_bbox_fraction": feature_value(row, metrics, "root_alignment") * bbox_fraction,
        "scene32768_torso_align_x_bbox_fraction": feature_value(row, metrics, "torso_alignment") * bbox_fraction,
        "scene32768_root_torso_align_x_bbox_fraction": feature_value(row, metrics, "root_torso_alignment")
        * bbox_fraction,
        "scene32768_face_align_x_bbox_fraction": feature_value(row, metrics, "face_normal_alignment") * bbox_fraction,
        "scene32768_face_line_exp_x_bbox_fraction": math.exp(-face_line) * bbox_fraction,
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")
    required = {"split", "sample_id", "object_id", "scene32768_bbox_fraction"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_csv}: {missing}")
    scene_graph_metrics = load_scene_graph_metrics(args.scene_graph_root)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        metrics = scene_graph_metrics.get(key, {})
        output = dict(row)
        output.update(build_interactions(row, metrics))
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in INTERACTION_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"wrote {args.output_csv}")
    print(f"features={','.join(INTERACTION_COLUMNS)}")


if __name__ == "__main__":
    main()
