#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "candidate_scores.csv"
)
DEFAULT_MORPHOLOGY_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_bbox_aspect_extras.csv"
)
DEFAULT_INPUT_MOTION_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_bbox_align_bodyface_proxy_16f.csv"
)
DEFAULT_TEMPORAL_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_temporal_attention6.csv"
)
DEFAULT_HAND_REACH_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_baseline10_observed_hand_reach.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_morphology_motion_interactions.csv"
)

INTERACTION_PAIRS = (
    ("motion_hand_bbox_proximity", "shape_inv_log_volume"),
    ("motion_hand_bbox_proximity", "object_shape_log_point_density"),
    ("motion_hand_bbox_proximity", "object_shape_bbox_aspect_max_over_min"),
    ("motion_hand_bbox_proximity", "object_shape_pca_linearity"),
    ("motion_hand_bbox_proximity", "object_shape_pca_scattering"),
    ("motion_hand_bbox_proximity", "object_shape_bottom_half_fraction"),
    ("motion_hand_bbox_proximity", "object_shape_vertical_lower_minus_upper_fraction"),
    ("motion_hand_approach_pos", "shape_inv_log_volume"),
    ("motion_hand_approach_pos", "object_shape_log_point_density"),
    ("motion_hand_approach_pos", "object_shape_bbox_aspect_max_over_min"),
    ("motion_hand_approach_pos", "object_shape_pca_linearity"),
    ("motion_hand_approach_pos", "object_shape_pca_scattering"),
    ("motion_hand_speed_toward_pos", "shape_inv_log_volume"),
    ("motion_hand_speed_toward_pos", "object_shape_log_point_density"),
    ("motion_hand_speed_toward_pos", "object_shape_bbox_aspect_max_over_min"),
    ("motion_obs_hand_last_proximity", "shape_inv_log_volume"),
    ("motion_obs_hand_approach_pos", "shape_inv_log_volume"),
    ("motion_root_bbox_proximity", "object_bbox_height"),
    ("motion_root_bbox_proximity", "object_bbox_log_xz_area"),
    ("motion_root_bbox_proximity", "object_shape_pca_planarity"),
    ("motion_root_bbox_proximity", "object_shape_bottom_half_fraction"),
    ("motion_root_bbox_proximity", "object_shape_vertical_lower_minus_upper_fraction"),
    ("motion_root_approach_pos", "object_bbox_height"),
    ("motion_root_approach_pos", "object_bbox_log_xz_area"),
    ("motion_root_approach_pos", "object_shape_pca_planarity"),
    ("motion_root_alignment", "object_bbox_height"),
    ("motion_root_alignment", "object_bbox_log_xz_area"),
    ("motion_root_alignment", "object_shape_bbox_aspect_max_over_min"),
    ("motion_root_alignment", "object_shape_pca_planarity"),
    ("motion_root_endpoint_proximity", "object_bbox_height"),
    ("motion_root_endpoint_proximity", "object_bbox_log_xz_area"),
    ("motion_root_endpoint_proximity", "object_shape_pca_planarity"),
    ("motion_face_ray_proximity", "object_shape_bbox_aspect_max_over_min"),
    ("motion_face_ray_proximity", "object_shape_pca_planarity"),
    ("motion_face_ray_proximity", "object_bbox_height"),
    ("motion_face_ray_proximity", "object_bbox_log_xz_area"),
    ("motion_head_cone_fraction", "object_shape_bbox_aspect_max_over_min"),
    ("motion_head_cone_fraction", "object_shape_pca_planarity"),
    ("motion_head_cone_fraction", "object_bbox_height"),
    ("motion_face_alignment", "object_shape_bbox_aspect_max_over_min"),
    ("motion_face_alignment", "object_shape_pca_planarity"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build label-free morphology x observed-motion interaction features. "
            "The output contains only sample/object ids plus interaction columns; it does "
            "not use want/action text, object names/categories, GT labels, teacher outputs, "
            "future frames, or final gaze."
        )
    )
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--morphology-csv", type=Path, default=DEFAULT_MORPHOLOGY_CSV)
    parser.add_argument("--input-motion-csv", type=Path, default=DEFAULT_INPUT_MOTION_CSV)
    parser.add_argument("--temporal-csv", type=Path, default=DEFAULT_TEMPORAL_CSV)
    parser.add_argument("--hand-reach-csv", type=Path, default=DEFAULT_HAND_REACH_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
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


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))


def load_lookup(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {key(row): row for row in read_csv(path)}


def exp_distance_score(value: float, scale: float = 1.0) -> float:
    return math.exp(-max(value, 0.0) / max(scale, 1.0e-8))


def positive(value: float) -> float:
    return max(value, 0.0)


def shape_values(row: dict[str, str]) -> dict[str, float]:
    values = {
        "object_bbox_height": parse_float(row.get("object_bbox_height")),
        "object_bbox_log_volume": parse_float(row.get("object_bbox_log_volume")),
        "object_bbox_log_xz_area": parse_float(row.get("object_bbox_log_xz_area")),
        "object_shape_pca_linearity": parse_float(row.get("object_shape_pca_linearity")),
        "object_shape_pca_planarity": parse_float(row.get("object_shape_pca_planarity")),
        "object_shape_pca_scattering": parse_float(row.get("object_shape_pca_scattering")),
        "object_shape_bbox_aspect_max_over_min": parse_float(row.get("object_shape_bbox_aspect_max_over_min")),
        "object_shape_bottom_half_fraction": parse_float(row.get("object_shape_bottom_half_fraction")),
        "object_shape_log_point_density": parse_float(row.get("object_shape_log_point_density")),
        "object_shape_vertical_lower_minus_upper_fraction": parse_float(
            row.get("object_shape_vertical_lower_minus_upper_fraction")
        ),
    }
    values["shape_inv_log_volume"] = 1.0 / (1.0 + max(values["object_bbox_log_volume"], 0.0))
    return values


def motion_values(
    input_row: dict[str, str],
    temporal_row: dict[str, str],
    hand_row: dict[str, str],
) -> dict[str, float]:
    return {
        "motion_root_bbox_proximity": exp_distance_score(parse_float(input_row.get("input_root_bbox_distance"))),
        "motion_hand_bbox_proximity": exp_distance_score(parse_float(input_row.get("input_hand_bbox_distance"))),
        "motion_face_bbox_proximity": exp_distance_score(parse_float(input_row.get("input_face_bbox_distance"))),
        "motion_root_approach_pos": positive(parse_float(input_row.get("input_root_approach_delta"))),
        "motion_hand_approach_pos": positive(parse_float(input_row.get("input_hand_approach_delta"))),
        "motion_face_approach_pos": positive(parse_float(input_row.get("input_face_approach_delta"))),
        "motion_root_alignment": parse_float(input_row.get("input_root_motion_alignment")),
        "motion_hand_alignment": parse_float(input_row.get("input_hand_motion_alignment")),
        "motion_face_alignment": parse_float(input_row.get("input_face_motion_alignment")),
        "motion_root_endpoint_proximity": exp_distance_score(
            parse_float(input_row.get("input_root_pred_endpoint_distance"))
        ),
        "motion_root_endpoint_alignment": parse_float(input_row.get("input_root_pred_endpoint_alignment")),
        "motion_hand_speed_toward_pos": positive(parse_float(input_row.get("input_hand_speed_toward_object"))),
        "motion_face_ray_proximity": exp_distance_score(parse_float(input_row.get("input_face_ray_distance_to_object"))),
        "motion_hand_heading_fraction": parse_float(temporal_row.get("input_hand_heading_object_fraction")),
        "motion_head_cone_fraction": parse_float(temporal_row.get("input_head_object_cone_fraction")),
        "motion_face_ray_perp_proximity": exp_distance_score(parse_float(temporal_row.get("input_face_ray_perp_mean"))),
        "motion_obs_hand_last_proximity": exp_distance_score(parse_float(hand_row.get("obs_hand_last_bbox_distance_3d"))),
        "motion_obs_hand_min_proximity": exp_distance_score(parse_float(hand_row.get("obs_hand_min_bbox_distance_3d"))),
        "motion_obs_hand_approach_pos": positive(parse_float(hand_row.get("obs_hand_approach_delta_bbox_3d"))),
    }


def interaction_column_name(motion_name: str, shape_name: str) -> str:
    motion_clean = motion_name.removeprefix("motion_")
    shape_clean = shape_name.removeprefix("object_").removeprefix("shape_")
    return f"mm_{motion_clean}_x_{shape_clean}"


def main() -> None:
    args = parse_args()
    baseline_rows = read_csv(args.baseline_csv)
    morphology_lookup = load_lookup(args.morphology_csv)
    input_lookup = load_lookup(args.input_motion_csv)
    temporal_lookup = load_lookup(args.temporal_csv)
    hand_lookup = load_lookup(args.hand_reach_csv)

    columns = [interaction_column_name(motion_name, shape_name) for motion_name, shape_name in INTERACTION_PAIRS]
    output_rows: list[dict[str, Any]] = []
    missing = 0
    for row in baseline_rows:
        row_key = key(row)
        morphology_row = morphology_lookup.get(row_key, {})
        input_row = input_lookup.get(row_key, {})
        temporal_row = temporal_lookup.get(row_key, {})
        hand_row = hand_lookup.get(row_key, {})
        if not morphology_row or not input_row:
            missing += 1
        shapes = shape_values(morphology_row)
        motions = motion_values(input_row, temporal_row, hand_row)
        output = {"split": row_key[0], "sample_id": row_key[1], "object_id": row_key[2]}
        for motion_name, shape_name in INTERACTION_PAIRS:
            output[interaction_column_name(motion_name, shape_name)] = motions[motion_name] * shapes[shape_name]
        output_rows.append(output)

    write_csv(args.output_csv, output_rows, ["split", "sample_id", "object_id", *columns])
    print(f"wrote {args.output_csv}")
    print(f"num_rows={len(output_rows)} missing_source_rows={missing}")
    print("interaction_columns=" + ",".join(columns))


if __name__ == "__main__":
    main()
