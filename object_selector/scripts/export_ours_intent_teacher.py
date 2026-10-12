#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.common.want_taxonomy import normalize_object_name  # noqa: E402


DEFAULT_GT_EVAL_CSV = (
    Path(".")
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "merged_fusion_sem_face_body_line"
    / "gt_eval_all.csv"
)
DEFAULT_CANDIDATE_CSV = (
    Path(".")
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "merged_fusion_sem_face_body_line"
    / "teacher_candidate_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "ours_intent_teacher"
)
DEFAULT_POINTS_BUILDER = (
    Path(".")
    / "dataset"
    / "create_final_object_points_from_selector.py"
)
DEFAULT_GIMO_DATASET_CSV = Path("data/GIMO/dataset.csv")
DEFAULT_GIMO_ROOT = Path("data/GIMO")
DEFAULT_OPENINS3D_ROOT = Path("data/OpenIns3D/output/results_demo")
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")

SEMANTIC_GAP = 0.0
POINTS_OUTPUT_NAME = "final_object_ours_intent_teacher_points_1024.pth"
NUM_OBJECT_POINTS = 1024

GROUPS = {
    "G1_stable_lowline": ["sit", "sleep", "clean", "throw", "cook", "write", "fix"],
    "G2_active_face": ["drink", "use", "open", "unknown"],
    "G3_bodyline": ["eat", "read", "light"],
    "G4_take_dress": ["take", "dress"],
}
WANT_TO_GROUP = {
    want: group_name
    for group_name, wants in GROUPS.items()
    for want in wants
}
GROUP_WEIGHTS = {
    "G1_stable_lowline": {
        "face_alignment_norm": 0.558962,
        "body_alignment_norm": 2.068599,
        "face_line_score": 0.535333,
    },
    "G2_active_face": {
        "face_alignment_norm": 0.547361,
        "body_alignment_norm": 2.031033,
        "face_line_score": 0.543432,
    },
    "G3_bodyline": {
        "face_alignment_norm": 0.520479,
        "body_alignment_norm": 2.027746,
        "face_line_score": 0.545710,
    },
    "G4_take_dress": {
        "face_alignment_norm": 0.518157,
        "body_alignment_norm": 2.027086,
        "face_line_score": 0.533567,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the ours_intent_teacher selector outputs and optionally "
            "build the matching object-point .pth tensors."
        )
    )
    parser.add_argument("--gt-eval-csv", default=str(DEFAULT_GT_EVAL_CSV))
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--build-object-points",
        action="store_true",
        help="If set, also build final_object_ours_intent_teacher_points_1024.pth.",
    )
    parser.add_argument("--points-builder-script", default=str(DEFAULT_POINTS_BUILDER))
    parser.add_argument("--gimo-dataset-csv", default=str(DEFAULT_GIMO_DATASET_CSV))
    parser.add_argument("--gimo-root", default=str(DEFAULT_GIMO_ROOT))
    parser.add_argument("--openins3d-root", default=str(DEFAULT_OPENINS3D_ROOT))
    parser.add_argument("--slice-root", default=str(DEFAULT_SLICE_ROOT))
    parser.add_argument("--points-output-name", default=POINTS_OUTPUT_NAME)
    parser.add_argument("--num-object-points", type=int, default=NUM_OBJECT_POINTS)
    return parser.parse_args()


def clean_row(raw_row: dict[str, object]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = str(key).lstrip("\ufeff").strip()
        if not clean_key:
            continue
        row[clean_key] = str(value).strip() if value is not None else ""
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def normalize_name(name: str) -> str:
    normalized = normalize_object_name(name)
    if normalized:
        return normalized
    return str(name or "").strip().lower().replace("_", " ")


def class_matches(pred_name: str, gt_name: str) -> bool:
    pred = normalize_name(pred_name)
    gt = normalize_name(gt_name)
    return bool(gt) and pred == gt


def sample_parts(sample_id: str) -> tuple[str, str, str, str]:
    parts = str(sample_id).split("|")
    if len(parts) != 4:
        return "", "", "", ""
    return parts[0], parts[1], parts[2], parts[3]


def candidate_pool(rows: list[dict[str, str]]) -> set[str]:
    max_semantic = max(parse_float(row.get("semantic_total"), 0.0) for row in rows)
    return {
        row.get("object_id", "")
        for row in rows
        if max_semantic - parse_float(row.get("semantic_total"), 0.0) <= SEMANTIC_GAP + 1.0e-8
    }


def fusion_score(row: dict[str, str], weights: dict[str, float]) -> float:
    return (
        parse_float(row.get("semantic_total"), 0.0)
        + weights["face_alignment_norm"] * parse_float(row.get("face_alignment_norm"), 0.0)
        + weights["body_alignment_norm"] * parse_float(row.get("body_alignment_norm"), 0.0)
        + weights["face_line_score"] * parse_float(row.get("face_line_score"), 0.0)
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    gt_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, object]:
    candidates_by_sample: dict[str, list[dict[str, str]]] = {}
    for row in candidate_rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            continue
        candidates_by_sample.setdefault(sample_id, []).append(row)

    candidate_output_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    gt_eval_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    split_exact = Counter()
    split_class = Counter()
    split_total = Counter()
    split_in_pool = Counter()

    for split_index, gt_row in enumerate(gt_rows):
        sample_id = gt_row.get("sample_id", "")
        want = gt_row.get("want", "unknown")
        group_name = WANT_TO_GROUP[want]
        weights = GROUP_WEIGHTS[group_name]
        rows = candidates_by_sample.get(sample_id, [])
        if not rows:
            raise RuntimeError(f"Missing candidate rows for sample_id={sample_id}")

        scene_name = gt_row.get("scene_name", "") or sample_parts(sample_id)[0]
        sequence_name = gt_row.get("sequence_name", "") or sample_parts(sample_id)[1]
        start_frame = gt_row.get("start_frame", "") or sample_parts(sample_id)[2]
        end_frame = gt_row.get("end_frame", "") or sample_parts(sample_id)[3]
        split = gt_row.get("split", "")
        gt_object_id = gt_row.get("gt_object_id", "")
        gt_object_name = gt_row.get("gt_object_name", "")

        pool = candidate_pool(rows)
        split_total[split] += 1
        if gt_object_id in pool:
            split_in_pool[split] += 1

        best_row = max(
            rows,
            key=lambda row: (
                row.get("object_id", "") in pool,
                fusion_score(row, weights),
                row.get("object_id", ""),
            ),
        )
        selected_object_id = best_row.get("object_id", "")
        selected_object_name = best_row.get("name", "")
        exact = selected_object_id == gt_object_id
        cls = exact or class_matches(selected_object_name, gt_object_name)
        split_exact[split] += int(exact)
        split_class[split] += int(cls)

        selector_rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "scene_name": scene_name,
                "sequence_name": sequence_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "want": want,
                "selected_object_id": selected_object_id,
                "selected_object_name": selected_object_name,
                "selected_matches_gt": str(exact),
                "selected_class_matches_gt": str(cls),
                "gt_object_id": gt_object_id,
                "gt_object": gt_object_name,
                "gt_action": want,
            }
        )
        gt_eval_rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "scene_name": scene_name,
                "sequence_name": sequence_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "want": want,
                "final_object": selected_object_id,
                "final_object_name": selected_object_name,
                "gt_object_id": gt_object_id,
                "gt_object_name": gt_object_name,
                "winner_stage": "semantic_top_shared_intent_geom",
                "tie_set": "semantic_gap_0",
                "exact_instance_match": int(exact),
                "class_name_match": int(cls),
            }
        )
        selection_rows.append(
            {
                "status": "ok",
                "split": split,
                "sample_id": sample_id,
                "scene_name": scene_name,
                "sequence_name": sequence_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "want": want,
                "group_name": group_name,
                "num_candidates": len(rows),
                "final_object": selected_object_id,
                "final_object_name": selected_object_name,
                "winner_stage": "semantic_top_shared_intent_geom",
                "tie_set": "semantic_gap_0",
                "error": "",
            }
        )

        for row in rows:
            object_id = row.get("object_id", "")
            candidate_output_rows.append(
                {
                    "split": split,
                    "split_index": split_index,
                    "sample_id": sample_id,
                    "scene_name": scene_name,
                    "sequence_name": sequence_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "want": want,
                    "group_name": group_name,
                    "object_id": object_id,
                    "name": row.get("name", ""),
                    "category": row.get("category", row.get("name", "")),
                    "semantic_total": row.get("semantic_total", ""),
                    "face_alignment_norm": row.get("face_alignment_norm", ""),
                    "body_alignment_norm": row.get("body_alignment_norm", ""),
                    "face_line_score": row.get("face_line_score", ""),
                    "w_sem": 1.0,
                    "w_face": weights["face_alignment_norm"],
                    "w_body": weights["body_alignment_norm"],
                    "w_line": weights["face_line_score"],
                    "fusion_total": fusion_score(row, weights),
                    "in_candidate_pool": int(object_id in pool),
                    "is_final": int(object_id == selected_object_id),
                    "is_gt": int(object_id == gt_object_id),
                }
            )

    summary = {
        "method": "ours_intent_teacher",
        "formula": "semantic_total + w_face*face_alignment_norm + w_body*body_alignment_norm + w_line*face_line_score",
        "semantic_gap": SEMANTIC_GAP,
        "future_features_used": False,
        "w_sem_fixed": 1.0,
        "group_mapping": GROUPS,
        "group_weights": GROUP_WEIGHTS,
        "num_samples": len(gt_rows),
        "metrics": {},
    }
    for split in ("train", "test"):
        total = split_total[split]
        exact = split_exact[split]
        cls = split_class[split]
        in_pool = split_in_pool[split]
        summary["metrics"][split] = {
            "total": total,
            "exact_instance_match": exact,
            "exact_instance_accuracy": exact / total if total else 0.0,
            "class_name_match": cls,
            "class_name_accuracy": cls / total if total else 0.0,
            "target_in_pool": in_pool,
            "target_in_pool_accuracy": in_pool / total if total else 0.0,
        }

    total_all = sum(split_total.values())
    exact_all = sum(split_exact.values())
    class_all = sum(split_class.values())
    in_pool_all = sum(split_in_pool.values())
    summary["metrics"]["all"] = {
        "total": total_all,
        "exact_instance_match": exact_all,
        "exact_instance_accuracy": exact_all / total_all if total_all else 0.0,
        "class_name_match": class_all,
        "class_name_accuracy": class_all / total_all if total_all else 0.0,
        "target_in_pool": in_pool_all,
        "target_in_pool_accuracy": in_pool_all / total_all if total_all else 0.0,
    }

    write_csv(
        output_dir / "selector_sample_summary.csv",
        selector_rows,
        [
            "split",
            "sample_id",
            "scene_name",
            "sequence_name",
            "start_frame",
            "end_frame",
            "want",
            "selected_object_id",
            "selected_object_name",
            "selected_matches_gt",
            "selected_class_matches_gt",
            "gt_object_id",
            "gt_object",
            "gt_action",
        ],
    )
    write_csv(
        output_dir / "gt_eval_all.csv",
        gt_eval_rows,
        [
            "split",
            "sample_id",
            "scene_name",
            "sequence_name",
            "start_frame",
            "end_frame",
            "want",
            "final_object",
            "final_object_name",
            "gt_object_id",
            "gt_object_name",
            "winner_stage",
            "tie_set",
            "exact_instance_match",
            "class_name_match",
        ],
    )
    write_csv(
        output_dir / "teacher_selection_summary.csv",
        selection_rows,
        [
            "status",
            "split",
            "sample_id",
            "scene_name",
            "sequence_name",
            "start_frame",
            "end_frame",
            "want",
            "group_name",
            "num_candidates",
            "final_object",
            "final_object_name",
            "winner_stage",
            "tie_set",
            "error",
        ],
    )
    write_csv(
        output_dir / "teacher_candidate_scores.csv",
        candidate_output_rows,
        [
            "split",
            "split_index",
            "sample_id",
            "scene_name",
            "sequence_name",
            "start_frame",
            "end_frame",
            "want",
            "group_name",
            "object_id",
            "name",
            "category",
            "semantic_total",
            "face_alignment_norm",
            "body_alignment_norm",
            "face_line_score",
            "w_sem",
            "w_face",
            "w_body",
            "w_line",
            "fusion_total",
            "in_candidate_pool",
            "is_final",
            "is_gt",
        ],
    )
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary


def build_object_points(args: argparse.Namespace, selector_summary_csv: Path) -> None:
    command = [
        sys.executable,
        str(Path(args.points_builder_script)),
        "--selector-summary-csv",
        str(selector_summary_csv),
        "--gimo-dataset-csv",
        str(Path(args.gimo_dataset_csv)),
        "--gimo-root",
        str(Path(args.gimo_root)),
        "--openins3d-root",
        str(Path(args.openins3d_root)),
        "--output-root",
        str(Path(args.slice_root)),
        "--num-object-points",
        str(args.num_object_points),
        "--output-name",
        args.points_output_name,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    gt_eval_csv = Path(args.gt_eval_csv)
    candidate_csv = Path(args.candidate_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = read_csv(gt_eval_csv)
    candidate_rows = read_csv(candidate_csv)
    summary = build_outputs(gt_rows, candidate_rows, output_dir)
    print(f"wrote {output_dir / 'selector_sample_summary.csv'}")
    print(f"wrote {output_dir / 'teacher_candidate_scores.csv'}")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))

    if args.build_object_points:
        build_object_points(args, output_dir / "selector_sample_summary.csv")
        print(
            "built object points:",
            Path(args.slice_root) / "train" / args.points_output_name,
            Path(args.slice_root) / "test" / args.points_output_name,
        )


if __name__ == "__main__":
    main()
