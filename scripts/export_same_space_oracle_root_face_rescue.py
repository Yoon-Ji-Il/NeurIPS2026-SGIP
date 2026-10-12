from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_SUMMARY = (
    REPO_ROOT / "object_selector" / "same_space_runs" / "baseline_teacher_no_want_v1" / "sample_summary.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "object_selector" / "same_space_runs" / "oracle_root_face_rescue_v1"
DEFAULT_TEACHER_CANDIDATE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "teacher_candidate_scores.csv"
)

FACE_HIGHER_BETTER = (
    "face_normal_alignment",
    "face_normal_horizontal_alignment",
)
FACE_LOWER_BETTER = ("face_normal_line_distance_xz",)
ROOT_HIGHER_BETTER = (
    "root_alignment",
    "root_horizontal_alignment",
    "root_torso_alignment",
    "root_torso_horizontal_alignment",
)
ROOT_LOWER_BETTER = ("root_line_distance_xz",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock correct same-space baseline selections, and for wrong samples apply "
            "an oracle rescue only when GT beats the current selection on root/face metrics."
        )
    )
    parser.add_argument("--baseline-summary-csv", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rescue-target",
        choices=("gt", "teacher"),
        default="gt",
        help="Rescue wrong baseline selections toward GT objects or toward the original teacher top-1 objects.",
    )
    parser.add_argument(
        "--teacher-candidate-csv",
        type=Path,
        default=DEFAULT_TEACHER_CANDIDATE_CSV,
        help="Original teacher candidate score csv used when --rescue-target=teacher.",
    )
    parser.add_argument(
        "--teacher-score-field",
        type=str,
        default="fusion_total",
        help="Teacher score column whose argmax defines teacher top-1 when --rescue-target=teacher.",
    )
    parser.add_argument(
        "--output-csv-name",
        type=str,
        default="sample_summary_oracle_root_face_rescue.csv",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{str(key).strip(): str(value).strip() for key, value in row.items() if key is not None} for row in csv.DictReader(file)]


def load_scene_metrics(scene_graph_root: Path) -> dict[str, dict[str, dict[str, object]]]:
    scene_by_sample: dict[str, dict[str, dict[str, object]]] = {}
    for split in ("train", "test"):
        path = scene_graph_root / split / "samples.jsonl"
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                scene_by_sample[payload["sample_id"]] = {
                    str(obj["object_id"]): obj for obj in payload.get("objects", [])
                }
    return scene_by_sample


def load_teacher_top1(teacher_candidate_csv: Path, score_field: str) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with teacher_candidate_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                grouped[sample_id].append({str(key).strip(): str(value).strip() for key, value in row.items() if key is not None})

    teacher_top1: dict[str, dict[str, str]] = {}
    for sample_id, rows in grouped.items():
        best = max(rows, key=lambda row: float(row.get(score_field, "0") or 0.0))
        teacher_top1[sample_id] = {
            "object_id": best.get("object_id", ""),
            "object_name": best.get("name", ""),
        }
    return teacher_top1


def descending_rank(objects: list[dict[str, object]], key: str) -> dict[str, int]:
    ordered = sorted(
        objects,
        key=lambda obj: float((obj.get("metrics") or {}).get(key, -1.0e9)),
        reverse=True,
    )
    return {str(obj["object_id"]): index + 1 for index, obj in enumerate(ordered)}


def ascending_rank(objects: list[dict[str, object]], key: str) -> dict[str, int]:
    ordered = sorted(
        objects,
        key=lambda obj: float((obj.get("metrics") or {}).get(key, 1.0e9)),
    )
    return {str(obj["object_id"]): index + 1 for index, obj in enumerate(ordered)}


def oracle_combo_scores(
    objects_by_id: dict[str, dict[str, object]],
) -> tuple[dict[str, float], dict[str, float]]:
    objects = list(objects_by_id.values())
    rank_tables: dict[str, dict[str, int]] = {}
    for key in (*FACE_HIGHER_BETTER, *ROOT_HIGHER_BETTER):
        rank_tables[key] = descending_rank(objects, key)
    for key in (*FACE_LOWER_BETTER, *ROOT_LOWER_BETTER):
        rank_tables[key] = ascending_rank(objects, key)

    face_scores: dict[str, float] = {}
    root_scores: dict[str, float] = {}
    for object_id in objects_by_id:
        face_scores[object_id] = sum(rank_tables[key][object_id] for key in (*FACE_HIGHER_BETTER, *FACE_LOWER_BETTER)) / len(
            FACE_HIGHER_BETTER + FACE_LOWER_BETTER
        )
        root_scores[object_id] = sum(rank_tables[key][object_id] for key in (*ROOT_HIGHER_BETTER, *ROOT_LOWER_BETTER)) / len(
            ROOT_HIGHER_BETTER + ROOT_LOWER_BETTER
        )
    return face_scores, root_scores


def export_oracle_rescue(
    *,
    baseline_rows: list[dict[str, str]],
    scenes_by_sample: dict[str, dict[str, dict[str, object]]],
    rescue_target: str,
    teacher_top1_by_sample: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "policy": f"{rescue_target}_root_face_rescue_same_space_v1",
        "splits": {},
    }

    for split in ("train", "test"):
        split_rows = [row for row in baseline_rows if row.get("split") == split]
        baseline_target_match = 0
        rescued_target_match = 0
        baseline_gt_exact = 0
        rescued_gt_exact = 0
        changed_count = 0
        rescue_count = 0
        branch_counts = {
            "base_correct_locked": 0,
            "root_combo": 0,
            "face_combo": 0,
            "no_rescue": 0,
        }

        for row in split_rows:
            sample_id = row["sample_id"]
            selected_object_id = row["selected_object_id"]
            gt_object_id = row["gt_object_id"]
            selected_object_name = row["selected_object_name"]
            gt_object_name = row["gt_object_name"]
            if rescue_target == "gt":
                target_object_id = gt_object_id
                target_object_name = gt_object_name
            elif rescue_target == "teacher":
                if teacher_top1_by_sample is None:
                    raise RuntimeError("teacher_top1_by_sample is required for rescue_target=teacher")
                teacher_target = teacher_top1_by_sample[sample_id]
                target_object_id = teacher_target["object_id"]
                target_object_name = teacher_target["object_name"]
            else:
                raise ValueError(f"Unsupported rescue_target: {rescue_target}")

            is_target_match = selected_object_id == target_object_id
            baseline_target_match += int(is_target_match)
            baseline_gt_exact += int(selected_object_id == gt_object_id)

            final_object_id = selected_object_id
            final_object_name = selected_object_name
            rescue_applied = 0
            oracle_branch = "base_correct_locked" if is_target_match else "no_rescue"
            root_combo_target_beats_selected = 0
            face_combo_target_beats_selected = 0

            if not is_target_match:
                objects_by_id = scenes_by_sample[sample_id]
                face_scores, root_scores = oracle_combo_scores(objects_by_id)
                root_combo_target_beats_selected = int(root_scores[target_object_id] < root_scores[selected_object_id])
                face_combo_target_beats_selected = int(face_scores[target_object_id] < face_scores[selected_object_id])
                if root_combo_target_beats_selected or face_combo_target_beats_selected:
                    final_object_id = target_object_id
                    final_object_name = target_object_name
                    rescue_applied = 1
                    rescue_count += 1
                    changed_count += 1
                    oracle_branch = "root_combo" if root_combo_target_beats_selected else "face_combo"

            rescued_target_match += int(final_object_id == target_object_id)
            rescued_gt_exact += int(final_object_id == gt_object_id)
            branch_counts[oracle_branch] += 1

            output_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "selected_object_id": final_object_id,
                    "selected_object_name": final_object_name,
                    "gt_object_id": gt_object_id,
                    "gt_object_name": gt_object_name,
                    "target_object_id": target_object_id,
                    "target_object_name": target_object_name,
                    "baseline_selected_object_id": selected_object_id,
                    "baseline_selected_object_name": selected_object_name,
                    "baseline_is_target_match": int(is_target_match),
                    "baseline_is_gt_exact": int(selected_object_id == gt_object_id),
                    "oracle_rescue_applied": rescue_applied,
                    "oracle_branch": oracle_branch,
                    "root_combo_target_beats_selected": root_combo_target_beats_selected,
                    "face_combo_target_beats_selected": face_combo_target_beats_selected,
                    "is_target_match": int(final_object_id == target_object_id),
                    "is_gt_exact": int(final_object_id == gt_object_id),
                }
            )

        summary["splits"][split] = {
            "total": len(split_rows),
            "baseline_target_match_count": baseline_target_match,
            "target_match_count": rescued_target_match,
            "baseline_gt_exact_count": baseline_gt_exact,
            "gt_exact_count": rescued_gt_exact,
            "changed_count": changed_count,
            "target_gain": rescued_target_match - baseline_target_match,
            "exact_gain": rescued_gt_exact - baseline_gt_exact,
            "oracle_rescue_applied_count": rescue_count,
            "branch_counts": branch_counts,
        }

    return output_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    baseline_rows = load_csv_rows(args.baseline_summary_csv)
    scenes_by_sample = load_scene_metrics(args.scene_graph_root)
    teacher_top1_by_sample = (
        load_teacher_top1(args.teacher_candidate_csv, args.teacher_score_field)
        if args.rescue_target == "teacher"
        else None
    )
    output_rows, summary = export_oracle_rescue(
        baseline_rows=baseline_rows,
        scenes_by_sample=scenes_by_sample,
        rescue_target=args.rescue_target,
        teacher_top1_by_sample=teacher_top1_by_sample,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / args.output_csv_name
    output_json = output_dir / "summary.json"

    summary["source_summary_csv"] = str(args.baseline_summary_csv)
    summary["output_csv"] = str(output_csv)
    summary["rescue_target"] = args.rescue_target
    if args.rescue_target == "teacher":
        summary["teacher_candidate_csv"] = str(args.teacher_candidate_csv)
        summary["teacher_score_field"] = args.teacher_score_field

    write_csv(output_csv, output_rows)
    write_json(output_json, summary)

    print(output_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
