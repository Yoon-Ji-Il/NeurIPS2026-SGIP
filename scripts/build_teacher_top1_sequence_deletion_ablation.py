#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_support_rank_features import (  # noqa: E402
    SUPPORT_COLUMNS,
    build_support_lookup,
)
from scripts.train_same_space_residual_reranker import ResidualReranker  # noqa: E402


DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_TEACHER_SUMMARY = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "selector_sample_summary.csv"
)
DEFAULT_BASE_CANDIDATES = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "candidate_scores.csv"
)
DEFAULT_STUDENT_CHECKPOINT = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "best_model.pt"
)
DEFAULT_TORSO_FEATURES = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_input_motion_temporal_attention6.csv"
)
DEFAULT_CLEAN_STUDENT_SUMMARY = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_plus_torso_cone_w2_seed7"
    / "sample_summary.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "object_selector"
    / "teacher_top1_sequence_deletion_seed7"
)

SEVERITY_COUNTS = (
    ("25pct", 9),
    ("50pct", 17),
    ("75pct", 25),
    ("100pct", 37),
)
TORSO_FEATURE = "input_torso_object_cone_fraction"
TORSO_WEIGHT = 2.0
TORSO_TRAIN_MEAN = 0.3263414138856484
TORSO_TRAIN_STD = 0.3143460090328361


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete the clean Qwen teacher Top-1 object from nested subsets of test "
            "sequence scene graphs, then rerun the frozen student selector."
        )
    )
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--teacher-summary-csv", type=Path, default=DEFAULT_TEACHER_SUMMARY)
    parser.add_argument("--base-candidate-csv", type=Path, default=DEFAULT_BASE_CANDIDATES)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--torso-feature-csv", type=Path, default=DEFAULT_TORSO_FEATURES)
    parser.add_argument("--clean-student-summary-csv", type=Path, default=DEFAULT_CLEAN_STUDENT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if clean_key:
            output[clean_key] = str(value).strip() if value is not None else ""
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def group_candidates(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("split", ""), row.get("sample_id", ""))].append(row)
    return dict(grouped)


def build_lookup(
    rows: list[dict[str, str]],
    *,
    split: str | None = None,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        sample_id = row.get("sample_id", "")
        if not sample_id:
            continue
        if sample_id in output:
            raise ValueError(f"Duplicate sample_id in lookup: {sample_id}")
        output[sample_id] = row
    return output


def load_feature_lookup(path: Path, column: str) -> dict[tuple[str, str, str], float]:
    output: dict[tuple[str, str, str], float] = {}
    for row in read_csv(path):
        key = (row.get("split", ""), row.get("sample_id", ""), row.get("object_id", ""))
        if all(key):
            output[key] = float(row.get(column, "0") or 0.0)
    return output


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ResidualReranker, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_columns = list(checkpoint["feature_columns"])
    model = ResidualReranker(
        input_dim=len(feature_columns),
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_residual=float(checkpoint["max_residual"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def select_sample(
    *,
    rows: list[dict[str, str]],
    graph_objects: list[dict[str, Any]],
    model: ResidualReranker,
    checkpoint: dict[str, Any],
    torso_lookup: dict[tuple[str, str, str], float],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot select from an empty candidate set")
    rows = sorted(rows, key=lambda row: (row.get("object_id", ""), row.get("name", "")))
    split = rows[0]["split"]
    sample_id = rows[0]["sample_id"]
    graph_ids = {str(obj["object_id"]) for obj in graph_objects}
    row_ids = {row["object_id"] for row in rows}
    if row_ids != graph_ids:
        raise ValueError(
            f"Candidate/scene-graph object mismatch for {sample_id}: "
            f"candidate_only={sorted(row_ids - graph_ids)} graph_only={sorted(graph_ids - row_ids)}"
        )

    support_lookup = build_support_lookup(
        {(split, sample_id): graph_objects},
        vertical_gap_low=-0.15,
        vertical_gap_high=0.35,
        larger_volume_ratio=1.5,
        min_overlap_frac=0.05,
    )
    base_logits = torch.tensor(
        [float(row["baseline_score"]) for row in rows],
        dtype=torch.float32,
    )
    base_posterior = torch.softmax(base_logits, dim=0)
    feature_columns = list(checkpoint["feature_columns"])
    feature_values: list[list[float]] = []
    for index, row in enumerate(rows):
        key = (split, sample_id, row["object_id"])
        values: list[float] = []
        for column in feature_columns:
            if column in SUPPORT_COLUMNS:
                values.append(float(support_lookup[key][column]))
            elif column == "baseline_student_posterior":
                values.append(float(base_posterior[index].item()))
            else:
                values.append(float(row.get(column, "0") or 0.0))
        feature_values.append(values)

    features = torch.tensor(feature_values, dtype=torch.float32)
    mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32)
    normalized = ((features - mean) / std).unsqueeze(0).to(device)
    model_input_logits = base_logits.unsqueeze(0).to(device)
    valid_mask = torch.ones_like(model_input_logits, dtype=torch.bool)
    outputs = model(normalized, model_input_logits, valid_mask)
    mlp_logits = outputs["logits"][0].cpu()
    residual = outputs["residual"][0].cpu()

    torso = torch.tensor(
        [torso_lookup[(split, sample_id, row["object_id"])] for row in rows],
        dtype=torch.float32,
    )
    torso_z = (torso - TORSO_TRAIN_MEAN) / TORSO_TRAIN_STD
    final_logits = mlp_logits + TORSO_WEIGHT * torso_z
    final_distribution = torch.softmax(final_logits, dim=0)
    selected_index = int(final_distribution.argmax().item())

    teacher_raw = torch.tensor(
        [float(row.get("teacher_posterior", "0") or 0.0) for row in rows],
        dtype=torch.float32,
    ).clamp_min(0.0)
    teacher_distribution = teacher_raw / teacher_raw.sum().clamp_min(1.0e-8)
    student_order = torch.argsort(final_distribution, descending=True).tolist()
    teacher_order = torch.argsort(teacher_distribution, descending=True).tolist()
    candidate_output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        candidate_output.append(
            {
                "split": split,
                "sample_id": sample_id,
                "object_id": row["object_id"],
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "teacher_posterior_renormalized": float(teacher_distribution[index].item()),
                "base_score": float(base_logits[index].item()),
                "base_posterior_renormalized": float(base_posterior[index].item()),
                "mlp_residual": float(residual[index].item()),
                "mlp_score": float(mlp_logits[index].item()),
                TORSO_FEATURE: float(torso[index].item()),
                "torso_cone_z": float(torso_z[index].item()),
                "student_score": float(final_logits[index].item()),
                "student_posterior": float(final_distribution[index].item()),
                "teacher_rank_after_deletion": int(teacher_order.index(index) + 1),
                "student_rank": int(student_order.index(index) + 1),
                "is_student_selected": int(index == selected_index),
            }
        )

    selected_row = rows[selected_index]
    return candidate_output, {
        "split": split,
        "sample_id": sample_id,
        "selected_object_id": selected_row["object_id"],
        "selected_object_name": selected_row.get("name", ""),
        "num_candidates": len(rows),
        "student_confidence": float(final_distribution[selected_index].item()),
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model(args.student_checkpoint, device)

    train_graphs = read_jsonl(args.scene_graph_root / "train" / "samples.jsonl")
    test_graphs = read_jsonl(args.scene_graph_root / "test" / "samples.jsonl")
    if len(test_graphs) != 37:
        raise ValueError(f"Expected 37 test sequences, found {len(test_graphs)}")
    test_ids = [str(row["sample_id"]) for row in test_graphs]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Test scene graphs contain duplicate sample_id values")

    teacher_lookup = build_lookup(read_csv(args.teacher_summary_csv), split="test")
    clean_student_rows = read_csv(args.clean_student_summary_csv)
    clean_student_lookup = build_lookup(clean_student_rows, split="test")
    train_student_rows = [row for row in clean_student_rows if row.get("split") == "train"]
    candidate_groups = group_candidates(read_csv(args.base_candidate_csv))
    torso_lookup = load_feature_lookup(args.torso_feature_csv, TORSO_FEATURE)

    missing_teacher = sorted(set(test_ids) - set(teacher_lookup))
    missing_student = sorted(set(test_ids) - set(clean_student_lookup))
    missing_candidates = sorted(
        sample_id for sample_id in test_ids if ("test", sample_id) not in candidate_groups
    )
    if missing_teacher or missing_student or missing_candidates:
        raise ValueError(
            f"Missing test inputs: teacher={missing_teacher}, student={missing_student}, "
            f"candidates={missing_candidates}"
        )

    permutation = list(test_ids)
    random.Random(args.seed).shuffle(permutation)
    rank_by_id = {sample_id: rank for rank, sample_id in enumerate(permutation, start=1)}
    order_rows = []
    for graph in test_graphs:
        sample_id = str(graph["sample_id"])
        rank = rank_by_id[sample_id]
        order_rows.append(
            {
                "sample_index": int(graph["sample_index"]),
                "deletion_rank_seed7": rank,
                "sample_id": sample_id,
                "teacher_top1_object_id": teacher_lookup[sample_id]["selected_object_id"],
                "teacher_top1_object_name": teacher_lookup[sample_id].get("selected_object_name", ""),
                "in_25pct": int(rank <= 9),
                "in_50pct": int(rank <= 17),
                "in_75pct": int(rank <= 25),
                "in_100pct": 1,
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "nested_sequence_order_seed7.csv", order_rows)

    # A no-deletion replay must recover every clean student Top-1 before ablations are built.
    clean_mismatches: list[dict[str, str]] = []
    for graph in test_graphs:
        sample_id = str(graph["sample_id"])
        _, selected = select_sample(
            rows=candidate_groups[("test", sample_id)],
            graph_objects=list(graph["objects"]),
            model=model,
            checkpoint=checkpoint,
            torso_lookup=torso_lookup,
            device=device,
        )
        expected = clean_student_lookup[sample_id]["selected_object_id"]
        if selected["selected_object_id"] != expected:
            clean_mismatches.append(
                {
                    "sample_id": sample_id,
                    "replayed": selected["selected_object_id"],
                    "expected": expected,
                }
            )
    if clean_mismatches:
        raise RuntimeError(
            f"Frozen student clean replay mismatch for {len(clean_mismatches)} samples: "
            f"{clean_mismatches[:3]}"
        )

    args_summary: dict[str, Any] = {
        "seed": int(args.seed),
        "device": str(device),
        "num_test_sequences": len(test_ids),
        "nested_counts": {label: count for label, count in SEVERITY_COUNTS},
        "teacher_summary_csv": str(args.teacher_summary_csv.resolve()),
        "base_candidate_csv": str(args.base_candidate_csv.resolve()),
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "torso_feature_csv": str(args.torso_feature_csv.resolve()),
        "clean_student_summary_csv": str(args.clean_student_summary_csv.resolve()),
        "clean_replay_top1_match": len(test_ids),
        "conditions": {},
    }

    for label, count in SEVERITY_COUNTS:
        condition_dir = args.output_root / f"seqdrop_{label}"
        affected = set(permutation[:count])
        altered_graphs: list[dict[str, Any]] = []
        candidate_output: list[dict[str, Any]] = []
        test_summary: list[dict[str, Any]] = []
        deletion_rows: list[dict[str, Any]] = []
        changed_from_clean = 0

        for graph in test_graphs:
            sample_id = str(graph["sample_id"])
            teacher = teacher_lookup[sample_id]
            teacher_id = teacher["selected_object_id"]
            original_objects = list(graph["objects"])
            is_affected = sample_id in affected
            remaining_objects = (
                [obj for obj in original_objects if str(obj["object_id"]) != teacher_id]
                if is_affected
                else original_objects
            )
            removed_count = len(original_objects) - len(remaining_objects)
            if removed_count != int(is_affected):
                raise RuntimeError(
                    f"Expected to remove exactly {int(is_affected)} teacher Top-1 node(s) "
                    f"from {sample_id}, removed {removed_count}"
                )
            if not remaining_objects:
                raise RuntimeError(f"Teacher deletion left no candidates for {sample_id}")

            altered_graph = dict(graph)
            altered_graph["objects"] = remaining_objects
            altered_graph["teacher_top1_sequence_deletion"] = {
                "condition": f"seqdrop_{label}",
                "seed": int(args.seed),
                "deletion_rank": int(rank_by_id[sample_id]),
                "affected": bool(is_affected),
                "removed_object_id": teacher_id if is_affected else "",
                "removed_object_name": teacher.get("selected_object_name", "") if is_affected else "",
            }
            altered_graphs.append(altered_graph)

            rows = candidate_groups[("test", sample_id)]
            remaining_rows = (
                [row for row in rows if row["object_id"] != teacher_id]
                if is_affected
                else rows
            )
            candidate_rows, selected = select_sample(
                rows=remaining_rows,
                graph_objects=remaining_objects,
                model=model,
                checkpoint=checkpoint,
                torso_lookup=torso_lookup,
                device=device,
            )
            for row in candidate_rows:
                row["condition"] = f"seqdrop_{label}"
                row["sequence_affected"] = int(is_affected)
                row["deleted_teacher_top1_object_id"] = teacher_id if is_affected else ""
            candidate_output.extend(candidate_rows)

            clean_selected = clean_student_lookup[sample_id]["selected_object_id"]
            gt_object_id = clean_student_lookup[sample_id].get("gt_object_id", "")
            changed = selected["selected_object_id"] != clean_selected
            changed_from_clean += int(changed)
            selected.update(
                {
                    "teacher_selected_object_id": teacher_id,
                    "teacher_selected_object_name": teacher.get("selected_object_name", ""),
                    "gt_object_id": gt_object_id,
                    "gt_object_name": clean_student_lookup[sample_id].get("gt_object_name", ""),
                    "sequence_affected": int(is_affected),
                    "deletion_rank_seed7": int(rank_by_id[sample_id]),
                    "deleted_teacher_top1_object_id": teacher_id if is_affected else "",
                    "clean_student_selected_object_id": clean_selected,
                    "selection_changed_from_clean": int(changed),
                    "is_teacher_match": int(selected["selected_object_id"] == teacher_id),
                    "is_gt_exact": int(selected["selected_object_id"] == gt_object_id),
                }
            )
            test_summary.append(selected)
            deletion_rows.append(
                {
                    "sample_index": int(graph["sample_index"]),
                    "sample_id": sample_id,
                    "deletion_rank_seed7": int(rank_by_id[sample_id]),
                    "sequence_affected": int(is_affected),
                    "teacher_top1_object_id": teacher_id,
                    "teacher_top1_object_name": teacher.get("selected_object_name", ""),
                    "original_candidate_count": len(original_objects),
                    "remaining_candidate_count": len(remaining_objects),
                    "student_clean_object_id": clean_selected,
                    "student_ablation_object_id": selected["selected_object_id"],
                    "selection_changed_from_clean": int(changed),
                }
            )

        if sum(int(row["sequence_affected"]) for row in test_summary) != count:
            raise RuntimeError(f"Incorrect affected count for {label}")
        for row in test_summary:
            if int(row["sequence_affected"]) and row["selected_object_id"] == row["teacher_selected_object_id"]:
                raise RuntimeError(f"Deleted teacher target was selected for {row['sample_id']}")

        combined_summary = [dict(row) for row in train_student_rows] + test_summary
        write_jsonl(condition_dir / "scene_graphs" / "train" / "samples.jsonl", train_graphs)
        write_jsonl(condition_dir / "scene_graphs" / "test" / "samples.jsonl", altered_graphs)
        write_csv(condition_dir / "candidate_scores.csv", candidate_output)
        write_csv(condition_dir / "test_sample_summary.csv", test_summary)
        write_csv(condition_dir / "sample_summary.csv", combined_summary)
        write_csv(condition_dir / "deletion_manifest.csv", deletion_rows)
        condition_summary = {
            "condition": f"seqdrop_{label}",
            "seed": int(args.seed),
            "affected_sequence_count": count,
            "total_test_sequence_count": len(test_ids),
            "nominal_percentage": label,
            "actual_percentage": 100.0 * count / len(test_ids),
            "selection_changed_from_clean_count": changed_from_clean,
            "train_scene_graph": "unchanged clean copy",
            "test_scene_graph": "teacher Top-1 node removed only for affected sequences",
            "student_inference": (
                "frozen baseline16 temporal-attention residual MLP, followed by the "
                "original torso-cone z-score addition with weight 2.0"
            ),
        }
        with (condition_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(condition_summary, file, ensure_ascii=False, indent=2)
        args_summary["conditions"][label] = condition_summary
        print(
            f"[{label}] affected={count}/37 changed_student_top1={changed_from_clean}/37 "
            f"output={condition_dir}"
        )

    with (args.output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(args_summary, file, ensure_ascii=False, indent=2)
    print(f"Clean replay: {len(test_ids)}/{len(test_ids)} Top-1 selections matched")
    print(f"Wrote ablation artifacts to {args.output_root}")


if __name__ == "__main__":
    main()
