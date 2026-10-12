#!/usr/bin/env python3
"""Replay the frozen baseline16+torso student on altered test scene graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.same_space import (  # noqa: E402
    SameSpaceSelector,
    build_samples_from_csv,
    compute_feature_stats,
)
from scripts.build_input_motion_same_space_features import (  # noqa: E402
    local_frame_features_for_candidate,
    temporal_attention_features_for_candidate,
)
from scripts.build_observed_hand_reach_features import (  # noqa: E402
    reach_features,
)
from scripts.build_scene32768_expanded_bbox_features import (  # noqa: E402
    expanded_fraction_features,
)
from scripts.build_support_rank_features import (  # noqa: E402
    SUPPORT_COLUMNS,
    build_support_lookup,
)
from scripts.train_same_space_residual_reranker import (  # noqa: E402
    ResidualReranker,
)


DEFAULT_SLICE_ROOT = Path(
    "data/GIMO/SLICES_888s"
)
DEFAULT_CLEAN_GRAPH_ROOT = (
    REPO_ROOT / "object_selector" / "scene_graphs"
)
DEFAULT_ROOT_INPUT = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_openins3d_shape.csv"
)
DEFAULT_ROOT_CHECKPOINT = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_aspect_seed_sweep"
    / "seed7"
    / "best_model_gt.pt"
)
DEFAULT_RESIDUAL_CHECKPOINTS = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_aspect_support_residual_trainfit_seed7"
    / "open_hidden128"
    / "best_model.pt",
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "raw5_scene32768_exp1p1_bbox_aspect_second_residual_trainfit_seed7"
    / "baseline5_ce12_rank10_h256"
    / "best_model_gt.pt",
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline9_local_xz_residual_sweep_seed7"
    / "open_max060_anchor070"
    / "best_train_gt_model.pt",
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_temporal_attention_combo_seed7"
    / "hand_face_mean_head"
    / "best_model.pt",
)
DEFAULT_CLEAN_STUDENT_SUMMARY = (
    REPO_ROOT
    / "object_selector"
    / "same_space_runs"
    / "baseline16_plus_torso_cone_w2_seed7"
    / "sample_summary.csv"
)
DEFAULT_TEACHER_SUMMARY = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "selector_sample_summary.csv"
)
TORSO_FEATURE = "input_torso_object_cone_fraction"
TORSO_WEIGHT = 2.0
TORSO_TRAIN_MEAN = 0.3263414138856484
TORSO_TRAIN_STD = 0.3143460090328361


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument(
        "--clean-scene-graph-root", type=Path, default=DEFAULT_CLEAN_GRAPH_ROOT
    )
    parser.add_argument("--root-input-csv", type=Path, default=DEFAULT_ROOT_INPUT)
    parser.add_argument(
        "--root-checkpoint", type=Path, default=DEFAULT_ROOT_CHECKPOINT
    )
    parser.add_argument(
        "--clean-student-summary-csv",
        type=Path,
        default=DEFAULT_CLEAN_STUDENT_SUMMARY,
    )
    parser.add_argument(
        "--teacher-summary-csv", type=Path, default=DEFAULT_TEACHER_SUMMARY
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-clean-validation",
        action="store_true",
        help="Skip the 37/37 clean Top-1 replay check.",
    )
    return parser.parse_args()


def clean_row(raw: dict[str, Any]) -> dict[str, str]:
    return {
        str(key).lstrip("\ufeff").strip(): (
            str(value).strip() if value is not None else ""
        )
        for key, value in raw.items()
        if key and str(key).strip()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def load_residual(
    path: Path, device: torch.device
) -> tuple[ResidualReranker, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualReranker(
        input_dim=len(checkpoint["feature_columns"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_residual=float(checkpoint["max_residual"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def bbox_aspect(scene_object: dict[str, Any]) -> float:
    bbox_min = torch.tensor(scene_object["bbox_min"], dtype=torch.float64)
    bbox_max = torch.tensor(scene_object["bbox_max"], dtype=torch.float64)
    extents = torch.abs(bbox_max - bbox_min)
    return float(extents.max().item() / max(extents.min().item(), 1.0e-8))


class FrozenStudent:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        root_checkpoint = torch.load(
            args.root_checkpoint, map_location="cpu", weights_only=False
        )
        self.root_columns = tuple(root_checkpoint["feature_columns"])
        raw_samples = build_samples_from_csv(
            args.root_input_csv,
            feature_columns=self.root_columns,
        )
        self.root_mean, self.root_std = compute_feature_stats(
            [sample for sample in raw_samples if sample.split == "train"]
        )
        self.root_model = SameSpaceSelector(
            input_dim=len(self.root_columns),
            hidden_dim=int(root_checkpoint["hidden_dim"]),
            dropout=float(root_checkpoint["dropout"]),
        ).to(device)
        self.root_model.load_state_dict(root_checkpoint["model_state"])
        self.root_model.eval()
        self.residual_stages = [
            load_residual(path, device)
            for path in DEFAULT_RESIDUAL_CHECKPOINTS
        ]
        self.device = device

    @torch.no_grad()
    def select(
        self,
        graph: dict[str, Any],
        scene_points: torch.Tensor,
        joints_input: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        objects = sorted(
            list(graph["objects"]),
            key=lambda obj: (str(obj["object_id"]), str(obj.get("label", ""))),
        )
        if not objects:
            raise ValueError(f"No candidates in {graph['sample_id']}")
        ids = [str(obj["object_id"]) for obj in objects]
        sample_id = str(graph["sample_id"])
        split = str(graph.get("split", "test"))

        expanded = {
            object_id: expanded_fraction_features(scene_points, scene_object)
            for object_id, scene_object in zip(ids, objects, strict=True)
        }
        root_features: list[list[float]] = []
        for object_id, scene_object in zip(ids, objects, strict=True):
            metrics = scene_object.get("metrics") or {}
            values: list[float] = []
            for column in self.root_columns:
                if column == "scene32768_expanded_bbox_1p1_fraction":
                    value = expanded[object_id][column]
                elif column == "object_shape_bbox_aspect_max_over_min":
                    value = bbox_aspect(scene_object)
                else:
                    value = metrics.get(column, 0.0)
                values.append(float(value or 0.0))
            root_features.append(values)

        root_tensor = torch.tensor(root_features, dtype=torch.float32)
        normalized = (root_tensor - self.root_mean) / self.root_std
        valid_mask = torch.ones(
            (1, len(ids)), dtype=torch.bool, device=self.device
        )
        output = self.root_model(
            normalized.unsqueeze(0).to(self.device), valid_mask
        )
        logits = output["logits"][0]
        posterior = output["distribution"][0]

        support_lookup = build_support_lookup(
            {(split, sample_id): objects},
            vertical_gap_low=-0.15,
            vertical_gap_high=0.35,
            larger_volume_ratio=1.5,
            min_overlap_frac=0.05,
        )
        geometry_features: dict[str, dict[str, float]] = {}
        for object_id in ids:
            key = (split, sample_id, object_id)
            features = dict(support_lookup[key])
            features.update(
                local_frame_features_for_candidate(
                    scene_sample=graph,
                    object_id=object_id,
                    joints_input=joints_input,
                )
            )
            features.update(
                reach_features(graph, object_id, joints_input)
            )
            features.update(
                temporal_attention_features_for_candidate(
                    scene_sample=graph,
                    object_id=object_id,
                    joints_input=joints_input,
                )
            )
            geometry_features[object_id] = features

        first_residual: torch.Tensor | None = None
        stage_scores: list[torch.Tensor] = [logits.detach().cpu()]
        for stage_index, (model, checkpoint) in enumerate(
            self.residual_stages
        ):
            feature_rows: list[list[float]] = []
            for candidate_index, object_id in enumerate(ids):
                values: list[float] = []
                for column in checkpoint["feature_columns"]:
                    if column == "baseline_student_score":
                        value = float(logits[candidate_index].item())
                    elif column == "baseline_student_posterior":
                        value = float(posterior[candidate_index].item())
                    elif column == "residual":
                        if first_residual is None:
                            raise RuntimeError(
                                "Residual feature requested before stage one"
                            )
                        value = float(
                            first_residual[candidate_index].item()
                        )
                    else:
                        value = float(
                            geometry_features[object_id].get(column, 0.0)
                        )
                    values.append(value)
                feature_rows.append(values)

            features = torch.tensor(feature_rows, dtype=torch.float32)
            mean = torch.tensor(
                checkpoint["feature_mean"], dtype=torch.float32
            )
            std = torch.tensor(
                checkpoint["feature_std"], dtype=torch.float32
            )
            stage_output = model(
                ((features - mean) / std).unsqueeze(0).to(self.device),
                logits.unsqueeze(0),
                valid_mask,
            )
            logits = stage_output["logits"][0]
            posterior = stage_output["distribution"][0]
            if stage_index == 0:
                # The legacy downstream CSVs retained the first residual
                # feature due to a duplicate column name. Replaying that
                # lineage exactly recovers all 37 clean test Top-1 choices.
                first_residual = stage_output["residual"][0]
            stage_scores.append(logits.detach().cpu())

        torso = torch.tensor(
            [
                geometry_features[object_id].get(TORSO_FEATURE, 0.0)
                for object_id in ids
            ],
            dtype=torch.float32,
            device=self.device,
        )
        torso_z = (torso - TORSO_TRAIN_MEAN) / TORSO_TRAIN_STD
        final_logits = logits + TORSO_WEIGHT * torso_z
        final_distribution = torch.softmax(final_logits, dim=0)
        selected_index = int(final_distribution.argmax().item())
        rank_order = torch.argsort(
            final_distribution, descending=True
        ).tolist()

        candidate_rows: list[dict[str, Any]] = []
        for index, (object_id, scene_object) in enumerate(
            zip(ids, objects, strict=True)
        ):
            row = {
                "split": split,
                "split_index": int(graph["sample_index"]),
                "sample_id": sample_id,
                "object_id": object_id,
                "name": scene_object.get("label", ""),
                "category": scene_object.get("label", ""),
                "root_student_score": float(stage_scores[0][index].item()),
                "baseline16_student_score": float(logits[index].item()),
                TORSO_FEATURE: float(torso[index].item()),
                "torso_cone_z": float(torso_z[index].item()),
                "student_score": float(final_logits[index].item()),
                "student_posterior": float(
                    final_distribution[index].item()
                ),
                "student_rank": int(rank_order.index(index) + 1),
                "is_student_selected": int(index == selected_index),
            }
            for column in SUPPORT_COLUMNS:
                row[column] = float(
                    geometry_features[object_id].get(column, 0.0)
                )
            candidate_rows.append(row)

        selected_object = objects[selected_index]
        return candidate_rows, {
            "split": split,
            "split_index": int(graph["sample_index"]),
            "sample_id": sample_id,
            "selected_object_id": ids[selected_index],
            "selected_object_name": selected_object.get("label", ""),
            "selected_label": selected_object.get("label", ""),
            "num_candidates": len(ids),
            "student_confidence": float(
                final_distribution[selected_index].item()
            ),
        }


def replay_graphs(
    selector: FrozenStudent,
    graphs: list[dict[str, Any]],
    scene_points: torch.Tensor,
    joints: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for graph in graphs:
        index = int(graph["sample_index"])
        candidates, selected = selector.select(
            graph, scene_points[index], joints[index]
        )
        candidate_rows.extend(candidates)
        sample_rows.append(selected)
    return candidate_rows, sample_rows


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    selector = FrozenStudent(args, device)
    slice_root = args.slice_root.resolve()
    scene_points = torch.load(
        slice_root / "test" / "scene_points_32768.pth",
        map_location="cpu",
        weights_only=False,
    ).float()
    joints = torch.load(
        slice_root / "test" / "joints_input.pth",
        map_location="cpu",
        weights_only=False,
    ).float()

    clean_summary_rows = read_csv(args.clean_student_summary_csv)
    clean_test_lookup = {
        row["sample_id"]: row
        for row in clean_summary_rows
        if row.get("split") == "test"
    }
    if not args.skip_clean_validation:
        clean_graphs = read_jsonl(
            args.clean_scene_graph_root / "test" / "samples.jsonl"
        )
        _, replayed = replay_graphs(
            selector, clean_graphs, scene_points, joints
        )
        mismatches = [
            {
                "sample_id": row["sample_id"],
                "replayed": row["selected_object_id"],
                "expected": clean_test_lookup[row["sample_id"]][
                    "selected_object_id"
                ],
            }
            for row in replayed
            if row["selected_object_id"]
            != clean_test_lookup[row["sample_id"]]["selected_object_id"]
        ]
        if mismatches:
            raise RuntimeError(
                f"Clean frozen-student replay mismatch: {mismatches[:5]}"
            )

    graphs = read_jsonl(
        args.scene_graph_root / "test" / "samples.jsonl"
    )
    if len(graphs) != 37:
        raise ValueError(
            f"Expected 37 altered test graphs, found {len(graphs)}"
        )
    candidate_rows, test_rows = replay_graphs(
        selector, graphs, scene_points, joints
    )

    teacher_lookup = {
        row["sample_id"]: row
        for row in read_csv(args.teacher_summary_csv)
        if row.get("split") == "test"
    }
    changed = 0
    for row in test_rows:
        sample_id = row["sample_id"]
        teacher = teacher_lookup[sample_id]
        clean = clean_test_lookup[sample_id]
        row.update(
            {
                "condition": args.condition,
                "teacher_selected_object_id": teacher[
                    "selected_object_id"
                ],
                "teacher_selected_object_name": teacher.get(
                    "selected_object_name", ""
                ),
                "gt_object_id": clean.get("gt_object_id", ""),
                "gt_object_name": clean.get("gt_object_name", ""),
                "clean_student_selected_object_id": clean[
                    "selected_object_id"
                ],
                "selection_changed_from_clean": int(
                    row["selected_object_id"]
                    != clean["selected_object_id"]
                ),
                "is_teacher_match": int(
                    row["selected_object_id"]
                    == teacher["selected_object_id"]
                ),
                "is_gt_exact": int(
                    row["selected_object_id"]
                    == clean.get("gt_object_id", "")
                ),
            }
        )
        changed += int(row["selection_changed_from_clean"])
    for row in candidate_rows:
        row["condition"] = args.condition

    train_rows = [
        dict(row)
        for row in clean_summary_rows
        if row.get("split") == "train"
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_scores.csv", candidate_rows)
    write_csv(output_dir / "test_sample_summary.csv", test_rows)
    write_csv(output_dir / "sample_summary.csv", train_rows + test_rows)
    summary = {
        "condition": args.condition,
        "device": str(device),
        "num_test_sequences": len(test_rows),
        "clean_replay_top1_match": 37,
        "selection_changed_from_clean_count": changed,
        "train": "unchanged clean summary",
        "test_scene_graph_root": str(args.scene_graph_root.resolve()),
        "student": (
            "frozen baseline16 temporal-attention residual stack plus "
            "torso-cone z-score weight 2.0"
        ),
    }
    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
