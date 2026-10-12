#!/usr/bin/env python3
"""Export HIK student-selector top-1 objects in non-walking cache order."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import (  # noqa: E402
    HIKObjectCatalog,
    iter_jsonl,
    sample_object_surface_points,
    stable_seed,
)


DEFAULT_ALIGNMENT_ROOT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking/SLICES_888s"
)
DEFAULT_OUTPUT_NAME = (
    "final_object_student_hik_motion_c_kl_h25f50_nonwalking_"
    "seed7_points_1024.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/HIK/SAST/data"),
    )
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--student-summary-csv",
        type=Path,
        default=(
            REPO_ROOT
            / "runs_hik"
            / "hik_h25f50_nonwalking_student_selector_kl_seed7"
            / "sample_summary.csv"
        ),
    )
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ALIGNMENT_ROOT,
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--num-object-points", type=int, default=1024)
    parser.add_argument("--limit-per-split", type=int, default=-1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                str(key).lstrip("\ufeff").strip(): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(handle)
        ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.num_object_points <= 0:
        raise ValueError("--num-object-points must be positive")
    if not args.output_name.endswith(".pth"):
        raise ValueError("--output-name must end with .pth")

    student_rows = read_csv(args.student_summary_csv)
    student_lookup = {
        (row.get("split", ""), row.get("sample_id", "")): row
        for row in student_rows
    }
    if len(student_lookup) != len(student_rows):
        raise ValueError(f"Duplicate student rows in {args.student_summary_csv}")

    catalog = HIKObjectCatalog(args.raw_root)
    summaries: dict[str, Any] = {}
    for split in ("train", "test"):
        manifest_path = args.alignment_root / split / "hik_h25f50_cache_manifest.json"
        alignment = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.limit_per_split >= 0:
            alignment = alignment[: args.limit_per_split]
        if any(str(row.get("event_action", "")) == "walking" for row in alignment):
            raise ValueError(f"Walking row found in {manifest_path}")
        graph_lookup = {
            str(graph["sample_id"]): graph
            for graph in iter_jsonl(args.scene_graph_root / split / "samples.jsonl")
        }
        tensor = torch.zeros(
            (len(alignment), args.num_object_points, 3),
            dtype=torch.float32,
        )
        mask = torch.zeros((len(alignment),), dtype=torch.bool)
        export_manifest = []
        missing = []
        for index, alignment_row in enumerate(alignment):
            sample_id = str(alignment_row["sample_id"])
            graph = graph_lookup.get(sample_id)
            selection = student_lookup.get((split, sample_id))
            entry = {
                "index": index,
                "split": split,
                "sample_id": sample_id,
                "event_action": alignment_row["event_action"],
                "kitchen": alignment_row["kitchen"],
                "future_anchor": alignment_row["future_anchor"],
                "stored_point_count": args.num_object_points,
            }
            if graph is None or selection is None:
                entry["status"] = (
                    "missing_scene_graph"
                    if graph is None
                    else "missing_student_selection"
                )
                missing.append(entry)
                export_manifest.append(entry)
                continue
            selected_id = str(selection.get("selected_object_id", ""))
            entry.update(
                {
                    "selected_object_id": selected_id,
                    "selected_object_name": selection.get(
                        "selected_object_name",
                        "",
                    ),
                    "student_confidence": selection.get(
                        "student_confidence",
                        "",
                    ),
                    "teacher_selected_object_id": selection.get(
                        "teacher_selected_object_id",
                        "",
                    ),
                    "student_matches_teacher": selection.get(
                        "is_teacher_match",
                        "",
                    ),
                }
            )
            candidate_ids = {
                str(candidate.get("object_id", ""))
                for candidate in graph.get("objects", [])
            }
            if selected_id not in candidate_ids:
                entry["status"] = "student_object_absent_from_candidate_graph"
                missing.append(entry)
                export_manifest.append(entry)
                continue
            scene_object = catalog.get(str(graph["kitchen"]), selected_id)
            if scene_object is None:
                entry["status"] = "student_object_missing_from_hik_catalog"
                missing.append(entry)
                export_manifest.append(entry)
                continue
            try:
                points = sample_object_surface_points(
                    scene_object,
                    int(graph["last_observed_frame"]),
                    graph["scene_origin_model_absolute"],
                    args.num_object_points,
                    stable_seed(sample_id, selected_id, args.num_object_points),
                )
                tensor[index] = torch.from_numpy(points)
                mask[index] = True
                entry.update(
                    {
                        "object_type": scene_object.label,
                        "object_shape": scene_object.shape,
                        "object_geometry_path": str(scene_object.geometry_path),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                entry["status"] = (
                    f"object_points_build_failed:{type(exc).__name__}:{exc}"
                )
                missing.append(entry)
            export_manifest.append(entry)

        split_dir = args.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / args.output_name
        stem = args.output_name[:-4]
        mask_path = split_dir / f"{stem}_mask.pth"
        manifest_output = split_dir / f"{stem}_manifest.json"
        missing_output = split_dir / f"{stem}_missing.json"
        meta_output = split_dir / f"{stem}_meta.json"
        torch.save(tensor, output_path)
        torch.save(mask, mask_path)
        write_json(manifest_output, export_manifest)
        write_json(missing_output, missing)
        meta = {
            "split": split,
            "walking_excluded": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "valid_count": int(mask.sum().item()),
            "missing_count": int((~mask).sum().item()),
            "num_object_points": args.num_object_points,
            "output_path": str(output_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "student_summary_csv": str(args.student_summary_csv.resolve()),
            "alignment_manifest": str(manifest_path.resolve()),
            "coordinate_frame": "target_root_centered_model_y_up",
            "sampling_rule": (
                "deterministic uniform surface sampling of the student top-1 "
                "HIK object at the last observed frame"
            ),
        }
        write_json(meta_output, meta)
        summaries[split] = meta
        print(
            f"[{split}] wrote {output_path} shape={tuple(tensor.shape)} "
            f"valid={int(mask.sum())}"
        )
        if missing:
            raise RuntimeError(
                f"{split}: failed to export {len(missing)} student objects; "
                f"see {missing_output}"
            )

    write_json(args.output_root / "hik_student_object_points_summary.json", summaries)


if __name__ == "__main__":
    main()
