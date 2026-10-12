#!/usr/bin/env python3
"""Export Qwen-selected HIK objects as split-aligned 1024-point tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import (  # noqa: E402
    HIKObjectCatalog,
    NONE_OBJECT_ID,
    iter_jsonl,
    load_selection_lookup,
    sample_object_surface_points,
    stable_seed,
)


DEFAULT_OUTPUT_NAME = (
    "final_object_qwen3_14b_action_geometry_h25f50_"
    "t25_gt29_points_1024.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/HIK/SAST/data"
        ),
    )
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "qwen3_14b_action_geometry_h25f50_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/HIK/"
            "hik_preprocessed/H25F50/SLICES_888s"
        ),
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--num-object-points", type=int, default=1024)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return parser.parse_args()


def selected_splits(value: str) -> tuple[str, ...]:
    return ("train", "test") if value == "all" else (value,)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_split(
    *,
    split: str,
    args: argparse.Namespace,
    catalog: HIKObjectCatalog,
) -> dict:
    graphs = list(iter_jsonl(args.scene_graph_root / split / "samples.jsonl"))
    selections = load_selection_lookup(
        args.selection_root / split / "selections.jsonl"
    )
    tensor = torch.zeros(
        (len(graphs), int(args.num_object_points), 3),
        dtype=torch.float32,
    )
    mask = torch.zeros((len(graphs),), dtype=torch.bool)
    manifest = []
    missing = []

    for index, graph in enumerate(graphs):
        sample_id = str(graph["sample_id"])
        selection = selections.get(sample_id)
        entry = {
            "index": index,
            "split": split,
            "sample_id": sample_id,
            "kitchen": graph["kitchen"],
            "pid": graph["pid"],
            "event_action": graph["event_action"],
            "future_anchor": graph["future_anchor"],
            "last_observed_frame": graph["last_observed_frame"],
            "stored_point_count": int(args.num_object_points),
        }
        if selection is None:
            entry["status"] = "missing_qwen_selection"
            manifest.append(entry)
            missing.append(entry)
            continue

        selected_id = str(selection.get("selected_object_id", "")).strip()
        entry.update(
            {
                "selected_object_id": selected_id,
                "selected_object_name": selection.get(
                    "selected_object_name",
                    "",
                ),
                "selector_status": selection.get("status", ""),
                "selector_confidence": selection.get("confidence"),
                "selector_rationale_short": selection.get(
                    "rationale_short",
                    "",
                ),
            }
        )
        if selected_id == NONE_OBJECT_ID:
            entry["status"] = "no_target_object"
            manifest.append(entry)
            continue

        scene_object = catalog.get(graph["kitchen"], selected_id)
        if scene_object is None:
            entry["status"] = "selected_object_missing_from_hik_scene"
            manifest.append(entry)
            missing.append(entry)
            continue

        try:
            points = sample_object_surface_points(
                scene_object,
                int(graph["last_observed_frame"]),
                graph["scene_origin_model_absolute"],
                int(args.num_object_points),
                stable_seed(sample_id, selected_id, args.num_object_points),
            )
            tensor[index] = torch.from_numpy(points)
            mask[index] = True
            entry.update(
                {
                    "object_type": scene_object.label,
                    "object_shape": scene_object.shape,
                    "object_geometry_path": str(scene_object.geometry_path),
                    "object_frame_multiplier": scene_object.frame_multiplier,
                    "status": "ok",
                }
            )
        except Exception as exc:
            entry["status"] = (
                f"object_points_build_failed:{type(exc).__name__}:{exc}"
            )
            missing.append(entry)
        manifest.append(entry)

    split_dir = args.output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / args.output_name
    stem = args.output_name[:-4] if args.output_name.endswith(".pth") else args.output_name
    mask_path = split_dir / f"{stem}_mask.pth"
    manifest_path = split_dir / f"{stem}_manifest.json"
    missing_path = split_dir / f"{stem}_missing.json"
    meta_path = split_dir / f"{stem}_meta.json"
    torch.save(tensor, output_path)
    torch.save(mask, mask_path)
    write_json(manifest_path, manifest)
    write_json(missing_path, missing)
    meta = {
        "split": split,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "valid_count": int(mask.sum().item()),
        "missing_count": int((~mask).sum().item()),
        "valid_object_count": int(mask.sum().item()),
        "no_target_or_missing_count": int((~mask).sum().item()),
        "missing_error_count": len(missing),
        "num_object_points": int(args.num_object_points),
        "output_path": str(output_path),
        "mask_path": str(mask_path),
        "scene_graph_path": str(
            (args.scene_graph_root / split / "samples.jsonl").resolve()
        ),
        "selection_path": str(
            (args.selection_root / split / "selections.jsonl").resolve()
        ),
        "coordinate_frame": "target_root_centered_model_y_up",
        "sampling_rule": (
            "deterministic uniform surface sampling of the Qwen-selected HIK "
            "object at the last observed frame"
        ),
        "walking_policy": (
            "__none__ is stored as zero points with a false object mask"
        ),
    }
    write_json(meta_path, meta)
    print(
        f"[{split}] wrote {output_path} shape={tuple(tensor.shape)} "
        f"valid={int(mask.sum())}"
    )
    return meta


def main() -> None:
    args = parse_args()
    if args.num_object_points <= 0:
        raise ValueError("--num-object-points must be positive")
    if not args.output_name.endswith(".pth"):
        raise ValueError("--output-name must end with .pth")
    catalog = HIKObjectCatalog(args.raw_root)
    summaries = {
        split: build_split(split=split, args=args, catalog=catalog)
        for split in selected_splits(args.split)
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "hik_final_object_points_summary.json", summaries)


if __name__ == "__main__":
    main()
