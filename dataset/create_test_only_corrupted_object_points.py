#!/usr/bin/env python3
"""Build altered test object points while copying the clean student train cache."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "dataset") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "dataset"))

from create_final_object_points_from_selector import (  # noqa: E402
    apply_scene_normalization,
    clean_row,
    load_raw_points,
    load_transform_context,
    norm_value,
    sample_key,
    sample_object_points,
    stable_seed,
)


DEFAULT_GIMO_ROOT = Path("data/GIMO")
DEFAULT_SLICE_ROOT = DEFAULT_GIMO_ROOT / "SLICES_888s"
DEFAULT_CLEAN_OBJECT_NAME = (
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-summary-csv", type=Path, required=True)
    parser.add_argument("--scene-graph-root", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--gimo-root", type=Path, default=DEFAULT_GIMO_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=DEFAULT_GIMO_ROOT / "dataset.csv",
    )
    parser.add_argument(
        "--clean-train-object-name", default=DEFAULT_CLEAN_OBJECT_NAME
    )
    parser.add_argument("--num-object-points", type=int, default=1024)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def split_name(row: dict[str, str]) -> str:
    return "train" if norm_value(row["training"]) == "1" else "test"


def read_graphs(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            graph = json.loads(line)
            sample_id = str(graph["sample_id"])
            if sample_id in output:
                raise ValueError(f"Duplicate scene graph sample_id: {sample_id}")
            output[sample_id] = graph
    return output


def main() -> None:
    args = parse_args()
    if not args.output_name.endswith(".pth"):
        raise ValueError("--output-name must end in .pth")
    slice_root = args.slice_root.resolve()
    train_dir = slice_root / "train"
    test_dir = slice_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name.removesuffix(".pth")
    mask_name = f"{stem}_mask.pth"

    clean_train_object = train_dir / args.clean_train_object_name
    clean_train_mask = train_dir / args.clean_train_object_name.replace(
        ".pth", "_mask.pth"
    )
    if not clean_train_object.is_file() or not clean_train_mask.is_file():
        raise FileNotFoundError(
            f"Missing clean train cache: {clean_train_object}, "
            f"{clean_train_mask}"
        )
    shutil.copy2(clean_train_object, train_dir / args.output_name)
    shutil.copy2(clean_train_mask, train_dir / mask_name)

    selections = {
        row["sample_id"]: row
        for row in read_csv(args.selector_summary_csv)
        if row.get("split") == "test"
    }
    graphs = read_graphs(
        args.scene_graph_root / "test" / "samples.jsonl"
    )
    rows = [
        row
        for row in read_csv(args.dataset_csv)
        if split_name(row) == "test"
    ]
    if len(rows) != 37 or len(selections) != 37 or len(graphs) != 37:
        raise ValueError(
            "Expected 37 test rows/selections/graphs, got "
            f"{len(rows)}/{len(selections)}/{len(graphs)}"
        )

    tensor = torch.zeros(
        (len(rows), args.num_object_points, 3), dtype=torch.float32
    )
    mask = torch.zeros((len(rows),), dtype=torch.bool)
    manifest: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    points_cache: dict[str, Any] = {}

    for index, row in enumerate(rows):
        key = sample_key(row)
        sample_id = "|".join(key)
        selection = selections.get(sample_id)
        graph = graphs.get(sample_id)
        record: dict[str, Any] = {
            "index": index,
            "split": "test",
            "sample_id": sample_id,
            "sample_key": list(key),
        }
        if selection is None or graph is None:
            record["status"] = (
                "missing_selection"
                if selection is None
                else "missing_graph"
            )
            missing.append(record)
            manifest.append(record)
            continue

        selected_id = selection["selected_object_id"]
        object_lookup = {
            str(obj["object_id"]): obj for obj in graph["objects"]
        }
        selected = object_lookup.get(selected_id)
        record.update(
            {
                "selected_object_id": selected_id,
                "selected_object_name": selection.get(
                    "selected_object_name", ""
                ),
            }
        )
        if selected is None:
            record["status"] = "selected_object_absent_from_graph"
            missing.append(record)
            manifest.append(record)
            continue

        ply_path = Path(str(selected["source_ply_path"]))
        record["object_ply_path"] = str(ply_path)
        record["object_name_from_graph"] = selected.get("label", "")
        try:
            cache_key = str(ply_path.resolve())
            if cache_key not in points_cache:
                points_cache[cache_key] = load_raw_points(ply_path)
            scale, transform_norm = load_transform_context(
                args.gimo_root, row
            )
            aligned = apply_scene_normalization(
                points_cache[cache_key], scale, transform_norm
            )
            sampled, replacement = sample_object_points(
                aligned,
                args.num_object_points,
                stable_seed(key, selected_id),
            )
            tensor[index] = torch.from_numpy(sampled)
            mask[index] = True
            record.update(
                {
                    "raw_point_count": int(aligned.shape[0]),
                    "stored_point_count": args.num_object_points,
                    "used_replacement": bool(replacement),
                    "status": "ok",
                }
            )
        except Exception as error:  # noqa: BLE001
            record["status"] = f"object_points_build_failed: {error}"
            missing.append(record)
        manifest.append(record)

    torch.save(tensor, test_dir / args.output_name)
    torch.save(mask, test_dir / mask_name)
    write_json(test_dir / f"{stem}_manifest.json", manifest)
    write_json(test_dir / f"{stem}_missing.json", missing)
    write_json(
        test_dir / f"{stem}_meta.json",
        {
            "split": "test",
            "shape": list(tensor.shape),
            "valid_count": int(mask.sum().item()),
            "missing_count": len(missing),
            "output_path": str(test_dir / args.output_name),
            "mask_path": str(test_dir / mask_name),
            "selector_summary_csv": str(
                args.selector_summary_csv.resolve()
            ),
            "scene_graph_root": str(args.scene_graph_root.resolve()),
            "train_cache": (
                "byte-for-byte copy of the clean student train cache"
            ),
        },
    )
    write_json(
        train_dir / f"{stem}_meta.json",
        {
            "split": "train",
            "source_object_path": str(clean_train_object),
            "source_mask_path": str(clean_train_mask),
            "output_path": str(train_dir / args.output_name),
            "mask_path": str(train_dir / mask_name),
            "train_cache": (
                "byte-for-byte copy of the clean student train cache"
            ),
        },
    )
    print(
        json.dumps(
            {
                "output_name": args.output_name,
                "test_shape": list(tensor.shape),
                "test_valid_count": int(mask.sum().item()),
                "test_missing_count": len(missing),
                "train_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
