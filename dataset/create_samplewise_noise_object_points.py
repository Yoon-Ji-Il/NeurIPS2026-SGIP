#!/usr/bin/env python3
"""Build object-point tensors from sample-wise noisy scene graphs and Top-1 CSV."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_DATASET_CSV = DEFAULT_GIMO_ROOT / "dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-summary-csv", type=Path, required=True)
    parser.add_argument("--scene-graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gimo-dataset-csv", type=Path, default=DEFAULT_DATASET_CSV
    )
    parser.add_argument("--gimo-root", type=Path, default=DEFAULT_GIMO_ROOT)
    parser.add_argument("--num-object-points", type=int, default=1024)
    parser.add_argument(
        "--output-name",
        default="final_object_qwen3_14b_segmentation_noise_points_1024.pth",
    )
    parser.add_argument("--train-reference-manifest", type=Path, required=True)
    parser.add_argument("--test-reference-manifest", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def load_graphs(root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = root / split / "samples.jsonl"
    lookup: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            graph = json.loads(line)
            identifier = str(
                graph.get("sample_uid") or graph["sample_id"]
            )
            if identifier in lookup:
                raise ValueError(f"Duplicate graph sample_id: {identifier}")
            lookup[identifier] = graph
    return lookup


def selector_lookup(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        split = str(row.get("split", "")).strip()
        split_index_text = str(row.get("split_index", "")).strip()
        if not split or not split_index_text:
            continue
        key = (split, int(split_index_text))
        if key in lookup:
            raise ValueError(f"Duplicate selector split/index: {key}")
        lookup[key] = row
    return lookup


def reference_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "|".join(str(value) for value in row["sample_key"]): row
        for row in rows
    }


def split_name(row: dict[str, str]) -> str:
    return "train" if norm_value(row["training"]) == "1" else "test"


def sample_id_from_key(key: tuple[str, str, str, str]) -> str:
    return "|".join(key)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_split(
    *,
    split: str,
    rows: list[dict[str, str]],
    selections: dict[tuple[str, int], dict[str, str]],
    graphs: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_rows = [row for row in rows if split_name(row) == split]
    tensor = torch.zeros(
        (len(split_rows), args.num_object_points, 3), dtype=torch.float32
    )
    mask = torch.zeros((len(split_rows),), dtype=torch.bool)
    manifest: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    points_cache: dict[str, Any] = {}

    for index, row in enumerate(split_rows):
        key = sample_key(row)
        identifier = sample_id_from_key(key)
        sample_uid = f"{split}:{index:04d}:{identifier}"
        selection = selections.get((split, index))
        graph = graphs.get(sample_uid)
        reference = references.get(identifier, {})
        record: dict[str, Any] = {
            "index": index,
            "split": split,
            "sample_id": identifier,
            "sample_uid": sample_uid,
            "sample_key": list(key),
            "scene_name": key[0],
            "sequence_name": key[1],
            "start_frame": key[2],
            "end_frame": key[3],
            "gt_object_id": reference.get("gt_object_id", ""),
        }
        if selection is None or graph is None:
            record["status"] = (
                "missing_selection" if selection is None else "missing_graph"
            )
            missing.append(record)
            manifest.append(record)
            continue

        selected_id = str(selection.get("selected_object_id", "")).strip()
        object_lookup = {
            str(value.get("object_id", "")): value
            for value in graph.get("objects", [])
        }
        selected = object_lookup.get(selected_id)
        record.update(
            {
                "selected_object_id": selected_id,
                "selected_object_name": selection.get("selected_label", ""),
                "selected_matches_gt": (
                    selected_id == str(reference.get("gt_object_id", ""))
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
        record["object_num_points_from_graph"] = selected.get(
            "raw_num_points", 0
        )
        try:
            path_key = str(ply_path.resolve())
            if path_key not in points_cache:
                points_cache[path_key] = load_raw_points(ply_path)
            scale, transform_norm = load_transform_context(args.gimo_root, row)
            aligned = apply_scene_normalization(
                points_cache[path_key], scale, transform_norm
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

    output_dir = args.output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    stem = args.output_name.removesuffix(".pth")
    mask_path = output_dir / f"{stem}_mask.pth"
    torch.save(tensor, output_path)
    torch.save(mask, mask_path)
    write_json(output_dir / f"{stem}_manifest.json", manifest)
    write_json(output_dir / f"{stem}_missing.json", missing)
    meta = {
        "split": split,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "valid_count": int(mask.sum().item()),
        "missing_count": len(missing),
        "output_path": str(output_path),
        "mask_path": str(mask_path),
        "selector_summary_csv": str(args.selector_summary_csv),
        "scene_graph_root": str(args.scene_graph_root),
    }
    write_json(output_dir / f"{stem}_meta.json", meta)
    return meta


def main() -> None:
    args = parse_args()
    if not args.output_name.endswith(".pth"):
        raise ValueError("--output-name must end with .pth")
    rows = load_csv(args.gimo_dataset_csv)
    selections = selector_lookup(load_csv(args.selector_summary_csv))
    references = {
        "train": reference_lookup(args.train_reference_manifest),
        "test": reference_lookup(args.test_reference_manifest),
    }
    summaries = []
    for split in ("train", "test"):
        summaries.append(
            build_split(
                split=split,
                rows=rows,
                selections=selections,
                graphs=load_graphs(args.scene_graph_root, split),
                references=references[split],
                args=args,
            )
        )
    write_json(
        args.output_root / "summary.json",
        {
            "scene_graph_root": str(args.scene_graph_root),
            "selector_summary_csv": str(args.selector_summary_csv),
            "splits": summaries,
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
