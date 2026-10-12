"""Build an aligned HIK H25F50 cache containing no walking sequences."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch


DEFAULT_SOURCE = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50"
)
DEFAULT_OUTPUT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking"
)
DEFAULT_OBJECT_NAME = (
    "final_object_qwen3_14b_action_geometry_h25f50_"
    "t25_gt29_points_1024.pth"
)
TENSOR_FILES = {
    "poses_input.pth": (25, 38),
    "poses_label.pth": (50, 38),
    "joints_input.pth": (25, 23, 3),
    "joints_label.pth": (50, 23, 3),
    "scene_points_1000.pth": (1000, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def filter_manifest(rows: list[dict], keep: list[int]) -> list[dict]:
    filtered = []
    for output_index, source_index in enumerate(keep):
        row = dict(rows[source_index])
        row["source_index"] = int(source_index)
        row["index"] = int(output_index)
        filtered.append(row)
    return filtered


def build_split(
    source_root: Path,
    output_root: Path,
    split: str,
    object_name: str,
) -> dict:
    source = source_root / "SLICES_888s" / split
    output = output_root / "SLICES_888s" / split
    object_stem = Path(object_name).stem
    model_manifest_path = source / "hik_h25f50_cache_manifest.json"
    object_manifest_path = source / f"{object_stem}_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    object_manifest = json.loads(
        object_manifest_path.read_text(encoding="utf-8")
    )
    if len(model_manifest) != len(object_manifest):
        raise ValueError(f"{split}: manifest lengths differ")
    model_ids = [row["sample_id"] for row in model_manifest]
    object_ids = [row["sample_id"] for row in object_manifest]
    if model_ids != object_ids:
        raise ValueError(f"{split}: motion and object sample order differs")

    keep = [
        index
        for index, row in enumerate(model_manifest)
        if row["event_action"] != "walking"
    ]
    excluded = len(model_manifest) - len(keep)
    index = torch.tensor(keep, dtype=torch.long)
    filtered_model_manifest = filter_manifest(model_manifest, keep)
    filtered_object_manifest = filter_manifest(object_manifest, keep)

    output.mkdir(parents=True, exist_ok=True)
    output_shapes = {}
    for name, trailing_shape in TENSOR_FILES.items():
        tensor = torch.load(source / name, map_location="cpu", weights_only=False)
        expected = (len(model_manifest), *trailing_shape)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{split}/{name}: {tuple(tensor.shape)} != {expected}"
            )
        filtered = tensor.index_select(0, index).contiguous()
        if filtered.dtype != torch.float32 or not torch.isfinite(filtered).all():
            raise ValueError(f"{split}/{name}: invalid dtype or non-finite values")
        atomic_torch_save(filtered, output / name)
        output_shapes[name] = list(filtered.shape)

    object_points = torch.load(
        source / object_name,
        map_location="cpu",
        weights_only=False,
    )
    object_mask = torch.load(
        source / f"{object_stem}_mask.pth",
        map_location="cpu",
        weights_only=False,
    ).bool()
    expected_object_shape = (len(model_manifest), 1024, 3)
    if tuple(object_points.shape) != expected_object_shape:
        raise ValueError(
            f"{split}: object shape {tuple(object_points.shape)} "
            f"!= {expected_object_shape}"
        )
    filtered_objects = object_points.index_select(0, index).contiguous().float()
    filtered_mask = object_mask.index_select(0, index).contiguous()
    if not filtered_mask.all():
        raise ValueError(f"{split}: a non-walking object mask is false")
    if not torch.isfinite(filtered_objects).all():
        raise ValueError(f"{split}: object points contain non-finite values")
    if not (filtered_objects.abs().sum(dim=(1, 2)) > 0).all():
        raise ValueError(f"{split}: a non-walking object point cloud is empty")

    atomic_torch_save(filtered_objects, output / object_name)
    atomic_torch_save(filtered_mask, output / f"{object_stem}_mask.pth")
    write_json(
        output / "hik_h25f50_cache_manifest.json",
        filtered_model_manifest,
    )
    write_json(
        output / f"{object_stem}_manifest.json",
        filtered_object_manifest,
    )
    write_json(output / f"{object_stem}_missing.json", [])

    people_counts = [
        int(row["other_people"]["num_people"])
        for row in filtered_model_manifest
    ]
    actions = Counter(
        row["event_action"] for row in filtered_model_manifest
    )
    summary = {
        "split": split,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "filter": "event_action != walking",
        "source_samples": len(model_manifest),
        "excluded_walking": excluded,
        "samples": len(keep),
        "action_counts": dict(sorted(actions.items())),
        "all_object_masks_true": bool(filtered_mask.all()),
        "object_shape": list(filtered_objects.shape),
        "tensor_shapes": output_shapes,
        "samples_with_people": sum(count > 0 for count in people_counts),
        "max_people": max(people_counts, default=0),
        "manifest_object_order_equal": [
            row["sample_id"] for row in filtered_model_manifest
        ]
        == [row["sample_id"] for row in filtered_object_manifest],
    }
    write_json(output / "hik_h25f50_cache_summary.json", summary)
    write_json(
        output / f"{object_stem}_meta.json",
        {
            "split": split,
            "shape": list(filtered_objects.shape),
            "dtype": str(filtered_objects.dtype),
            "valid_count": len(keep),
            "missing_count": 0,
            "valid_object_count": len(keep),
            "no_target_or_missing_count": 0,
            "missing_error_count": 0,
            "num_object_points": 1024,
            "source_object_path": str(source / object_name),
            "output_path": str(output / object_name),
            "mask_path": str(output / f"{object_stem}_mask.pth"),
            "coordinate_frame": "target_root_centered_model_y_up",
            "filter": "walking sequences removed; all masks are true",
            "embedding_policy": (
                "object embeddings are recomputed online by the model's "
                "object encoder during evaluation and fine-tuning"
            ),
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    summaries = {
        split: build_split(
            args.source_root.resolve(),
            args.output_root.resolve(),
            split,
            args.object_name,
        )
        for split in ("train", "test")
    }
    write_json(
        args.output_root.resolve() / "hik_h25f50_nonwalking_summary.json",
        summaries,
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
