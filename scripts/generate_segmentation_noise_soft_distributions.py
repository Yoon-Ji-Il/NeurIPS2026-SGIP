#!/usr/bin/env python3
"""Generate Qwen soft candidate distributions for every segmentation-noise condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gaze_qwen_object_selector import (  # noqa: E402
    LocalQwenSemanticClient,
    run_selector,
)


CONDITIONS = (
    "random_instance_deletion_10pct",
    "random_instance_deletion_20pct",
    "target_neighbor_merge",
    "target_two_way_split",
    "partial_noisy_target_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    parser.add_argument("--softmax-temperature", type=float, default=4.0)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "object_selector"
            / "teacher_targets"
            / "segmentation_noise_soft"
        ),
    )
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "object_selector" / "scene_graphs" / "segmentation_noise",
    )
    parser.add_argument(
        "--gimo-root", type=Path, default=Path("data/GIMO")
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=REPO_ROOT / "dataset" / "combined_dataset.csv",
    )
    return parser.parse_args()


def condition_complete(path: Path) -> bool:
    summary_path = path / "run_summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_rows = summary.get("splits", [])
    return bool(split_rows) and all(
        int(row.get("processed", 0)) == int(row.get("completed", -1))
        and int(row.get("failed", -1)) == 0
        for row in split_rows
    )


def selector_args(args: argparse.Namespace, condition: str) -> SimpleNamespace:
    slice_root = args.gimo_root / "SLICES_888s"
    return SimpleNamespace(
        gimo_root=args.gimo_root,
        gimo_dataset_csv=args.gimo_root / "dataset.csv",
        metadata_csv=args.metadata_csv,
        slice_root=slice_root,
        scene_graph_root=args.scene_graph_root / condition,
        output_dir=args.output_root / condition,
        openins3d_root=Path("data/OpenIns3D/output/results_demo"),
        output_root=slice_root,
        split=args.split,
        limit=args.limit,
        sample_id=args.sample_id,
        backend="qwen",
        selection_mode="soft_logits",
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
        max_new_tokens=1,
        softmax_temperature=args.softmax_temperature,
        last_k_frames=3,
        temporal_decay=0.7,
        bbox_margin=0.08,
        point_distance_scale=0.20,
        centroid_distance_scale=0.75,
        bbox_hit_weight=1.5,
        point_score_weight=1.0,
        centroid_score_weight=0.35,
        save_prompts=args.save_prompts,
        build_points=False,
        num_object_points=1024,
        object_output_name="",
    )


def main() -> None:
    args = parse_args()
    pending = [
        condition
        for condition in args.conditions
        if not (
            args.skip_complete
            and condition_complete(args.output_root / condition)
        )
    ]
    if not pending:
        print("All requested condition distributions are already complete.")
        return

    client = LocalQwenSemanticClient(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
        max_new_tokens=1,
    )
    summaries = []
    for condition in pending:
        print(f"=== soft distribution: {condition} ===", flush=True)
        summary = run_selector(
            selector_args(args, condition),
            llm_client=client,
        )
        summaries.append({"condition": condition, **summary})

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_name = (
        "all_conditions_summary.json"
        if tuple(args.conditions) == CONDITIONS
        else f"conditions_{'__'.join(args.conditions)}_summary.json"
    )
    target = args.output_root / summary_name
    target.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
