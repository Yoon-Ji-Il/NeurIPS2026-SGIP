#!/usr/bin/env python3
"""Merge deterministic HIK Qwen selection shards in scene-graph order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--shards-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "qwen3_14b_action_geometry_h25f50_shards",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "qwen3_14b_action_geometry_h25f50_v1",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def selected_splits(value: str) -> tuple[str, ...]:
    return ("train", "test") if value == "all" else (value,)


def main() -> None:
    args = parse_args()
    shard_dirs = sorted(path for path in args.shards_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories under {args.shards_root}")

    split_summaries = {}
    for split in selected_splits(args.split):
        lookup: dict[str, dict] = {}
        sources: dict[str, str] = {}
        for shard_dir in shard_dirs:
            selection_path = shard_dir / split / "selections.jsonl"
            if not selection_path.is_file():
                continue
            for row in iter_jsonl(selection_path):
                sample_id = str(row["sample_id"])
                if sample_id in lookup:
                    if row != lookup[sample_id]:
                        raise ValueError(
                            f"Conflicting duplicate {sample_id} in "
                            f"{sources[sample_id]} and {selection_path}"
                        )
                    continue
                lookup[sample_id] = row
                sources[sample_id] = str(selection_path)

        graph_rows = list(
            iter_jsonl(args.scene_graph_root / split / "samples.jsonl")
        )
        graph_ids = [str(row["sample_id"]) for row in graph_rows]
        graph_id_set = set(graph_ids)
        extra = sorted(set(lookup) - graph_id_set)
        if extra:
            raise ValueError(
                f"{split}: shard outputs contain {len(extra)} unknown sample IDs"
            )
        missing = [sample_id for sample_id in graph_ids if sample_id not in lookup]
        if missing and not args.allow_missing:
            raise ValueError(
                f"{split}: missing {len(missing)} selections; "
                f"first={missing[:5]}"
            )

        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selections.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for sample_id in graph_ids:
                if sample_id in lookup:
                    output.write(
                        json.dumps(
                            lookup[sample_id],
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        split_summaries[split] = {
            "scene_graph_count": len(graph_ids),
            "selection_count": len(graph_ids) - len(missing),
            "missing_count": len(missing),
            "missing_sample_ids": missing,
            "output_path": str(output_path.resolve()),
        }
        print(
            f"[{split}] selections={len(graph_ids) - len(missing)}/"
            f"{len(graph_ids)} missing={len(missing)}"
        )

    summary = {
        "scene_graph_root": str(args.scene_graph_root.resolve()),
        "shards_root": str(args.shards_root.resolve()),
        "shard_directories": [str(path.resolve()) for path in shard_dirs],
        "splits": split_summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
