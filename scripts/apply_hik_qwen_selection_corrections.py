#!/usr/bin/env python3
"""Apply validated Qwen correction rows over a complete base selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import (  # noqa: E402
    EVENT_TARGET_TYPE,
    iter_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graph-root", type=Path, required=True)
    parser.add_argument("--base-selection-root", type=Path, required=True)
    parser.add_argument("--corrections-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corrections = {}
    for path in args.corrections_root.glob("shard_*/*/selections.jsonl"):
        for row in iter_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in corrections and corrections[sample_id] != row:
                raise ValueError(f"Conflicting correction for {sample_id}")
            corrections[sample_id] = row
    if not corrections:
        raise ValueError(f"No corrections found under {args.corrections_root}")

    summary = {"corrected_count": len(corrections), "splits": {}}
    used = set()
    for split in ("train", "test"):
        graph_rows = list(
            iter_jsonl(args.scene_graph_root / split / "samples.jsonl")
        )
        graph_lookup = {str(row["sample_id"]): row for row in graph_rows}
        base_lookup = {
            str(row["sample_id"]): row
            for row in iter_jsonl(
                args.base_selection_root / split / "selections.jsonl"
            )
        }
        output_rows = []
        for graph in graph_rows:
            sample_id = str(graph["sample_id"])
            row = corrections.get(sample_id, base_lookup.get(sample_id))
            if row is None:
                raise ValueError(f"Missing selection for {sample_id}")
            expected_type = EVENT_TARGET_TYPE.get(str(graph["event_action"]))
            if expected_type is None:
                compatible = row["selected_object_id"] == "__none__"
            else:
                selected = next(
                    (
                        candidate
                        for candidate in graph["objects"]
                        if candidate["object_id"] == row["selected_object_id"]
                    ),
                    None,
                )
                compatible = (
                    selected is not None
                    and selected.get("label") == expected_type
                )
            if not compatible:
                raise ValueError(
                    f"Semantically incompatible selection for {sample_id}: "
                    f"{row['selected_object_id']}"
                )
            if sample_id in corrections:
                used.add(sample_id)
            output_rows.append(row)

        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selections.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for row in output_rows:
                output.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        summary["splits"][split] = {
            "selection_count": len(output_rows),
            "output_path": str(output_path.resolve()),
        }

    unused = sorted(set(corrections) - used)
    if unused:
        raise ValueError(f"Corrections not present in graph: {unused}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "correction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
