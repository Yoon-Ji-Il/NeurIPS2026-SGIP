#!/usr/bin/env python3
"""Merge sharded HIK candidate semantic scores in non-walking manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import iter_jsonl  # noqa: E402


DEFAULT_ALIGNMENT_ROOT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking/SLICES_888s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--limit-per-split", type=int, default=-1)
    return parser.parse_args()


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "walking_excluded": True,
        "alignment_root": str(args.alignment_root.resolve()),
        "shard_roots": [str(path.resolve()) for path in args.shard_roots],
        "output_root": str(args.output_root.resolve()),
        "splits": {},
    }
    for split in ("train", "test"):
        lookup: dict[str, dict[str, Any]] = {}
        duplicate_count = 0
        for shard_root in args.shard_roots:
            path = shard_root / split / "semantic_scores.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            for row in iter_jsonl(path):
                sample_id = str(row["sample_id"])
                if sample_id in lookup:
                    duplicate_count += 1
                    if canonical(lookup[sample_id]) != canonical(row):
                        raise ValueError(
                            f"Conflicting duplicate semantic row: {split}/{sample_id}"
                        )
                else:
                    lookup[sample_id] = row

        manifest_path = args.alignment_root / split / "hik_h25f50_cache_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.limit_per_split >= 0:
            manifest = manifest[: args.limit_per_split]
        expected_ids = [str(row["sample_id"]) for row in manifest]
        if any(str(row.get("event_action", "")) == "walking" for row in manifest):
            raise ValueError(f"Walking row found in {manifest_path}")
        missing = [sample_id for sample_id in expected_ids if sample_id not in lookup]
        extra = sorted(set(lookup) - set(expected_ids))
        if missing or extra:
            raise ValueError(
                f"{split} shard coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
            )
        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "semantic_scores.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for sample_id in expected_ids:
                row = lookup[sample_id]
                if str(row.get("event_action", "")) == "walking":
                    raise ValueError(f"Walking semantic row found: {sample_id}")
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary["splits"][split] = {
            "expected": len(expected_ids),
            "merged": len(lookup),
            "duplicate_identical_rows": duplicate_count,
            "output": str(output_path.resolve()),
        }

    (args.output_root / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
