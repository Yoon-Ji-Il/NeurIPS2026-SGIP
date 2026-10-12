#!/usr/bin/env python3
"""Audit HIK events at their unchanged native H25F50 timing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.hik_raw_1s2s import (  # noqa: E402
    HIK1s2sTemporalSpec,
    RawHIKIndex,
    audit_test_windows,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default=(
            "data/HIK/SAST/data"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "./"
            "runs_hik/manifests_h25f50"
        ),
    )
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main():
    args = parse_args()
    spec = HIK1s2sTemporalSpec()
    index = RawHIKIndex(args.raw_root)
    manifest, summary = audit_test_windows(index, spec)

    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "hik_test_h25f50.jsonl"
    summary_path = output_root / "hik_test_h25f50_summary.json"
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in manifest
    )
    atomic_write_text(manifest_path, manifest_text)
    atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )

    print(json.dumps(summary, indent=2))
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
