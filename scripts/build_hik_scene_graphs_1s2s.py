#!/usr/bin/env python3
"""Build action-aware HIK H25F50 scene graphs without changing GIMO code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import (  # noqa: E402
    DEFAULT_CONE_COSINE,
    write_graphs,
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
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--action", default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--scene-radius", type=float, default=3.5)
    parser.add_argument("--cone-cosine", type=float, default=DEFAULT_CONE_COSINE)
    parser.add_argument("--max-candidates", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_graphs(
        raw_root=args.raw_root,
        output_root=args.output_root,
        split=args.split,
        action=args.action,
        limit=args.limit,
        scene_radius=args.scene_radius,
        cone_cosine=args.cone_cosine,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
