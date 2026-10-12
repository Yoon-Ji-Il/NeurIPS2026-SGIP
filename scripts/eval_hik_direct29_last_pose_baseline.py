#!/usr/bin/env python3
"""Evaluate the zero-residual direct29 initialization on HIK non-walking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hik_metrics import compute_hik_horizon_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direct29-root",
        type=Path,
        default=Path(
            "data/HIK/"
            "hik_preprocessed/H25F50_nonwalking_direct29"
        ),
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "runs_hik/hik_direct29_last_pose_baseline_nonwalking.json",
    )
    return parser.parse_args()


def metric_for_indices(predicted, target, indices):
    return compute_hik_horizon_metrics(
        predicted[indices],
        target[indices],
        input_seq_len=25,
        output_seq_len=50,
        fps=25,
    )


def main():
    args = parse_args()
    cache = args.direct29_root / "SLICES_888s" / args.split
    observed = torch.load(
        cache / "joints29_input.pth",
        map_location="cpu",
        weights_only=False,
    ).float()
    future = torch.load(
        cache / "joints29_label.pth",
        map_location="cpu",
        weights_only=False,
    ).float()
    manifest = json.loads(
        (cache / "hik_direct29_cache_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    last = observed[:, -1]
    last_root = last[:, [0]]
    last_relative = last - last_root
    predicted_future = (
        last_root[:, None]
        + last_relative[:, None]
    ).expand(-1, 50, -1, -1).contiguous()
    predicted = torch.cat([observed, predicted_future], dim=1)
    target = torch.cat([observed, future], dim=1)
    all_indices = torch.arange(len(manifest))
    by_action = {}
    for action in sorted({row["event_action"] for row in manifest}):
        indices = torch.tensor(
            [
                index
                for index, row in enumerate(manifest)
                if row["event_action"] == action
            ],
            dtype=torch.long,
        )
        by_action[action] = {
            "samples": len(indices),
            **metric_for_indices(predicted, target, indices),
        }
    result = {
        "split": args.split,
        "samples": len(manifest),
        "representation": "native_hik_poses3d_direct29",
        "joint_count": 29,
        "baseline": "last_observed_root_and_root_relative_pose",
        "warning": (
            "Native 29-joint MPJPE is not numerically interchangeable with "
            "the previous SMPL-X 23-joint metric."
        ),
        "overall": metric_for_indices(
            predicted,
            target,
            all_indices,
        ),
        "by_action": by_action,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
