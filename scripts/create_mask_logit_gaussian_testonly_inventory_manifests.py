#!/usr/bin/env python3
"""Create 37-sample inventory manifests for the nine Gaussian test conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "data/OpenIns3D/output/results_demo_rebuttal_final"
)
DEFAULT_TEST_MANIFEST = Path(
    "data/GIMO/SLICES_888s/test/"
    "final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_"
    "points_1024_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "object_selector"
    / "mask_logit_gaussian_testonly"
    / "inventories"
)
SEVERITIES = ("mild", "moderate", "severe")
SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    samples = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    if len(samples) != 37:
        raise ValueError(f"Expected 37 test samples, found {len(samples)}")

    summary: dict[str, Any] = {
        "source_root": str(source_root),
        "test_manifest": str(args.test_manifest.resolve()),
        "num_test_samples": len(samples),
        "conditions": {},
    }
    for severity in SEVERITIES:
        for seed in SEEDS:
            condition = f"mask_logit_gaussian_{severity}_seed{seed}"
            records: list[dict[str, Any]] = []
            for expected_index, sample in enumerate(samples):
                index = int(sample["index"])
                if index != expected_index:
                    raise ValueError(
                        f"Test index mismatch: expected {expected_index}, got {index}"
                    )
                sample_id = "|".join(str(value) for value in sample["sample_key"])
                scene_name = str(sample["scene_name"])
                output_directory = source_root / f"{scene_name}_{condition}"
                catalog = output_directory / "label_pointcloud_export.json"
                if not catalog.is_file():
                    raise FileNotFoundError(catalog)
                records.append(
                    {
                        "condition": condition,
                        "split": "test",
                        "sample_index": index,
                        "sample_id": sample_id,
                        "sample_uid": f"test:{index:04d}:{sample_id}",
                        "sample_key": sample["sample_key"],
                        "scene_name": scene_name,
                        "sequence_name": sample["sequence_name"],
                        "start_frame": str(sample["start_frame"]),
                        "end_frame": str(sample["end_frame"]),
                        "output_directory": str(output_directory),
                    }
                )
            target = output_root / f"{condition}.json"
            write_json(
                target,
                {
                    "schema_version": 1,
                    "condition": condition,
                    "test_only": True,
                    "num_samples": len(records),
                    "split_counts": {"train": 0, "test": len(records)},
                    "samples": records,
                },
            )
            summary["conditions"][condition] = str(target)
            print(f"{condition}: {len(records)} test samples -> {target}")

    write_json(output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
