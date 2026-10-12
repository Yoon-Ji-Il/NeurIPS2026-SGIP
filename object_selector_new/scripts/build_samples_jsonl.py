#!/usr/bin/env python3
"""Build README-shaped samples.jsonl from the existing scene-graph dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.common.want_taxonomy import validate_canonical_want  # noqa: E402
from object_selector_new.geometry_features import DEFAULT_GEOMETRY_COLUMNS  # noqa: E402


DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_DATASET_CSV = REPO_ROOT / "dataset" / "dataset_3_with_want.csv"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "samples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert scene_graphs JSONL into README-shaped samples.jsonl.")
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--verify-lines", type=int, default=5)
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _parse_int(value: Any, field_name: str) -> int:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} is empty")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def _vector3(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be length-3, got {value!r}")
    return [float(value[0]), float(value[1]), float(value[2])]


def build_sample_id(scene: str, sequence_name: str, start_frame: Any, end_frame: Any) -> str:
    return "|".join(
        [
            _clean_text(scene),
            _clean_text(sequence_name),
            str(_parse_int(start_frame, "start_frame")),
            str(_parse_int(end_frame, "end_frame")),
        ]
    )


def load_dataset_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    duplicate_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            scene = _clean_text(row.get("scene"))
            sequence_name = _clean_text(row.get("sequence_path"))
            want = _clean_text(row.get("want"))
            if not scene or not sequence_name or not want:
                continue
            sample_id = build_sample_id(
                scene,
                sequence_name,
                row.get("start_frame"),
                row.get("end_frame"),
            )
            split = "train" if _clean_text(row.get("training")) == "1" else "test"
            gt_instance = _parse_int(row.get("GT_instance"), "GT_instance")
            payload = {
                "sample_id": sample_id,
                "split": split,
                "want": validate_canonical_want(want),
                "gt_object_id": f"instance_{gt_instance:04d}",
                "gt_label": _clean_text(row.get("object")),
                "action": _clean_text(row.get("action")),
            }
            previous = lookup.get(sample_id)
            if previous is not None:
                duplicate_count += 1
                if previous != payload:
                    raise ValueError(
                        f"Conflicting duplicate dataset rows for sample_id={sample_id}: "
                        f"{previous} vs {payload}"
                    )
                continue
            lookup[sample_id] = payload
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate dataset rows with identical payloads.")
    return lookup


def build_scene_graph(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "node_id": str(obj["object_id"]),
                "node_type": "object",
                "name": str(obj.get("label", "")),
                "metadata": {
                    "category": str(obj.get("label", "")),
                    "semantic_label_id": obj.get("semantic_label_id"),
                },
            }
            for obj in sample.get("objects", [])
        ],
        "edges": [],
        "metadata": {
            "scene_name": sample.get("scene_name"),
            "sequence_name": sample.get("sequence_name"),
            "coordinate_frame": sample.get("coordinate_frame"),
            "source_coordinate_frame": sample.get("source_coordinate_frame"),
            "start_frame": sample.get("start_frame"),
            "end_frame": sample.get("end_frame"),
            "transform": sample.get("transform"),
        },
    }


def build_candidate(obj: dict[str, Any], *, gt_object_id: str) -> dict[str, Any]:
    metrics = obj.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Candidate {obj.get('object_id')} is missing a metrics dict")
    missing = [key for key in DEFAULT_GEOMETRY_COLUMNS if key not in metrics]
    if missing:
        raise ValueError(
            f"Candidate {obj.get('object_id')} is missing geometry metrics: {missing}"
        )
    return {
        "object_id": str(obj["object_id"]),
        "name": str(obj.get("label", "")),
        "category": str(obj.get("label", "")),
        "position": _vector3(obj.get("centroid"), f"{obj.get('object_id')}.centroid"),
        "metrics": {key: float(metrics[key]) for key in DEFAULT_GEOMETRY_COLUMNS},
        "is_gt": str(obj["object_id"]) == gt_object_id,
        "metadata": {
            "semantic_label_id": obj.get("semantic_label_id"),
            "source_ply_path": obj.get("source_ply_path"),
            "raw_num_points": obj.get("raw_num_points"),
            "sampled_num_points_for_metrics": obj.get("sampled_num_points_for_metrics"),
            "bbox_min": obj.get("bbox_min"),
            "bbox_max": obj.get("bbox_max"),
        },
    }


def build_output_row(sample: dict[str, Any], dataset_row: dict[str, str]) -> dict[str, Any]:
    human = sample.get("human_final_frame")
    if not isinstance(human, dict):
        raise ValueError(f"sample {sample.get('sample_id')} is missing human_final_frame")
    face_origin = human.get("eye_mid") or human.get("head") or human.get("pelvis")
    face_direction = human.get("face_normal_direction")
    if face_origin is None or face_direction is None:
        raise ValueError(f"sample {sample.get('sample_id')} is missing face orientation vectors")

    candidates = [
        build_candidate(obj, gt_object_id=dataset_row["gt_object_id"])
        for obj in sample.get("objects", [])
    ]
    gt_count = sum(1 for candidate in candidates if candidate["is_gt"])
    if gt_count != 1:
        raise ValueError(
            f"sample {sample.get('sample_id')} has {gt_count} GT candidates; expected exactly 1"
        )

    return {
        "sample_id": str(sample["sample_id"]),
        "split": dataset_row["split"],
        "want": dataset_row["want"],
        "scene_graph": build_scene_graph(sample),
        "object_candidates": candidates,
        "face_normal_ray": {
            "origin": _vector3(face_origin, "face_normal_ray.origin"),
            "direction": _vector3(face_direction, "face_normal_ray.direction"),
        },
        "root_direction": _vector3(human.get("root_direction"), "root_direction"),
        "torso_direction": _vector3(human.get("torso_direction"), "torso_direction"),
        "pelvis_position": _vector3(human.get("pelvis"), "pelvis_position"),
        "head_position": _vector3(human.get("head"), "head_position"),
        "metadata": {
            "scene_name": sample.get("scene_name"),
            "sequence_name": sample.get("sequence_name"),
            "start_frame": sample.get("start_frame"),
            "end_frame": sample.get("end_frame"),
            "source_coordinate_frame": sample.get("source_coordinate_frame"),
            "gt_object_id": dataset_row["gt_object_id"],
            "gt_label": dataset_row["gt_label"],
            "action": dataset_row["action"],
        },
    }


def write_output(
    *,
    scene_graph_root: Path,
    dataset_lookup: dict[str, dict[str, str]],
    output_jsonl: Path,
) -> tuple[int, dict[str, int]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "test": 0}
    written = 0
    duplicate_scene_rows = 0
    with output_jsonl.open("w", encoding="utf-8") as file:
        for split in ("train", "test"):
            input_path = scene_graph_root / split / "samples.jsonl"
            seen_payloads: dict[str, dict[str, Any]] = {}
            with input_path.open("r", encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    sample_id = str(sample.get("sample_id", "")).strip()
                    comparable = dict(sample)
                    comparable.pop("sample_index", None)
                    previous = seen_payloads.get(sample_id)
                    if previous is not None:
                        duplicate_scene_rows += 1
                        if previous != comparable:
                            raise ValueError(
                                f"Conflicting duplicate scene_graph rows for sample_id={sample_id}"
                            )
                        continue
                    seen_payloads[sample_id] = comparable
                    if sample_id not in dataset_lookup:
                        raise KeyError(f"Missing dataset row for sample_id={sample_id}")
                    dataset_row = dataset_lookup[sample_id]
                    if dataset_row["split"] != split:
                        raise ValueError(
                            f"split mismatch for {sample_id}: scene_graph={split}, dataset={dataset_row['split']}"
                        )
                    output_row = build_output_row(sample, dataset_row)
                    file.write(json.dumps(output_row, ensure_ascii=False))
                    file.write("\n")
                    counts[split] += 1
                    written += 1
    if duplicate_scene_rows:
        print(f"Skipped {duplicate_scene_rows} duplicate scene_graph rows with identical payloads.")
    return written, counts


def verify_first_rows(path: Path, *, expected_keys: tuple[str, ...], num_rows: int) -> None:
    print(f"Expected metric keys ({len(expected_keys)}): {', '.join(expected_keys)}")
    checked = 0
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            if checked >= num_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            candidates = row.get("object_candidates", [])
            print(f"[verify] line={line_index} sample_id={sample_id} candidates={len(candidates)}")
            for candidate in candidates:
                object_id = str(candidate.get("object_id", ""))
                metrics = candidate.get("metrics")
                if not isinstance(metrics, dict):
                    print(f"Missing metrics dict at sample_id={sample_id} object_id={object_id}")
                    raise SystemExit(1)
                missing = [key for key in expected_keys if key not in metrics]
                if missing:
                    print(
                        "Missing metric keys at "
                        f"sample_id={sample_id} object_id={object_id}: {', '.join(missing)}"
                    )
                    raise SystemExit(1)
            checked += 1
    print(f"Verified first {checked} lines successfully.")


def main() -> None:
    args = parse_args()
    dataset_lookup = load_dataset_lookup(args.dataset_csv)
    written, counts = write_output(
        scene_graph_root=args.scene_graph_root,
        dataset_lookup=dataset_lookup,
        output_jsonl=args.output_jsonl,
    )
    print(
        f"Wrote {written} rows to {args.output_jsonl} "
        f"(train={counts['train']}, test={counts['test']})."
    )
    verify_first_rows(
        args.output_jsonl,
        expected_keys=DEFAULT_GEOMETRY_COLUMNS,
        num_rows=max(int(args.verify_lines), 0),
    )


if __name__ == "__main__":
    main()
