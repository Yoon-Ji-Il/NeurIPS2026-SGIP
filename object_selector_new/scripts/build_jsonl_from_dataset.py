"""Build the input JSONL for export_teacher_csv from the project's existing
dataset CSV + scene-graph layout.

Inputs (defaults):
  --dataset-csv     /path/to/dataset/dataset_3_with_want.csv
  --scene-graph-root /path/to/object_selector/scene_graphs
  --output-jsonl    /path/to/runs/jsonl/dataset_v1.jsonl

The dataset CSV provides (sample_id, want, GT_instance, training-flag).
The scene graph (one jsonl per split, samples.jsonl) provides geometry +
human_final_frame pose.

We join on sample_id = "{scene}|{sequence_path}|{start_frame}|{end_frame}".

Output JSONL conforms to the schema consumed by
  object_selector_new/scripts/export_teacher_csv.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CSV = REPO_ROOT / "dataset" / "dataset_3_with_want.csv"
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "runs" / "jsonl" / "dataset_v1.jsonl"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build JSONL from project dataset + scene graphs.")
    p.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    p.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    p.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    p.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Which splits to merge into one JSONL (CSV training=1 → train, =0 → test).",
    )
    p.add_argument(
        "--face-origin-source",
        choices=("eye_mid", "head"),
        default="eye_mid",
        help="Which point in human_final_frame to use as face_normal_ray.origin.",
    )
    p.add_argument("--limit", type=int, default=-1, help="-1 = all rows.")
    return p.parse_args()


def normalize_number_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def sample_id_from_row(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scene", "")).strip(),
            str(row.get("sequence_path", "")).strip(),
            normalize_number_text(row.get("start_frame", "")),
            normalize_number_text(row.get("end_frame", "")),
        ]
    )


def normalize_gt_instance(value: Any) -> str:
    """CSV 'GT_instance' -> 'instance_NNNN'."""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("instance_"):
        return text
    try:
        number = int(float(text))
    except ValueError:
        match = re.search(r"(\d+)", text)
        if not match:
            return text
        number = int(match.group(1))
    return f"instance_{number:04d}"


def split_from_training_flag(value: Any) -> str:
    text = str(value).strip()
    if text == "1":
        return "train"
    if text == "0":
        return "test"
    return ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_dataset_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cleaned = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            sid = sample_id_from_row(cleaned)
            if not sid.split("|")[1]:
                continue
            cleaned["sample_id"] = sid
            cleaned["gt_object_id"] = normalize_gt_instance(cleaned.get("GT_instance", ""))
            cleaned["split"] = split_from_training_flag(cleaned.get("training", ""))
            out.setdefault(sid, cleaned)
    return out


def iter_scene_graphs(scene_graph_root: Path, split: str) -> Iterable[dict[str, Any]]:
    path = scene_graph_root / split / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene graph jsonl: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _vec(v: Any) -> list[float]:
    if v is None:
        raise ValueError("missing 3d vector")
    return [float(v[0]), float(v[1]), float(v[2])]


def build_jsonl_row(
    scene_graph: dict[str, Any],
    metadata: dict[str, Any],
    split: str,
    face_origin_source: str,
) -> dict[str, Any]:
    hff = scene_graph.get("human_final_frame")
    if not isinstance(hff, dict):
        raise ValueError(f"sample {scene_graph.get('sample_id')} missing human_final_frame")

    face_origin_key = "eye_mid" if face_origin_source == "eye_mid" else "head"
    face_origin = hff.get(face_origin_key) or hff.get("head") or hff.get("pelvis")

    gt_id = metadata.get("gt_object_id", "")
    objects_out = []
    for obj in scene_graph.get("objects", []):
        oid = str(obj.get("object_id", ""))
        objects_out.append(
            {
                "object_id": oid,
                "name": str(obj.get("label", "")),
                "category": str(obj.get("label", "")),
                "position": _vec(obj.get("centroid")),
                "metrics": dict(obj.get("metrics", {})),
                "is_gt": bool(oid == gt_id) if gt_id else False,
            }
        )

    if not objects_out:
        raise ValueError(f"sample {scene_graph.get('sample_id')} has no objects")

    gt_count = sum(1 for o in objects_out if o["is_gt"])
    if gt_count > 1:
        raise ValueError(f"sample {scene_graph.get('sample_id')} has {gt_count} GT (expected 1)")

    scene_nodes = [
        {
            "node_id": str(obj.get("object_id", "")),
            "node_type": "object",
            "name": str(obj.get("label", "")),
            "metadata": {"semantic_label_id": obj.get("semantic_label_id")},
        }
        for obj in scene_graph.get("objects", [])
    ]

    return {
        "sample_id": str(scene_graph.get("sample_id", "")),
        "split": split,
        "want": str(metadata.get("want", "")).strip(),
        "scene_graph": {
            "nodes": scene_nodes,
            "edges": [],
            "metadata": {
                "scene_name": scene_graph.get("scene_name", ""),
                "sequence_name": scene_graph.get("sequence_name", ""),
                "coordinate_frame": scene_graph.get("coordinate_frame", ""),
            },
        },
        "object_candidates": objects_out,
        "face_normal_ray": {
            "origin": _vec(face_origin),
            "direction": _vec(hff.get("face_normal_direction")),
        },
        "root_direction": _vec(hff.get("root_direction")),
        "torso_direction": _vec(hff.get("torso_direction")),
        "pelvis_position": _vec(hff.get("pelvis")),
        "head_position": _vec(hff.get("head")) if hff.get("head") is not None else None,
        "metadata": {
            "scene": metadata.get("scene", ""),
            "sequence_path": metadata.get("sequence_path", ""),
            "start_frame": metadata.get("start_frame", ""),
            "end_frame": metadata.get("end_frame", ""),
            "action": metadata.get("action", ""),
            "object": metadata.get("object", ""),
            "GT_instance": metadata.get("GT_instance", ""),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    metadata_by_sample = load_dataset_metadata(args.dataset_csv)
    print(f"Loaded {len(metadata_by_sample)} CSV samples from {args.dataset_csv}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    by_split: dict[str, int] = {}
    no_meta = 0
    no_want = 0
    errors: list[tuple[str, str]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for split in splits:
            for sg in iter_scene_graphs(args.scene_graph_root, split):
                sid = str(sg.get("sample_id", ""))
                meta = metadata_by_sample.get(sid)
                if meta is None:
                    no_meta += 1
                    continue
                want = str(meta.get("want", "")).strip()
                if not want:
                    no_want += 1
                    continue
                try:
                    row = build_jsonl_row(
                        sg,
                        meta,
                        split=split,
                        face_origin_source=args.face_origin_source,
                    )
                except Exception as exc:
                    errors.append((sid, str(exc)))
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                by_split[split] = by_split.get(split, 0) + 1
                if args.limit > 0 and written >= args.limit:
                    break
            if args.limit > 0 and written >= args.limit:
                break

    print(f"\nWrote {written} rows to {args.output_jsonl}")
    print(f"Per split: {by_split}")
    print(f"Skipped (no CSV metadata): {no_meta}")
    print(f"Skipped (no want label):   {no_want}")
    if errors:
        print(f"Errors: {len(errors)} — first 5:")
        for sid, msg in errors[:5]:
            print(f"  {sid}: {msg}")


if __name__ == "__main__":
    main()
