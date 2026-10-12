"""Export teacher results to a long-format CSV for downstream training.

Reads a JSONL of TeacherInputSample-shaped dicts (one per line), runs the
teacher pipeline on each, and writes a CSV with one row per (sample, candidate).

Output schema (one row per candidate):
    split, sample_id, object_id, name, category, is_gt,
    fusion_total, fusion_posterior, semantic_total, match, ease,
    face_alignment_norm, body_alignment_norm,
    is_teacher_selected,
    + every geometry column listed via --geometry-columns
      (these are precomputed in the input JSONL under `objects[i].metrics`)

Input JSONL row schema:
    {
      "sample_id": str,
      "split": "train" | "test",
      "want": str,
      "scene_graph": { ... SceneGraph dict ... },
      "object_candidates": [
          {"object_id": str, "name": str, "category": str,
           "position": [x, y, z],
           "metrics": {distance: ..., face_normal_alignment: ..., ...},
           "is_gt": bool},
          ...
      ],
      "face_normal_ray": {"origin": [...], "direction": [...]},
      "root_direction": [...], "torso_direction": [...],
      "pelvis_position": [...], "head_position": [...] (optional)
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.common.selector_interfaces import (  # noqa: E402
    ObjectCandidate,
    Ray3D,
    SceneGraph,
    SceneGraphEdge,
    SceneGraphNode,
    TeacherInputSample,
)
from object_selector_new.geometry_features import DEFAULT_GEOMETRY_COLUMNS  # noqa: E402
from object_selector_new.teacher.fusion_scorer import FusionScoringConfig  # noqa: E402
from object_selector_new.teacher.object_selector import TeacherObjectSelector  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher and export long-format CSV.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--geometry-columns",
        type=str,
        default=",".join(DEFAULT_GEOMETRY_COLUMNS),
        help="Comma-separated geometry metric column names to mirror from input JSONL.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("openai", "qwen", "mock"),
        default="mock",
        help="Which semantic LLM client to use.",
    )
    parser.add_argument(
        "--openai-env-file",
        type=Path,
        default=None,
        help="Optional .env file for OpenAI; only used when --llm-backend openai.",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3-14B",
    )
    parser.add_argument(
        "--qwen-quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
    )
    parser.add_argument(
        "--fusion-temperature",
        type=float,
        default=1.0,
        help="Temperature inside fusion_scorer's posterior softmax (separate from student-side T).",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="-1 = all.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sample reconstruction
# ---------------------------------------------------------------------------
def _vec(v: Any) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def build_scene_graph(payload: dict[str, Any]) -> SceneGraph:
    nodes = tuple(
        SceneGraphNode(
            node_id=str(n["node_id"]),
            node_type=str(n.get("node_type", "object")),
            name=str(n.get("name", "")),
            metadata=dict(n.get("metadata", {})),
        )
        for n in payload.get("nodes", [])
    )
    edges = tuple(
        SceneGraphEdge(
            source_id=str(e["source_id"]),
            target_id=str(e["target_id"]),
            relation=str(e.get("relation", "")),
            metadata=dict(e.get("metadata", {})),
        )
        for e in payload.get("edges", [])
    )
    return SceneGraph(nodes=nodes, edges=edges, metadata=dict(payload.get("metadata", {})))


def build_teacher_input(row: dict[str, Any]) -> tuple[TeacherInputSample, list[dict[str, Any]]]:
    """Returns (sample, raw_candidate_records). The raw records carry per-candidate
    metrics + is_gt, which the exporter writes verbatim to CSV alongside teacher outputs."""
    candidates = []
    raw_records = []
    for c in row["object_candidates"]:
        candidates.append(
            ObjectCandidate(
                object_id=str(c["object_id"]),
                name=str(c["name"]),
                category=str(c.get("category", "")),
                position=_vec(c["position"]),
                metadata=dict(c.get("metadata", {})),
            )
        )
        raw_records.append(
            {
                "object_id": str(c["object_id"]),
                "name": str(c["name"]),
                "category": str(c.get("category", "")),
                "metrics": dict(c.get("metrics", {})),
                "is_gt": bool(c.get("is_gt", False)),
            }
        )

    face = row["face_normal_ray"]
    sample = TeacherInputSample(
        sample_id=str(row["sample_id"]),
        want=str(row["want"]),
        supporting_actions=(),
        scene_graph=build_scene_graph(row.get("scene_graph", {})),
        object_candidates=tuple(candidates),
        face_normal_ray=Ray3D(origin=_vec(face["origin"]), direction=_vec(face["direction"])),
        root_direction=_vec(row["root_direction"]),
        torso_direction=_vec(row["torso_direction"]),
        pelvis_position=_vec(row["pelvis_position"]),
        head_position=_vec(row["head_position"]) if row.get("head_position") is not None else None,
        metadata=dict(row.get("metadata", {})),
    )
    return sample, raw_records


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------
def build_llm_client(args: argparse.Namespace) -> Any:
    if args.llm_backend == "openai":
        from object_selector_new.teacher.llm_reasoning_adapter import OpenAIReasoningAdapter
        return OpenAIReasoningAdapter(env_file=args.openai_env_file)
    if args.llm_backend == "qwen":
        from object_selector_new.teacher.qwen_client import LocalQwenSemanticClient
        return LocalQwenSemanticClient(
            model_name=args.qwen_model,
            quantization=args.qwen_quantization,
        )
    from object_selector_new.teacher.semantic_scorer import MockSemanticLLMClient
    return MockSemanticLLMClient()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def iterate_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def require_geometry_metric(
    *,
    sample_id: str,
    object_id: str,
    metrics: dict[str, Any],
    column: str,
) -> float:
    if column not in metrics:
        raise ValueError(
            f"Missing geometry metric {column!r} at sample_id={sample_id} object_id={object_id}"
        )
    return float(metrics[column])


def main() -> None:
    args = parse_args()
    geometry_columns = tuple(c.strip() for c in args.geometry_columns.split(",") if c.strip())

    llm = build_llm_client(args)
    fusion_config = FusionScoringConfig(temperature=args.fusion_temperature)
    teacher = TeacherObjectSelector(llm_client=llm, fusion_config=fusion_config)

    base_columns = [
        "split", "sample_id", "object_id", "name", "category", "is_gt",
        "fusion_total", "fusion_posterior", "semantic_total", "match", "ease",
        "face_alignment_norm", "body_alignment_norm",
        "is_teacher_selected",
    ]
    fieldnames = base_columns + list(geometry_columns)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for row in iterate_jsonl(args.input_jsonl):
            split = str(row.get("split", ""))
            sample, raw_records = build_teacher_input(row)
            result = teacher.select(sample)

            for rec in raw_records:
                oid = rec["object_id"]
                fs = result.fusion_scores.get(oid)
                ss = result.semantic_scores.get(oid)
                if fs is None or ss is None:
                    raise RuntimeError(f"Missing teacher output for {oid}")
                csv_row: dict[str, Any] = {
                    "split": split,
                    "sample_id": sample.sample_id,
                    "object_id": oid,
                    "name": rec["name"],
                    "category": rec["category"],
                    "is_gt": int(bool(rec["is_gt"])),
                    "fusion_total": fs.final_score,
                    "fusion_posterior": fs.posterior,
                    "semantic_total": ss.semantic_total,
                    "match": ss.match,
                    "ease": ss.ease,
                    "face_alignment_norm": fs.feature_scores["face_alignment_norm"],
                    "body_alignment_norm": fs.feature_scores["body_alignment_norm"],
                    "is_teacher_selected": int(oid == result.final_object),
                }
                for col in geometry_columns:
                    csv_row[col] = require_geometry_metric(
                        sample_id=sample.sample_id,
                        object_id=oid,
                        metrics=rec["metrics"],
                        column=col,
                    )
                writer.writerow(csv_row)

            count += 1
            if args.max_samples > 0 and count >= args.max_samples:
                break

        print(f"Wrote {count} samples to {args.output_csv}")


if __name__ == "__main__":
    main()
