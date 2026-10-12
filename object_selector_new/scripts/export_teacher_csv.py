"""Export teacher results to a long-format CSV for downstream training.

Reads a JSONL of TeacherInputSample-shaped dicts (one per line), runs the
teacher pipeline (rubric LLM scoring + rubric softmax + motion tie-break),
and writes a CSV with one row per (sample, candidate).

Output schema (one row per candidate):
    split, sample_id, object_id, name, category, is_gt,
    want_affordance, want_state_viability, action_priority, scene_grounded_support,
    rubric, rubric_posterior,
    motion_raw, motion_norm, motion_score,
    final_score, fusion_posterior,
    is_in_tie_set, tie_set_size, is_tie_sample, winner_stage,
    is_teacher_selected,
    reasoning_trace_json,
    + every geometry column listed via --geometry-columns
      (these are precomputed in the input JSONL under `objects[i].metrics`)

Input JSONL row schema (unchanged from previous version):
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
from datetime import datetime, timezone
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector_new.common.selector_interfaces import (  # noqa: E402
    MotionCueSpec,
    ObjectCandidate,
    PromptGeometrySpec,
    Ray3D,
    SceneGraph,
    SceneGraphEdge,
    SceneGraphNode,
    TeacherInputSample,
)
from object_selector_new.teacher.fusion_scorer import FusionScoringConfig  # noqa: E402
from object_selector_new.teacher.object_selector import TeacherObjectSelector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_columns(raw: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in raw.split(",") if c.strip())


def _maybe_split_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(float(p.strip()) for p in raw.split(",") if p.strip())


def _maybe_split_indices(raw: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    indices = frozenset(int(part.strip()) for part in raw.split(",") if part.strip())
    if any(index < 0 for index in indices):
        raise ValueError("--sample-indices values must be >= 0")
    return indices


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
        default=(
            "distance,root_centroid_distance_3d,head_centroid_distance_3d,"
            "root_min_point_distance_3d,head_min_point_distance_3d,"
            "face_normal_alignment,face_normal_horizontal_alignment,"
            "root_alignment,root_horizontal_alignment,"
            "torso_alignment,torso_horizontal_alignment,root_torso_alignment,"
            "root_line_distance_xz,torso_line_distance_xz"
        ),
        help="Comma-separated geometry metric column names to mirror from input JSONL.",
    )

    # Prompt geometry exposure
    parser.add_argument(
        "--prompt-primary-columns",
        type=str,
        default="root_horizontal_alignment,face_normal_alignment",
        help="Geometry columns exposed in the prompt as PRIMARY signals "
        "(drive scene_grounded_support).",
    )
    parser.add_argument(
        "--prompt-aux-columns",
        type=str,
        default="root_alignment,torso_alignment,distance",
        help="Geometry columns exposed in the prompt as AUXILIARY (context only).",
    )

    # Motion cue
    parser.add_argument(
        "--motion-cue-columns",
        type=str,
        default="root_horizontal_alignment,face_normal_alignment",
        help="Geometry columns combined into the motion cue used for tie-break.",
    )
    parser.add_argument(
        "--motion-cue-aggregation",
        choices=("sum", "mean", "weighted_sum"),
        default="sum",
    )
    parser.add_argument(
        "--motion-cue-weights",
        type=str,
        default=None,
        help="Comma-separated floats; required when aggregation=weighted_sum.",
    )
    parser.add_argument(
        "--motion-cue-normalize",
        choices=("minmax_within_tieset", "minmax_global", "none"),
        default="minmax_within_tieset",
    )

    # Fusion config
    parser.add_argument("--rubric-temperature", type=float, default=20.0)
    parser.add_argument("--tie-threshold", type=int, default=5)
    parser.add_argument("--motion-weight", type=float, default=1.0)

    # LLM backend
    parser.add_argument(
        "--llm-backend",
        choices=("openai", "qwen", "ollama", "mock"),
        default="mock",
    )
    parser.add_argument("--openai-env-file", type=Path, default=None)
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--qwen-quantization", choices=("4bit", "8bit", "none"), default="4bit")
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=4096,
        help="max_new_tokens for Qwen generation. Long enough to fit per-candidate "
        "rubric scores + 4-key reasoning_trace for ~12 candidates.",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Remote Ollama base URL. Only serialized prompt text is transmitted.",
    )
    parser.add_argument("--ollama-model", type=str, default="qwen3:4b")
    parser.add_argument("--ollama-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--ollama-context-length",
        type=int,
        default=16384,
        help="Ollama context allocation. 16K avoids the large VRAM cost of a model's maximum context.",
    )
    parser.add_argument("--ollama-timeout", type=float, default=300.0)
    parser.add_argument("--ollama-keep-alive", type=str, default="10m")

    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="",
        help="Optional comma-separated absolute sample indexes to process, e.g. 5,39,50,56.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of first sample (0-based) to process. Use with --end-index for sharding.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Index of last sample (exclusive). -1 means until end. Use with --start-index for sharding.",
    )
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
    """Returns (sample, raw_candidate_records).

    IMPORTANT: each ObjectCandidate.metadata carries the full `metrics` dict
    so that motion-cue computation in fusion_scorer can look up arbitrary
    geometry columns by name.
    """
    candidates = []
    raw_records = []
    for c in row["object_candidates"]:
        metrics = dict(c.get("metrics", {}))
        candidates.append(
            ObjectCandidate(
                object_id=str(c["object_id"]),
                name=str(c["name"]),
                category=str(c.get("category", "")),
                position=_vec(c["position"]),
                metadata={
                    **dict(c.get("metadata", {})),
                    "metrics": metrics,
                },
            )
        )
        raw_records.append(
            {
                "object_id": str(c["object_id"]),
                "name": str(c["name"]),
                "category": str(c.get("category", "")),
                "metrics": metrics,
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
            max_new_tokens=args.qwen_max_new_tokens,
        )
    if args.llm_backend == "ollama":
        from object_selector_new.teacher.ollama_client import RemoteOllamaSemanticClient

        return RemoteOllamaSemanticClient(
            base_url=args.ollama_base_url,
            model_name=args.ollama_model,
            max_new_tokens=args.ollama_max_new_tokens,
            context_length=args.ollama_context_length,
            timeout=args.ollama_timeout,
            keep_alive=args.ollama_keep_alive,
        )
    from object_selector_new.teacher.semantic_scorer import MockSemanticLLMClient

    return MockSemanticLLMClient()


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------
def iterate_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


MAX_SAMPLE_RETRIES = 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    geometry_columns = _split_columns(args.geometry_columns)
    sample_indices = _maybe_split_indices(args.sample_indices)

    motion_cue = MotionCueSpec(
        columns=_split_columns(args.motion_cue_columns),
        aggregation=args.motion_cue_aggregation,
        weights=_maybe_split_weights(args.motion_cue_weights),
        normalize=args.motion_cue_normalize,
    )
    geometry_spec = PromptGeometrySpec(
        primary_columns=_split_columns(args.prompt_primary_columns),
        auxiliary_columns=_split_columns(args.prompt_aux_columns),
    )
    fusion_config = FusionScoringConfig(
        rubric_temperature=args.rubric_temperature,
        tie_threshold=args.tie_threshold,
        motion_weight=args.motion_weight,
        motion_cue=motion_cue,
    )

    llm = build_llm_client(args)
    teacher = TeacherObjectSelector(
        llm_client=llm,
        fusion_config=fusion_config,
        geometry_spec=geometry_spec,
    )

    base_columns = [
        "split",
        "sample_id",
        "object_id",
        "name",
        "category",
        "is_gt",
        "want_affordance",
        "want_state_viability",
        "action_priority",
        "scene_grounded_support",
        "rubric",
        "rubric_posterior",
        "motion_raw",
        "motion_norm",
        "motion_score",
        "final_score",
        "fusion_posterior",
        "is_in_tie_set",
        "tie_set_size",
        "is_tie_sample",
        "winner_stage",
        "is_teacher_selected",
        "reasoning_trace_json",
    ]
    fieldnames = base_columns + list(geometry_columns)
    error_log_path = args.output_csv.parent / f"{args.output_csv.stem}.errors.jsonl"
    progress_log_path = args.output_csv.parent / f"{args.output_csv.stem}.progress.log"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output_csv.open("w", encoding="utf-8", newline="") as f,
        error_log_path.open("w", encoding="utf-8") as error_f,
        progress_log_path.open("w", encoding="utf-8") as progress_f,
    ):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def log_progress(event: str, **fields: Any) -> None:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
            line = f"[{stamp}] {event} {message}".rstrip()
            print(line, file=sys.stderr, flush=True)
            progress_f.write(line + "\n")
            progress_f.flush()

        count = 0
        tie_count = 0
        skipped_count = 0
        sample_idx = -1
        for row in iterate_jsonl(args.input_jsonl):
            sample_idx += 1
            if sample_indices and sample_idx not in sample_indices:
                continue
            if sample_idx < args.start_index:
                continue
            if args.end_index >= 0 and sample_idx >= args.end_index:
                break
            split = str(row.get("split", ""))
            sample, raw_records = build_teacher_input(row)
            log_progress(
                "start",
                sample_idx=sample_idx,
                sample_id=sample.sample_id,
                split=split,
                want=sample.want,
                candidates=len(raw_records),
            )
            last_error: Exception | None = None
            result = None
            for attempt in range(1, MAX_SAMPLE_RETRIES + 1):
                try:
                    result = teacher.select(sample)
                    break
                except Exception as exc:
                    last_error = exc
                    log_progress(
                        "retry_fail",
                        sample_idx=sample_idx,
                        sample_id=sample.sample_id,
                        attempt=attempt,
                        max_retries=MAX_SAMPLE_RETRIES,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            if result is None:
                skipped_count += 1
                error_payload = {
                    "sample_idx": sample_idx,
                    "sample_id": sample.sample_id,
                    "split": split,
                    "want": sample.want,
                    "error_type": type(last_error).__name__ if last_error is not None else "UnknownError",
                    "error": str(last_error) if last_error is not None else "unknown failure",
                    "traceback": (
                        "".join(
                            traceback.format_exception(
                                type(last_error),
                                last_error,
                                last_error.__traceback__,
                            )
                        )
                        if last_error is not None
                        else ""
                    ),
                }
                error_f.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                error_f.flush()
                log_progress(
                    "skip",
                    sample_idx=sample_idx,
                    sample_id=sample.sample_id,
                    split=split,
                    want=sample.want,
                )
                continue

            tie_set_size = len(result.tie_set)
            is_tie_sample = tie_set_size >= 2
            if is_tie_sample:
                tie_count += 1

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
                    "want_affordance": ss.want_affordance,
                    "want_state_viability": ss.want_state_viability,
                    "action_priority": ss.action_priority,
                    "scene_grounded_support": ss.scene_grounded_support,
                    "rubric": ss.rubric,
                    "rubric_posterior": fs.rubric_posterior,
                    "motion_raw": fs.feature_scores.get("motion_raw", 0.0),
                    "motion_norm": fs.feature_scores.get("motion_norm", 0.0),
                    "motion_score": fs.motion_score,
                    "final_score": fs.final_score,
                    "fusion_posterior": fs.posterior,
                    "is_in_tie_set": int(bool(fs.is_in_tie_set)),
                    "tie_set_size": tie_set_size,
                    "is_tie_sample": int(is_tie_sample),
                    "winner_stage": result.winner_stage,
                    "is_teacher_selected": int(oid == result.final_object),
                    "reasoning_trace_json": json.dumps(dict(ss.reasoning_trace), ensure_ascii=False),
                }
                for col in geometry_columns:
                    csv_row[col] = rec["metrics"].get(col, 0.0)
                writer.writerow(csv_row)
            f.flush()
            log_progress(
                "ok",
                sample_idx=sample_idx,
                sample_id=sample.sample_id,
                candidates=len(raw_records),
                tie=int(is_tie_sample),
            )

            count += 1
            if args.max_samples > 0 and count >= args.max_samples:
                break

        print(
            f"Wrote {count} samples to {args.output_csv} "
            f"[range {args.start_index}:{args.end_index if args.end_index >= 0 else 'end'}] "
            f"(tie_samples={tie_count}/{count}, tie_rate={tie_count / max(count, 1):.2%}, "
            f"skipped={skipped_count}, errors={error_log_path})"
        )


if __name__ == "__main__":
    main()
