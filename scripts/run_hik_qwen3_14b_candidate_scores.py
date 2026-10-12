#!/usr/bin/env python3
"""Score every non-walking HIK candidate with the legacy match/ease rubric.

This is the HIK counterpart of the semantic stage used by
``run_remote_additive_geometry_teacher.py``.  Qwen only supplies integer
``match`` and ``ease`` scores.  Geometry and privileged future-motion fusion
are deliberately handled by a separate deterministic script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import iter_jsonl, stable_seed  # noqa: E402
from object_selector.teacher.qwen_client import LocalQwenSemanticClient  # noqa: E402


DEFAULT_ALIGNMENT_ROOT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking/SLICES_888s"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runs_hik" / "qwen3_14b_h25f50_nonwalking_candidate_semantics"
)

INSTRUCTIONS = """You are the semantic scoring teacher for HIK object selection.

Score every candidate object for the target person's event_action.
For each candidate, output:
- match: integer 1..5 for functional compatibility with the event/action.
- ease: integer 1..5 for how natural that object type is for the detailed observed actions.

Use event_action, active observed actions, object_id, and object label.  Do not
perform geometric tie-breaking and do not select only one object.  Every
candidate must be scored exactly once.  Keep each reasoning_trace to one short
sentence.  Return strict JSON only, without markdown."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=DEFAULT_ALIGNMENT_ROOT,
        help=(
            "Non-walking HIK cache root. Its manifests define the exact sample "
            "set and order; walking rows in the full scene graph are never queried."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit-per-split", type=int, default=-1)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--backend", choices=("qwen", "mock"), default="qwen")
    parser.add_argument("--model-name", default="Qwen/Qwen3-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def selected_splits(value: str) -> tuple[str, ...]:
    return ("train", "test") if value == "all" else (value,)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    fence = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in response: {cleaned[:300]!r}")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Qwen response must be a JSON object")
    return value


def parse_integer_score(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer in [1,5]")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
    else:
        raise ValueError(f"{field} must be an integer in [1,5]")
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be in [1,5], got {score}")
    return score


def stable_shuffle(sample_id: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    output = list(candidates)
    random.Random(seed).shuffle(output)
    return output


def build_prompt(graph: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    candidates = stable_shuffle(str(graph["sample_id"]), list(graph.get("objects", [])))
    if not candidates:
        raise ValueError(f"No candidates for {graph['sample_id']}")
    payload_candidates = []
    for index, candidate in enumerate(candidates):
        payload_candidates.append(
            {
                "choice": chr(ord("A") + index),
                "object_id": candidate.get("object_id", ""),
                "label": candidate.get("label", ""),
                "candidate_reason": candidate.get("candidate_reason", ""),
            }
        )
    payload = {
        "sample_id": graph["sample_id"],
        "event_action": graph.get("event_action", ""),
        "expected_object_type_hint": graph.get("expected_object_type_hint"),
        "active_actions_last_observed": graph.get("active_actions_last_observed", []),
        "active_actions_observed": graph.get("active_actions_observed", []),
        "candidate_objects": payload_candidates,
        "rubric": {
            "match": "integer 1..5: functional compatibility",
            "ease": "integer 1..5: natural/easy interaction for the observed actions",
            "reasoning_trace": "one short sentence",
        },
        "output_schema": {
            "sample_id": graph["sample_id"],
            "event_action": graph.get("event_action", ""),
            "semantic_scores": [
                {
                    "object_id": "candidate object_id",
                    "match": "integer 1..5",
                    "ease": "integer 1..5",
                    "reasoning_trace": "one short sentence",
                }
            ],
        },
    }
    return (
        f"{INSTRUCTIONS}\nINPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    ), candidates


def validate_response(
    payload: dict[str, Any],
    graph: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(payload.get("sample_id", "")).strip()
    if sample_id != str(graph["sample_id"]):
        raise ValueError(f"sample_id mismatch: {sample_id!r}")
    raw_scores = payload.get("semantic_scores")
    if isinstance(raw_scores, dict):
        raw_scores = [
            {"object_id": object_id, **dict(value)}
            for object_id, value in raw_scores.items()
        ]
    if not isinstance(raw_scores, list):
        raise ValueError("semantic_scores must be a list or object")

    expected = [str(candidate.get("object_id", "")) for candidate in candidates]
    scores: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_scores):
        if not isinstance(raw, dict):
            raise ValueError(f"semantic_scores[{index}] must be an object")
        object_id = str(raw.get("object_id", "")).strip()
        if not object_id or object_id in scores:
            raise ValueError(f"Invalid or duplicate object_id: {object_id!r}")
        match = parse_integer_score(raw.get("match"), f"{object_id}.match")
        ease = parse_integer_score(raw.get("ease"), f"{object_id}.ease")
        trace = " ".join(str(raw.get("reasoning_trace", "")).split()[:24])
        scores[object_id] = {
            "object_id": object_id,
            "match": match,
            "ease": ease,
            "semantic_total": match + ease,
            "reasoning_trace": trace or "omitted",
        }

    missing = [object_id for object_id in expected if object_id not in scores]
    extra = sorted(set(scores) - set(expected))
    if missing or extra:
        raise ValueError(f"candidate mismatch: missing={missing}, extra={extra}")
    return {
        "sample_id": graph["sample_id"],
        "split": graph["split"],
        "kitchen": graph["kitchen"],
        "pid": graph["pid"],
        "future_anchor": graph["future_anchor"],
        "event_action": graph["event_action"],
        "semantic_scores": [scores[object_id] for object_id in expected],
        "status": "ok",
    }


def mock_response(graph: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    expected_label = str(graph.get("expected_object_type_hint") or "")
    scores = []
    for candidate in candidates:
        label = str(candidate.get("label", ""))
        compatible = bool(expected_label) and label == expected_label
        match = 5 if compatible else 1
        ease = 5 if compatible else 2
        scores.append(
            {
                "object_id": str(candidate.get("object_id", "")),
                "match": match,
                "ease": ease,
                "semantic_total": match + ease,
                "reasoning_trace": "mock semantic compatibility score",
            }
        )
    return {
        "sample_id": graph["sample_id"],
        "split": graph["split"],
        "kitchen": graph["kitchen"],
        "pid": graph["pid"],
        "future_anchor": graph["future_anchor"],
        "event_action": graph["event_action"],
        "semantic_scores": scores,
        "status": "mock",
    }


def load_alignment_ids(root: Path, split: str) -> list[str]:
    path = root / split / "hik_h25f50_cache_manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = []
    for row in rows:
        if str(row.get("event_action", "")) == "walking":
            raise ValueError(f"Walking row found in non-walking manifest: {row}")
        ids.append(str(row["sample_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return ids


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row.get("sample_id", ""))
        for row in iter_jsonl(path)
        if row.get("sample_id")
    }


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    args.output_root.mkdir(parents=True, exist_ok=True)

    client: LocalQwenSemanticClient | None = None
    processed = failed = 0
    expected_total = 0
    for split in selected_splits(args.split):
        alignment_ids = load_alignment_ids(args.alignment_root, split)
        if args.sample_id:
            alignment_ids = [
                sample_id for sample_id in alignment_ids if sample_id == args.sample_id
            ]
        if args.limit_per_split >= 0:
            alignment_ids = alignment_ids[: args.limit_per_split]
        expected_total += len(alignment_ids)

        graph_lookup = {
            str(graph["sample_id"]): graph
            for graph in iter_jsonl(args.scene_graph_root / split / "samples.jsonl")
        }
        missing_graphs = [sample_id for sample_id in alignment_ids if sample_id not in graph_lookup]
        if missing_graphs:
            raise KeyError(f"Missing HIK scene graphs: {missing_graphs[:5]}")

        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "semantic_scores.jsonl"
        failure_path = output_dir / "failures.jsonl"
        seen = set() if args.no_resume else existing_ids(output_path)
        mode = "w" if args.no_resume else "a"
        failure_mode = "w" if args.no_resume else "a"
        with (
            output_path.open(mode, encoding="utf-8") as output,
            failure_path.open(failure_mode, encoding="utf-8") as failures,
        ):
            for sample_id in alignment_ids:
                if stable_seed("hik_semantic_shard", sample_id) % args.num_shards != args.shard_index:
                    continue
                if sample_id in seen:
                    continue
                graph = graph_lookup[sample_id]
                if str(graph.get("event_action", "")) == "walking":
                    raise ValueError(f"Walking graph reached semantic scorer: {sample_id}")
                prompt, candidates = build_prompt(graph)
                if args.backend == "mock":
                    result = mock_response(graph, candidates)
                else:
                    if client is None:
                        client = LocalQwenSemanticClient(
                            model_name=args.model_name,
                            device=args.device,
                            torch_dtype=args.torch_dtype,
                            quantization=args.quantization,
                            max_new_tokens=args.max_new_tokens,
                        )
                    result = None
                    last_error: Exception | None = None
                    for _ in range(max(args.max_retries, 1)):
                        try:
                            raw = client.complete(prompt)
                            result = validate_response(
                                extract_json_object(raw),
                                graph,
                                candidates,
                            )
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_error = exc
                    if result is None:
                        failed += 1
                        failures.write(
                            json.dumps(
                                {
                                    "sample_id": sample_id,
                                    "split": split,
                                    "error_type": type(last_error).__name__,
                                    "error": str(last_error),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        failures.flush()
                        print(f"[failed] {sample_id}: {last_error}", file=sys.stderr)
                        continue
                output.write(
                    json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                )
                output.flush()
                seen.add(sample_id)
                processed += 1
                print(f"[{split}] semantic scores: {sample_id}")

    summary = {
        "backend": args.backend,
        "model_name": args.model_name if args.backend == "qwen" else "",
        "scene_graph_root": str(args.scene_graph_root.resolve()),
        "alignment_root": str(args.alignment_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "walking_excluded_by_alignment_manifest": True,
        "expected_rows_selected_this_invocation": expected_total,
        "processed_this_run": processed,
        "failed_this_run": failed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume": not args.no_resume,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failed:
        raise RuntimeError(f"Qwen semantic scoring failed for {failed} samples")


if __name__ == "__main__":
    main()
