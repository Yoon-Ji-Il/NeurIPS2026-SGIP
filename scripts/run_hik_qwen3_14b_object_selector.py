#!/usr/bin/env python3
"""Select one HIK object per H25F50 event with action + geometry context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import (  # noqa: E402
    EVENT_TARGET_TYPE,
    NONE_OBJECT_ID,
    iter_jsonl,
    stable_seed,
)
from object_selector.teacher.qwen_client import LocalQwenSemanticClient  # noqa: E402


INSTRUCTIONS = """You are the object-selection teacher for the Humans in Kitchens motion-forecasting experiment.

Choose exactly one candidate object for the target person's imminent interaction.

Evidence priority:
1. Interpret event_action and the detailed observed HIK actions.
2. Use object label/type to determine functional compatibility.
3. Use geometry_score, distance, body/face alignment, and cone fractions to choose among compatible instances.

geometry_score is an action-independent spatial prior, not a semantic answer. A nearby object with the wrong function must not beat a compatible object. Use __none__ only when the event has no target object, especially walking. Return JSON only, with no markdown or extra text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "qwen3_14b_action_geometry_h25f50_v1",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--query-walking", action="store_true")
    parser.add_argument("--allow-geometry-fallback", action="store_true")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Deterministically divide sample IDs across this many workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index within --num-shards.",
    )
    parser.add_argument(
        "--sample-id-file",
        type=Path,
        default=None,
        help="Optional newline-delimited sample IDs to process.",
    )
    parser.add_argument(
        "--compatible-only",
        action="store_true",
        help=(
            "Restrict correction prompts to the event action's compatible "
            "object type."
        ),
    )
    return parser.parse_args()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {str(key): rounded(item, digits) for key, item in value.items()}
    return value


def stable_shuffle(sample_id: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed = int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    output = list(candidates)
    random.Random(seed).shuffle(output)
    return output


def build_prompt(
    graph: dict[str, Any],
    *,
    compatible_only: bool = False,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    candidates = list(graph.get("objects", []))
    if compatible_only:
        expected_type = EVENT_TARGET_TYPE.get(str(graph.get("event_action", "")))
        if expected_type is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("label") == expected_type
            ]
        if not candidates:
            raise ValueError(
                f"No compatible candidate for {graph['sample_id']} "
                f"and action={graph.get('event_action')}"
            )
    candidates = stable_shuffle(str(graph["sample_id"]), candidates)
    if len(candidates) > 26:
        raise ValueError(f"At most 26 candidates are supported, got {len(candidates)}")
    choices = []
    payload_candidates = []
    metric_names = (
        "geometry_score",
        "root_bbox_distance_3d",
        "head_bbox_distance_3d",
        "hand_bbox_distance_3d",
        "root_horizontal_alignment",
        "torso_horizontal_alignment",
        "face_horizontal_alignment",
        "torso_cone_fraction",
        "face_cone_fraction",
        "root_heading_cone_fraction",
        "root_line_distance_xz",
        "torso_line_distance_xz",
        "face_line_distance_xz",
    )
    for index, candidate in enumerate(candidates):
        choice = chr(ord("A") + index)
        choices.append((choice, candidate))
        metrics = candidate.get("metrics", {})
        payload_candidates.append(
            {
                "choice": choice,
                "object_id": candidate.get("object_id", ""),
                "label": candidate.get("label", ""),
                "candidate_reason": candidate.get("candidate_reason", ""),
                "centroid": candidate.get("centroid", []),
                "bbox_min": candidate.get("bbox_min", []),
                "bbox_max": candidate.get("bbox_max", []),
                "geometry": {
                    name: metrics.get(name)
                    for name in metric_names
                    if name in metrics
                },
            }
        )
    payload = {
        "sample_id": graph["sample_id"],
        "event_action": graph.get("event_action", ""),
        "active_actions_last_observed": graph.get(
            "active_actions_last_observed",
            [],
        ),
        "active_actions_observed": graph.get("active_actions_observed", []),
        "expected_object_type_hint": graph.get("expected_object_type_hint"),
        "human_final_frame": graph.get("human_final_frame", {}),
        "candidate_objects": payload_candidates,
        "valid_choices": [choice for choice, _ in choices],
        "output_schema": {
            "sample_id": graph["sample_id"],
            "choice": "one valid choice letter",
            "selected_object": "object_id paired with that choice",
            "confidence": "number from 0 to 1",
            "rationale_short": "one short sentence",
        },
    }
    prompt = (
        f"{INSTRUCTIONS}\n"
        f"{json.dumps(rounded(payload), ensure_ascii=False, separators=(',', ':'))}"
    )
    return prompt, choices


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    fence = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in response: {cleaned[:300]!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(payload, dict):
        raise ValueError("Qwen response JSON must be an object")
    return payload


def validate_response(
    payload: dict[str, Any],
    graph: dict[str, Any],
    choices: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    choice_lookup = {choice: candidate for choice, candidate in choices}
    object_lookup = {
        str(candidate["object_id"]): (choice, candidate)
        for choice, candidate in choices
    }
    choice = str(payload.get("choice", "")).strip().upper()
    selected = str(payload.get("selected_object", "")).strip()
    reported_selected = selected
    status = "ok"
    if choice in choice_lookup:
        expected = str(choice_lookup[choice]["object_id"])
        if selected and selected != expected:
            # The letter is an exact closed-set key. Qwen occasionally
            # abbreviates, spell-corrects, or copies another object ID even
            # though its valid choice points to the intended candidate.
            status = "ok_choice_authoritative"
        selected = expected
    elif selected in object_lookup:
        choice = object_lookup[selected][0]
    else:
        raise ValueError(
            f"Response has no valid choice/object: choice={choice!r}, "
            f"selected_object={selected!r}"
        )
    confidence = float(payload.get("confidence", 0.5))
    if not math.isfinite(confidence):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)
    candidate = object_lookup[selected][1]
    return {
        "sample_id": graph["sample_id"],
        "split": graph["split"],
        "kitchen": graph["kitchen"],
        "pid": graph["pid"],
        "future_anchor": graph["future_anchor"],
        "event_action": graph["event_action"],
        "choice": choice,
        "selected_object_id": selected,
        "selected_object_name": candidate.get("label", ""),
        "confidence": confidence,
        "rationale_short": str(payload.get("rationale_short", "")).strip(),
        "reported_selected_object": reported_selected,
        "selected_geometry_score": candidate.get("metrics", {}).get(
            "geometry_score"
        ),
        "status": status,
    }


def geometry_fallback(
    graph: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    candidates = list(graph.get("objects", []))
    if graph.get("event_action") == "walking":
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("object_id") == NONE_OBJECT_ID
            ),
            None,
        )
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                float(candidate.get("metrics", {}).get("geometry_score", 0.0)),
                str(candidate.get("object_id", "")),
            ),
            default=None,
        )
    if selected is None:
        raise ValueError(f"No candidate available for {graph['sample_id']}")
    return {
        "sample_id": graph["sample_id"],
        "split": graph["split"],
        "kitchen": graph["kitchen"],
        "pid": graph["pid"],
        "future_anchor": graph["future_anchor"],
        "event_action": graph["event_action"],
        "choice": "",
        "selected_object_id": selected["object_id"],
        "selected_object_name": selected.get("label", ""),
        "confidence": 1.0 if selected["object_id"] == NONE_OBJECT_ID else 0.0,
        "rationale_short": status,
        "selected_geometry_score": selected.get("metrics", {}).get(
            "geometry_score"
        ),
        "status": status,
    }


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row["sample_id"])
        for row in iter_jsonl(path)
        if row.get("sample_id")
    }


def selected_splits(value: str) -> tuple[str, ...]:
    return ("train", "test") if value == "all" else (value,)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            "--shard-index must satisfy "
            f"0 <= shard-index < {args.num_shards}, got {args.shard_index}"
        )
    requested_ids: set[str] | None = None
    if args.sample_id_file is not None:
        requested_ids = {
            line.strip()
            for line in args.sample_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not requested_ids:
            raise ValueError("--sample-id-file contains no sample IDs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    client: LocalQwenSemanticClient | None = None
    processed = failed = auto_none = 0

    for split in selected_splits(args.split):
        graph_path = args.scene_graph_root / split / "samples.jsonl"
        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selections.jsonl"
        seen = set() if args.no_resume else existing_ids(output_path)
        mode = "w" if args.no_resume else "a"
        split_processed = 0

        with output_path.open(mode, encoding="utf-8") as output:
            for graph in iter_jsonl(graph_path):
                sample_id = str(graph["sample_id"])
                assigned_shard = (
                    stable_seed("qwen_shard", sample_id) % args.num_shards
                )
                if assigned_shard != args.shard_index:
                    continue
                if requested_ids is not None and sample_id not in requested_ids:
                    continue
                if args.sample_id and sample_id != args.sample_id:
                    continue
                if sample_id in seen:
                    continue
                if args.limit >= 0 and split_processed >= args.limit:
                    break

                if (
                    graph.get("event_action") == "walking"
                    and not args.query_walking
                ):
                    result = geometry_fallback(
                        graph,
                        status="auto_none_for_walking",
                    )
                    auto_none += 1
                elif args.backend == "mock":
                    result = geometry_fallback(
                        graph,
                        status="mock_geometry_top1",
                    )
                else:
                    if client is None:
                        client = LocalQwenSemanticClient(
                            model_name=args.model_name,
                            device=args.device,
                            torch_dtype=args.torch_dtype,
                            quantization=args.quantization,
                            max_new_tokens=args.max_new_tokens,
                        )
                    prompt, choices = build_prompt(
                        graph,
                        compatible_only=args.compatible_only,
                    )
                    last_error: Exception | None = None
                    result = None
                    for _ in range(max(args.max_retries, 1)):
                        try:
                            raw = client.complete(prompt)
                            result = validate_response(
                                extract_json_object(raw),
                                graph,
                                choices,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                    if result is None:
                        failed += 1
                        if not args.allow_geometry_fallback:
                            print(
                                f"[failed] {sample_id}: "
                                f"{type(last_error).__name__}: {last_error}",
                                file=sys.stderr,
                            )
                            continue
                        result = geometry_fallback(
                            graph,
                            status=(
                                "qwen_failed_geometry_fallback:"
                                f"{type(last_error).__name__}"
                            ),
                        )

                output.write(
                    json.dumps(result, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                output.flush()
                seen.add(sample_id)
                processed += 1
                split_processed += 1
                print(
                    f"[{split}] {sample_id} -> "
                    f"{result['selected_object_id']} ({result['status']})"
                )

    summary = {
        "backend": args.backend,
        "model_name": args.model_name if args.backend == "qwen" else "",
        "scene_graph_root": str(args.scene_graph_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "processed_this_run": processed,
        "failed_this_run": failed,
        "auto_none_for_walking_this_run": auto_none,
        "resume": not args.no_resume,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_id_file": (
            str(args.sample_id_file.resolve())
            if args.sample_id_file is not None
            else ""
        ),
        "compatible_only": args.compatible_only,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
