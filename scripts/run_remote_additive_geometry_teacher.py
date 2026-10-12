#!/usr/bin/env python3
"""Run the legacy intent scorer and the fixed additive geometry/motion fusion.

This reproduces the selection rule used to build
``final_object_qwen3_14b_motion_c_final_points_1024.pth``.  The LLM supplies
only ``match`` and ``ease``.  Precomputed, model-independent geometry/motion
features are then added with the historical fixed weights, and the final
object is the argmax of that additive fusion score.  No tie-break is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.common.selector_interfaces import (  # noqa: E402
    ObjectCandidate,
    SemanticScore,
    SemanticScoringResult,
    TeacherInputSample,
    sample_id_from_dataset_row,
)
from object_selector.common.supporting_action_taxonomy import (  # noqa: E402
    get_supporting_actions_for_want,
)
from object_selector.common.want_taxonomy import (  # noqa: E402
    normalize_object_name,
)
from object_selector.teacher.ollama_client import (  # noqa: E402
    RemoteOllamaStructuredClient,
)
from object_selector.teacher.openai_responses_client import (  # noqa: E402
    OpenAIResponsesStructuredClient,
)
from object_selector.teacher.semantic_scorer import (  # noqa: E402
    build_semantic_prompt,
    extract_json_object,
    semantic_response_json_schema,
)


LEGACY_ROOT = Path(".")
DEFAULT_DATASET_CSV = LEGACY_ROOT / "dataset" / "dataset_3_with_want.csv"
DEFAULT_SCENE_GRAPH_ROOT = LEGACY_ROOT / "object_selector" / "scene_graphs"
DEFAULT_FEATURE_TEMPLATE = (
    LEGACY_ROOT
    / "object_selector/teacher_targets/qwen3_14b_sequence_selector_gpu1_5"
    / "motion_augmented/teacher_candidate_scores_motion.csv"
)
DEFAULT_GT_TEMPLATE = (
    LEGACY_ROOT
    / "object_selector/teacher_targets/qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_final/gt_eval_all.csv"
)
DEFAULT_FUSION_SUMMARY = (
    LEGACY_ROOT
    / "object_selector/fusion_weight_runs/qwen3_motion_c_selector_weights_final"
    / "summary.json"
)

GEOMETRY_FEATURES = {
    "face_alignment_norm",
    "body_alignment_norm",
    "face_line_score",
}
MOTION_FEATURES = {
    "future_hand_contact_score",
    "future_root_endpoint_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy match+ease teacher with additive geometry/motion fusion."
    )
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model-name", default="qwen3:4b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="none",
    )
    parser.add_argument("--openai-http-retries", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--feature-template-csv", type=Path, default=DEFAULT_FEATURE_TEMPLATE)
    parser.add_argument("--gt-template-csv", type=Path, default=DEFAULT_GT_TEMPLATE)
    parser.add_argument("--fusion-summary-json", type=Path, default=DEFAULT_FUSION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--semantic-source-csv",
        type=Path,
        default=None,
        help="Offline regression mode: reuse match/ease from this CSV instead of Ollama.",
    )
    return parser.parse_args()


def clean_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key).lstrip("\ufeff").strip(): str(value).strip() if value is not None else ""
        for key, value in row.items()
        if key is not None and str(key).lstrip("\ufeff").strip()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_score(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer in [1,5]")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
    else:
        raise ValueError(f"{field} must be an integer in [1,5]")
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be an integer in [1,5], got {score}")
    return score


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_contiguous(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    previous: tuple[str, str] | None = None
    for row in rows:
        key = (row.get("split", ""), row.get("sample_id", ""))
        if not groups or key != previous:
            groups.append([])
        groups[-1].append(row)
        previous = key
    return groups


def load_dataset_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for raw in csv.DictReader(file):
            row = clean_row(raw)
            sample_id = sample_id_from_dataset_row(row)
            if sample_id in rows and rows[sample_id] != row:
                raise ValueError(f"conflicting duplicate dataset sample_id: {sample_id}")
            rows.setdefault(sample_id, row)
    return rows


def load_scene_payloads(root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for split in ("train", "test"):
        with (root / split / "samples.jsonl").open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    payloads.append(json.loads(line))
    return payloads


def prepare_sample(sample: TeacherInputSample) -> TeacherInputSample:
    candidates = tuple(
        ObjectCandidate(
            object_id=candidate.object_id,
            name=normalize_object_name(candidate.name) or candidate.name,
            category=normalize_object_name(candidate.category) or candidate.category,
            position=candidate.position,
            metadata={
                **candidate.metadata,
                "raw_name": candidate.metadata.get("raw_name", candidate.name),
                "raw_category": candidate.metadata.get("raw_category", candidate.category),
            },
        )
        for candidate in sample.object_candidates
    )
    return TeacherInputSample(
        sample_id=sample.sample_id,
        want=sample.want,
        supporting_actions=tuple(get_supporting_actions_for_want(sample.want)),
        scene_graph=sample.scene_graph,
        object_candidates=candidates,
        face_normal_ray=sample.face_normal_ray,
        root_direction=sample.root_direction,
        torso_direction=sample.torso_direction,
        pelvis_position=sample.pelvis_position,
        head_position=sample.head_position,
        metadata={**sample.metadata, "supporting_actions_source": "canonical_want_taxonomy"},
    )


def sample_with_candidates(
    sample: TeacherInputSample,
    candidates: tuple[ObjectCandidate, ...],
) -> TeacherInputSample:
    return TeacherInputSample(
        sample_id=sample.sample_id,
        want=sample.want,
        supporting_actions=sample.supporting_actions,
        scene_graph=sample.scene_graph,
        object_candidates=candidates,
        face_normal_ray=sample.face_normal_ray,
        root_direction=sample.root_direction,
        torso_direction=sample.torso_direction,
        pelvis_position=sample.pelvis_position,
        head_position=sample.head_position,
        metadata=sample.metadata,
    )


def parse_response_for_sample(text: str, sample: TeacherInputSample) -> SemanticScoringResult:
    payload = extract_json_object(text)
    if str(payload.get("sample_id", "")).strip() != sample.sample_id:
        raise ValueError("semantic response sample_id mismatch")
    if str(payload.get("want", "")).strip() != sample.want:
        raise ValueError("semantic response want mismatch")
    raw_scores = payload.get("semantic_scores")
    if isinstance(raw_scores, Mapping):
        items = [{"object_id": key, **dict(value)} for key, value in raw_scores.items()]
    elif isinstance(raw_scores, list):
        items = raw_scores
    else:
        raise ValueError("semantic_scores must be a list or object")

    expected = [candidate.object_id for candidate in sample.object_candidates]
    scores: dict[str, SemanticScore] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"semantic_scores[{index}] must be an object")
        object_id = str(item.get("object_id", "")).strip()
        if len(expected) == 1 and len(items) == 1 and object_id != expected[0]:
            object_id = expected[0]
        if not object_id or object_id in scores:
            raise ValueError(f"invalid or duplicate semantic object_id: {object_id!r}")
        match = parse_score(item.get("match"), f"{object_id}.match")
        ease = parse_score(item.get("ease"), f"{object_id}.ease")
        trace_value = item.get("reasoning_trace", "")
        if isinstance(trace_value, list):
            trace_value = " ".join(str(part) for part in trace_value)
        trace_words = str(trace_value).strip().split()
        trace = " ".join(trace_words[:24]) or "omitted"
        scores[object_id] = SemanticScore(
            object_id=object_id,
            match=match,
            ease=ease,
            semantic_total=float(match + ease),
            reasoning_trace=(trace,),
        )

    missing = [object_id for object_id in expected if object_id not in scores]
    extra = sorted(set(scores) - set(expected))
    if missing or extra:
        raise ValueError(f"semantic ID mismatch: missing={missing}, extra={extra}")
    return SemanticScoringResult(
        sample_id=sample.sample_id,
        want=sample.want,
        semantic_scores=scores,
        metadata={"semantic_total_source": "computed_match_plus_ease"},
    )


def score_once(
    sample: TeacherInputSample,
    client: RemoteOllamaStructuredClient,
) -> SemanticScoringResult:
    candidate_ids = [candidate.object_id for candidate in sample.object_candidates]
    text = client.complete_json(
        instructions="Return semantic scoring JSON only.",
        prompt=build_semantic_prompt(sample),
        json_schema=semantic_response_json_schema(candidate_ids),
        schema_name="semantic_scoring_result",
    )
    return parse_response_for_sample(text, sample)


def score_with_fallback(
    sample: TeacherInputSample,
    client: RemoteOllamaStructuredClient,
    retries: int,
) -> SemanticScoringResult:
    last_error: Exception | None = None
    for _ in range(max(retries, 1)):
        try:
            return score_once(sample, client)
        except Exception as exc:  # noqa: BLE001 - model output validation retry.
            last_error = exc
    if len(sample.object_candidates) == 1:
        raise last_error or RuntimeError("semantic scoring failed")

    merged: dict[str, SemanticScore] = {}
    for candidate in sample.object_candidates:
        result = score_once(sample_with_candidates(sample, (candidate,)), client)
        merged[candidate.object_id] = result.semantic_scores[candidate.object_id]
    return SemanticScoringResult(
        sample_id=sample.sample_id,
        want=sample.want,
        semantic_scores=merged,
        metadata={"candidate_fallback": "single_candidate_calls"},
    )


@dataclass(frozen=True)
class InputBundle:
    split: str
    payload: dict[str, Any]
    sample: TeacherInputSample
    feature_group: list[dict[str, str]]
    gt_row: dict[str, str]


def build_bundles(args: argparse.Namespace) -> list[InputBundle]:
    dataset_rows = load_dataset_rows(args.dataset_csv)
    payloads = load_scene_payloads(args.scene_graph_root)
    feature_groups = group_contiguous(read_csv(args.feature_template_csv))
    gt_rows = read_csv(args.gt_template_csv)
    if not (len(payloads) == len(feature_groups) == len(gt_rows)):
        raise ValueError(
            "sample count mismatch: "
            f"scene={len(payloads)} feature={len(feature_groups)} gt={len(gt_rows)}"
        )

    bundles: list[InputBundle] = []
    for index, (payload, group, gt_row) in enumerate(zip(payloads, feature_groups, gt_rows)):
        sample_id = str(payload.get("sample_id", ""))
        if sample_id not in dataset_rows:
            raise KeyError(f"no dataset row for sample {sample_id!r}")
        sample = prepare_sample(
            TeacherInputSample.from_scene_graph_and_dataset_row(payload, dataset_rows[sample_id])
        )
        if sample.sample_id != gt_row.get("sample_id") or sample.sample_id != group[0].get("sample_id"):
            raise ValueError(f"sample order mismatch at index {index}: {sample.sample_id}")
        sample_ids = sorted(candidate.object_id for candidate in sample.object_candidates)
        feature_ids = sorted(row.get("object_id", "") for row in group)
        if sample_ids != feature_ids:
            raise ValueError(f"candidate ID mismatch at index {index}: {sample.sample_id}")
        bundles.append(
            InputBundle(
                split=str(payload.get("split", gt_row.get("split", ""))),
                payload=payload,
                sample=sample,
                feature_group=group,
                gt_row=gt_row,
            )
        )
    if args.max_samples > 0:
        bundles = bundles[: args.max_samples]
    return bundles


def semantic_rows_from_source(
    bundles: list[InputBundle],
    source_path: Path,
) -> list[list[dict[str, Any]]]:
    source_groups = group_contiguous(read_csv(source_path))
    if len(source_groups) < len(bundles):
        raise ValueError("semantic source has fewer sample groups than requested")
    output: list[list[dict[str, Any]]] = []
    for index, (bundle, group) in enumerate(zip(bundles, source_groups)):
        if group[0].get("sample_id") != bundle.sample.sample_id:
            raise ValueError(f"semantic source sample mismatch at index {index}")
        by_id = {row["object_id"]: row for row in group}
        rows: list[dict[str, Any]] = []
        for candidate in bundle.sample.object_candidates:
            row = by_id[candidate.object_id]
            match = parse_score(row.get("match"), f"{candidate.object_id}.match")
            ease = parse_score(row.get("ease"), f"{candidate.object_id}.ease")
            rows.append(
                {
                    "sample_index": index,
                    "split": bundle.split,
                    "sample_id": bundle.sample.sample_id,
                    "want": bundle.sample.want,
                    "object_id": candidate.object_id,
                    "name": candidate.name,
                    "category": candidate.category,
                    "match": match,
                    "ease": ease,
                    "semantic_total": float(match + ease),
                    "reasoning_trace": row.get("reasoning_trace", ""),
                }
            )
        output.append(rows)
    return output


def generate_semantic_rows(
    bundles: list[InputBundle],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    if args.semantic_source_csv is not None:
        return semantic_rows_from_source(bundles, args.semantic_source_csv)

    if args.backend == "ollama":
        client = RemoteOllamaStructuredClient(
            base_url=args.ollama_base_url,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            timeout=args.timeout,
            keep_alive=args.keep_alive,
        )
    else:
        api_key = os.environ.get(args.openai_api_key_env, "")
        if not api_key.strip():
            raise RuntimeError(
                f"missing OpenAI API key: set environment variable "
                f"{args.openai_api_key_env}"
            )
        client = OpenAIResponsesStructuredClient(
            api_key=api_key,
            model_name=args.model_name,
            base_url=args.openai_base_url,
            max_output_tokens=args.max_new_tokens,
            reasoning_effort=args.openai_reasoning_effort,
            timeout=args.timeout,
            retry_attempts=args.openai_http_retries,
        )
    error_path = args.output_dir / "semantic_errors.jsonl"
    progress_path = args.output_dir / "progress.log"
    checkpoint_path = args.output_dir / "semantic_checkpoint.jsonl"
    checkpointed: dict[int, list[dict[str, Any]]] = {}
    if checkpoint_path.exists():
        lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    print(
                        f"warning: ignoring truncated final checkpoint line {line_number}",
                        flush=True,
                    )
                    continue
                raise
            if payload.get("backend") != args.backend or payload.get("model_name") != args.model_name:
                raise ValueError(
                    f"checkpoint provider/model mismatch in {checkpoint_path}: "
                    f"{payload.get('backend')}/{payload.get('model_name')}"
                )
            index = int(payload["sample_index"])
            if not 0 <= index < len(bundles):
                raise ValueError(f"checkpoint sample_index out of range: {index}")
            if payload.get("sample_id") != bundles[index].sample.sample_id:
                raise ValueError(f"checkpoint sample_id mismatch at index {index}")
            rows = payload.get("rows")
            if not isinstance(rows, list):
                raise ValueError(f"checkpoint rows are invalid at index {index}")
            expected_ids = sorted(
                candidate.object_id for candidate in bundles[index].sample.object_candidates
            )
            actual_ids = sorted(str(row.get("object_id", "")) for row in rows)
            if actual_ids != expected_ids:
                raise ValueError(f"checkpoint candidate mismatch at index {index}")
            checkpointed[index] = rows

    errors = 0
    with error_path.open("w", encoding="utf-8") as error_file, progress_path.open(
        "a", encoding="utf-8"
    ) as progress_file, checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
        for index, bundle in enumerate(bundles):
            if index in checkpointed:
                skip_line = (
                    f"resume skip index={index} sample_id={bundle.sample.sample_id!r}"
                )
                print(skip_line, flush=True)
                progress_file.write(skip_line + "\n")
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            start_line = (
                f"[{stamp}] start index={index} split={bundle.split} "
                f"sample_id={bundle.sample.sample_id!r} want={bundle.sample.want!r} "
                f"candidates={len(bundle.sample.object_candidates)}"
            )
            print(start_line, flush=True)
            progress_file.write(start_line + "\n")
            progress_file.flush()
            try:
                result = score_with_fallback(bundle.sample, client, args.sample_retries)
                rows = []
                for candidate in bundle.sample.object_candidates:
                    score = result.semantic_scores[candidate.object_id]
                    rows.append(
                        {
                            "sample_index": index,
                            "split": bundle.split,
                            "sample_id": bundle.sample.sample_id,
                            "want": bundle.sample.want,
                            "object_id": candidate.object_id,
                            "name": candidate.name,
                            "category": candidate.category,
                            "match": score.match,
                            "ease": score.ease,
                            "semantic_total": score.semantic_total,
                            "reasoning_trace": " ".join(score.reasoning_trace),
                        }
                    )
                checkpointed[index] = rows
                checkpoint_file.write(
                    json.dumps(
                        {
                            "backend": args.backend,
                            "model_name": args.model_name,
                            "sample_index": index,
                            "sample_id": bundle.sample.sample_id,
                            "rows": rows,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                checkpoint_file.flush()
                ok_line = f"[{stamp}] ok index={index} sample_id={bundle.sample.sample_id!r}"
                print(ok_line, flush=True)
                progress_file.write(ok_line + "\n")
            except Exception as exc:  # noqa: BLE001 - preserve batch diagnostics.
                errors += 1
                payload = {
                    "sample_index": index,
                    "sample_id": bundle.sample.sample_id,
                    "want": bundle.sample.want,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }
                error_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                error_file.flush()
                fail_line = f"[{stamp}] failed index={index} error={str(exc)!r}"
                print(fail_line, flush=True)
                progress_file.write(fail_line + "\n")
            progress_file.flush()
    if errors:
        raise RuntimeError(
            f"semantic generation failed for {errors} samples; see {error_path}"
        )
    missing = [index for index in range(len(bundles)) if index not in checkpointed]
    if missing:
        raise RuntimeError(f"semantic checkpoint is missing sample indices: {missing}")
    return [checkpointed[index] for index in range(len(bundles))]


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def class_matches(pred_name: str, gt_name: str) -> bool:
    pred = normalize_object_name(pred_name) or pred_name.strip().lower().replace("_", " ")
    gt = normalize_object_name(gt_name) or gt_name.strip().lower().replace("_", " ")
    return bool(gt) and pred == gt


def fuse(
    bundles: list[InputBundle],
    semantic_groups: list[list[dict[str, Any]]],
    fusion_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_columns = [str(column) for column in fusion_summary["feature_columns"]]
    semantic_gap = float(fusion_summary["selected_by_cv"]["semantic_gap"])
    weights_by_want = fusion_summary["weights_by_want"]
    candidate_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    gt_eval_rows: list[dict[str, Any]] = []

    for index, (bundle, semantic_group) in enumerate(zip(bundles, semantic_groups)):
        semantic_by_id = {str(row["object_id"]): row for row in semantic_group}
        template_by_id = {row["object_id"]: row for row in bundle.feature_group}
        object_ids = sorted(template_by_id)
        if set(object_ids) != set(semantic_by_id):
            raise ValueError(f"semantic/template candidate mismatch at sample {index}")
        want = bundle.sample.want
        weights = weights_by_want.get(want) or weights_by_want.get("unknown")
        if not isinstance(weights, Mapping):
            raise KeyError(f"no fusion weights for want {want!r}")
        max_semantic = max(float(semantic_by_id[object_id]["semantic_total"]) for object_id in object_ids)

        scored: list[dict[str, Any]] = []
        pool_scores: list[float] = []
        for object_id in object_ids:
            semantic = semantic_by_id[object_id]
            template = template_by_id[object_id]
            feature_values: dict[str, float] = {}
            for column in feature_columns:
                feature_values[column] = (
                    float(semantic["semantic_total"])
                    if column == "semantic_total"
                    else parse_float(template.get(column), 0.0)
                )
            components = {
                column: feature_values[column] * parse_float(weights.get(column), 0.0)
                for column in feature_columns
            }
            semantic_component = components.get("semantic_total", 0.0)
            geometry_component = sum(
                value for column, value in components.items() if column in GEOMETRY_FEATURES
            )
            motion_component = sum(
                value for column, value in components.items() if column in MOTION_FEATURES
            )
            additive_score = sum(components.values())
            in_pool = max_semantic - float(semantic["semantic_total"]) <= semantic_gap + 1.0e-8
            if in_pool:
                pool_scores.append(additive_score)
            scored.append(
                {
                    "object_id": object_id,
                    "semantic": semantic,
                    "template": template,
                    "features": feature_values,
                    "weights": {column: parse_float(weights.get(column), 0.0) for column in feature_columns},
                    "semantic_component": semantic_component,
                    "geometry_component": geometry_component,
                    "motion_component": motion_component,
                    "additive_score": additive_score,
                    "in_pool": in_pool,
                }
            )
        probabilities = iter(softmax(pool_scores))
        for item in scored:
            item["posterior"] = next(probabilities) if item["in_pool"] else 0.0
            item["masked_score"] = item["additive_score"] if item["in_pool"] else -1.0e9
        selected = max(scored, key=lambda item: item["masked_score"])

        gt_object_id = bundle.gt_row.get("gt_object_id", "")
        gt_object_name = bundle.gt_row.get("gt_object_name", "")
        exact = selected["object_id"] == gt_object_id
        selected_name = str(selected["template"].get("name", ""))
        class_match = exact or class_matches(selected_name, gt_object_name)
        payload = bundle.payload
        selector_rows.append(
            {
                "sample_index": index,
                "split": bundle.split,
                "sample_id": bundle.sample.sample_id,
                "scene_name": payload.get("scene_name", ""),
                "sequence_name": payload.get("sequence_name", ""),
                "start_frame": payload.get("start_frame", ""),
                "end_frame": payload.get("end_frame", ""),
                "want": want,
                "selected_object_id": selected["object_id"],
                "selected_object_name": selected_name,
                "selected_matches_gt": int(exact),
                "selected_class_matches_gt": int(class_match),
                "gt_object_id": gt_object_id,
                "gt_object": gt_object_name,
                "gt_object_name": gt_object_name,
                "gt_action": want,
                "semantic_gap": semantic_gap,
                "selected_semantic_total": selected["semantic"]["semantic_total"],
                "selected_semantic_component": selected["semantic_component"],
                "selected_geometry_component": selected["geometry_component"],
                "selected_motion_component": selected["motion_component"],
                "selected_additive_fusion_score": selected["additive_score"],
            }
        )
        gt_eval_rows.append(
            {
                "sample_index": index,
                "split": bundle.split,
                "sample_id": bundle.sample.sample_id,
                "scene_name": payload.get("scene_name", ""),
                "sequence_name": payload.get("sequence_name", ""),
                "start_frame": payload.get("start_frame", ""),
                "end_frame": payload.get("end_frame", ""),
                "want": want,
                "final_object": selected["object_id"],
                "final_object_name": selected_name,
                "selected_object_id": selected["object_id"],
                "selected_object_name": selected_name,
                "gt_object_id": gt_object_id,
                "gt_object_name": gt_object_name,
                "selected_matches_gt": int(exact),
                "selected_class_matches_gt": int(class_match),
                "winner_stage": "additive_geometry_fusion",
                "exact_instance_match": int(exact),
                "class_name_match": int(class_match),
            }
        )
        for item in scored:
            semantic = item["semantic"]
            template = item["template"]
            row: dict[str, Any] = {
                "sample_index": index,
                "split": bundle.split,
                "sample_id": bundle.sample.sample_id,
                "want": want,
                "object_id": item["object_id"],
                "name": template.get("name", semantic.get("name", "")),
                "category": template.get("category", semantic.get("category", "")),
                "is_gt": int(item["object_id"] == gt_object_id),
                "match": semantic["match"],
                "ease": semantic["ease"],
                "semantic_total": semantic["semantic_total"],
                "reasoning_trace": semantic.get("reasoning_trace", ""),
                "selected_object_id": selected["object_id"],
                "selected_object_name": selected_name,
                "gt_object_id": gt_object_id,
                "gt_object_name": gt_object_name,
                "selected_matches_gt": int(exact),
                "selected_class_matches_gt": int(class_match),
            }
            for column in feature_columns:
                row[column] = item["features"][column]
                row[f"weight_{column}"] = item["weights"][column]
            row.update(
                {
                    "semantic_component": item["semantic_component"],
                    "geometry_component": item["geometry_component"],
                    "motion_component": item["motion_component"],
                    "additive_fusion_score": item["additive_score"],
                    "masked_fusion_score": item["masked_score"],
                    "fusion_posterior": item["posterior"],
                    "semantic_gap": semantic_gap,
                    "in_candidate_pool": int(item["in_pool"]),
                    "is_selected": int(item["object_id"] == selected["object_id"]),
                    "winner_stage": "additive_geometry_fusion",
                }
            )
            candidate_rows.append(row)
    return candidate_rows, selector_rows, gt_eval_rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("all", "train", "test"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        total = len(subset)
        exact = sum(int(row["exact_instance_match"]) for row in subset)
        class_match = sum(int(row["class_name_match"]) for row in subset)
        output[split] = {
            "total": total,
            "exact_instance_match": exact,
            "exact_instance_accuracy": exact / max(total, 1),
            "class_name_match": class_match,
            "class_name_accuracy": class_match / max(total, 1),
        }
    return output


def flatten(groups: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for group in groups for row in group]


def main() -> None:
    args = parse_args()
    if args.sample_retries < 1:
        raise ValueError("--sample-retries must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = build_bundles(args)
    semantic_groups = generate_semantic_rows(bundles, args)
    fusion_summary = json.loads(args.fusion_summary_json.read_text(encoding="utf-8"))
    candidate_rows, selector_rows, gt_eval_rows = fuse(bundles, semantic_groups, fusion_summary)

    selection_by_sample = {
        (str(row["split"]), str(row["sample_id"])): row for row in selector_rows
    }
    semantic_rows = flatten(semantic_groups)
    for row in semantic_rows:
        selection = selection_by_sample[(str(row["split"]), str(row["sample_id"]))]
        row.update(
            {
                "selected_object_id": selection["selected_object_id"],
                "selected_object_name": selection["selected_object_name"],
                "gt_object_id": selection["gt_object_id"],
                "gt_object_name": selection["gt_object_name"],
                "selected_matches_gt": selection["selected_matches_gt"],
                "selected_class_matches_gt": selection["selected_class_matches_gt"],
                "is_selected": int(row["object_id"] == selection["selected_object_id"]),
                "is_gt": int(row["object_id"] == selection["gt_object_id"]),
            }
        )
    semantic_fields = [
        "sample_index",
        "split",
        "sample_id",
        "want",
        "object_id",
        "name",
        "category",
        "match",
        "ease",
        "semantic_total",
        "reasoning_trace",
        "selected_object_id",
        "selected_object_name",
        "gt_object_id",
        "gt_object_name",
        "selected_matches_gt",
        "selected_class_matches_gt",
        "is_selected",
        "is_gt",
    ]
    write_csv(args.output_dir / "semantic_scores.csv", semantic_rows, semantic_fields)
    feature_columns = [str(column) for column in fusion_summary["feature_columns"]]
    candidate_fields = [
        "sample_index",
        "split",
        "sample_id",
        "want",
        "object_id",
        "name",
        "category",
        "is_gt",
        "match",
        "ease",
        "semantic_total",
        "reasoning_trace",
        "selected_object_id",
        "selected_object_name",
        "gt_object_id",
        "gt_object_name",
        "selected_matches_gt",
        "selected_class_matches_gt",
        *[column for column in feature_columns if column != "semantic_total"],
        *[f"weight_{column}" for column in feature_columns],
        "semantic_component",
        "geometry_component",
        "motion_component",
        "additive_fusion_score",
        "masked_fusion_score",
        "fusion_posterior",
        "semantic_gap",
        "in_candidate_pool",
        "is_selected",
        "winner_stage",
    ]
    selector_fields = list(selector_rows[0]) if selector_rows else []
    gt_fields = list(gt_eval_rows[0]) if gt_eval_rows else []
    write_csv(args.output_dir / "teacher_candidate_scores.csv", candidate_rows, candidate_fields)
    write_csv(args.output_dir / "selector_sample_summary.csv", selector_rows, selector_fields)
    write_csv(args.output_dir / "gt_eval_all.csv", gt_eval_rows, gt_fields)
    write_csv(
        args.output_dir / "gt_eval_misses.csv",
        [row for row in gt_eval_rows if not int(row["exact_instance_match"])],
        gt_fields,
    )
    metric_payload = metrics(gt_eval_rows)
    (args.output_dir / "gt_eval_summary.json").write_text(
        json.dumps(metric_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_summary = {
        "backend": "offline" if args.semantic_source_csv is not None else args.backend,
        "model_name": args.model_name,
        "ollama_base_url": (
            args.ollama_base_url
            if args.semantic_source_csv is None and args.backend == "ollama"
            else None
        ),
        "openai_base_url": (
            args.openai_base_url
            if args.semantic_source_csv is None and args.backend == "openai"
            else None
        ),
        "openai_api_key_env": (
            args.openai_api_key_env
            if args.semantic_source_csv is None and args.backend == "openai"
            else None
        ),
        "openai_reasoning_effort": (
            args.openai_reasoning_effort
            if args.semantic_source_csv is None and args.backend == "openai"
            else None
        ),
        "api_response_storage": False if args.backend == "openai" else None,
        "selection_mode": "fixed_additive_geometry_motion_fusion",
        "uses_tie_break": False,
        "formula": "sum(weight[feature] * feature) over fixed historical features",
        "feature_columns": feature_columns,
        "semantic_gap": fusion_summary["selected_by_cv"]["semantic_gap"],
        "feature_template_csv": str(args.feature_template_csv),
        "feature_template_sha256": sha256(args.feature_template_csv),
        "fusion_summary_json": str(args.fusion_summary_json),
        "fusion_summary_sha256": sha256(args.fusion_summary_json),
        "semantic_source_csv": str(args.semantic_source_csv) if args.semantic_source_csv else None,
        "output_dir": str(args.output_dir),
        "num_samples": len(bundles),
        "num_candidate_rows": len(candidate_rows),
        "metrics": metric_payload,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
