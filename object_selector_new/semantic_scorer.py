"""Semantic scoring (LLM-based) of candidate objects vs the requested want."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from object_selector_new.common.selector_interfaces import (
    ObjectCandidate,
    SceneGraph,
    SemanticScore,
    SemanticScoringResult,
    TeacherInputSample,
)
from object_selector_new.common.want_taxonomy import (
    map_object_to_want,
    normalize_object_name,
)


PROMPT_INPUT_MARKER = "INPUT_JSON:"
MAX_REASONING_WORDS = 24


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def scene_graph_summary(scene_graph: SceneGraph) -> dict[str, Any]:
    object_nodes = [
        {
            "node_id": n.node_id,
            "node_type": n.node_type,
            "name": n.name,
            "category": n.metadata.get("category"),
        }
        for n in scene_graph.nodes
        if n.node_type == "object"
    ]
    return {
        "metadata": {
            k: v for k, v in scene_graph.metadata.items()
            if k in {"scene_name", "sequence_name", "coordinate_frame"}
        },
        "object_nodes": object_nodes,
        "edges": [
            {"source_id": e.source_id, "target_id": e.target_id, "relation": e.relation}
            for e in scene_graph.edges
        ],
        "geometry_note": "Numeric positions are excluded from semantic scoring.",
    }


def candidate_prompt_payload(candidate: ObjectCandidate) -> dict[str, Any]:
    return {
        "object_id": candidate.object_id,
        "name": candidate.name,
        "category": candidate.category,
        "normalized_name": normalize_object_name(candidate.name),
        "canonical_object_want": map_object_to_want(candidate.name),
        "metadata": {
            k: v for k, v in candidate.metadata.items() if k in {"semantic_label_id"}
        },
    }


def build_semantic_prompt(sample: TeacherInputSample) -> str:
    payload = {
        "sample_id": sample.sample_id,
        "want": sample.want,
        "supporting_actions": list(sample.supporting_actions),
        "scene_graph_summary": scene_graph_summary(sample.scene_graph),
        "candidate_objects": [candidate_prompt_payload(c) for c in sample.object_candidates],
        "canonical_policy": {
            "object_to_want": "Use canonical_object_want as a strong semantic prior.",
            "priority": "Same canonical want as requested → higher score.",
            "fallback_use": "use is fallback-only; specific wants take priority.",
        },
        "rubric": {
            "match": "Integer 1..5. How well the object satisfies the want.",
            "ease":  "Integer 1..5. How easy the interaction is in this scene.",
            "semantic_total": "Computed by code as match + ease.",
            "reasoning_trace": "One short sentence.",
        },
        "output_schema": {
            "sample_id": sample.sample_id,
            "want": sample.want,
            "semantic_scores": [
                {"object_id": "id", "match": "1..5", "ease": "1..5", "reasoning_trace": "..."}
            ],
        },
    }
    return (
        "You are a semantic scoring teacher for human-object target selection.\n"
        "Score every candidate object with respect to the want.\n"
        "Use canonical_object_want as the main object taxonomy prior.\n"
        "Do not perform geometric tie-breaking. Do not use distances or directions.\n"
        "Keep reasoning_trace concise. Return strict JSON only.\n"
        f"{PROMPT_INPUT_MARKER}\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------
def strip_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(str(text).strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s < 0 or e < s:
            raise
        payload = json.loads(cleaned[s : e + 1])
    if not isinstance(payload, dict):
        raise ValueError("Semantic response must be a JSON object")
    return payload


def require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} is required")
    return text


def validate_integer_score(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int in [1,5]")
    if isinstance(value, int):
        v = value
    elif isinstance(value, str) and re.fullmatch(r"[1-5]", value.strip()):
        v = int(value.strip())
    else:
        raise ValueError(f"{name} must be int in [1,5]")
    if not 1 <= v <= 5:
        raise ValueError(f"{name} must be int in [1,5]")
    return v


def validate_reasoning_trace(trace: str, object_id: str) -> None:
    if not trace:
        raise ValueError(f"reasoning_trace for {object_id!r} must be non-empty")
    if len(trace.split()) > MAX_REASONING_WORDS:
        raise ValueError(
            f"reasoning_trace for {object_id!r} exceeds {MAX_REASONING_WORDS} words"
        )


def parse_reasoning_trace(value: Any, object_id: str) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"reasoning_trace for {object_id!r} must be one sentence")
        value = value[0]
    trace = str(value).strip() if value is not None else ""
    validate_reasoning_trace(trace, object_id)
    return trace


def parse_semantic_response(text: str) -> SemanticScoringResult:
    payload = extract_json_object(text)
    sample_id = require_text(payload, "sample_id")
    want = require_text(payload, "want")
    raw = payload.get("semantic_scores")
    if raw is None:
        raise ValueError("semantic_scores is required")

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, Mapping):
        items = [{"object_id": oid, **dict(v)} for oid, v in raw.items()]
    else:
        raise ValueError("semantic_scores must be list or object")

    out: dict[str, SemanticScore] = {}
    for i, r in enumerate(items):
        if not isinstance(r, Mapping):
            raise ValueError(f"semantic_scores[{i}] must be object")
        oid = require_text(r, "object_id")
        if oid in out:
            raise ValueError(f"Duplicate semantic score for {oid!r}")
        m = validate_integer_score(r.get("match"), f"{oid}.match")
        e = validate_integer_score(r.get("ease"), f"{oid}.ease")
        trace = parse_reasoning_trace(r.get("reasoning_trace"), oid)
        out[oid] = SemanticScore(
            object_id=oid,
            match=m,
            ease=e,
            semantic_total=float(m + e),
            reasoning_trace=(trace,),
        )

    return SemanticScoringResult(
        sample_id=sample_id,
        want=want,
        semantic_scores=out,
        metadata={"semantic_total_source": "computed_match_plus_ease"},
    )


def validate_semantic_result_for_sample(result: SemanticScoringResult, sample: TeacherInputSample) -> None:
    if result.sample_id != sample.sample_id:
        raise ValueError(f"sample_id mismatch: {result.sample_id} vs {sample.sample_id}")
    if result.want != sample.want:
        raise ValueError(f"want mismatch: {result.want} vs {sample.want}")
    expected = [c.object_id for c in sample.object_candidates]
    actual = set(result.semantic_scores)
    missing = [oid for oid in expected if oid not in actual]
    extra = sorted(actual - set(expected))
    if missing:
        raise ValueError(f"missing semantic scores: {missing}")
    if extra:
        raise ValueError(f"extra semantic scores: {extra}")


# ---------------------------------------------------------------------------
# LLM dispatch
# ---------------------------------------------------------------------------
def call_llm_client(llm_client: Any, prompt: str) -> str:
    if hasattr(llm_client, "complete"):
        response = llm_client.complete(prompt)
    elif hasattr(llm_client, "generate"):
        response = llm_client.generate(prompt)
    elif callable(llm_client):
        response = llm_client(prompt)
    else:
        raise TypeError("llm_client must expose complete/generate or be callable")
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, Mapping):
        return json.dumps(response, ensure_ascii=False)
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    return str(response)


def score_candidates_with_llm(sample: TeacherInputSample, llm_client: Any) -> SemanticScoringResult:
    prompt = build_semantic_prompt(sample)
    text = call_llm_client(llm_client, prompt)
    result = parse_semantic_response(text)
    validate_semantic_result_for_sample(result, sample)
    return result


# ---------------------------------------------------------------------------
# Mock client (for unit tests)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MockSemanticLLMClient:
    fixed_response: str | Mapping[str, Any] | None = None

    def complete(self, prompt: str) -> str:
        if self.fixed_response is not None:
            return (
                self.fixed_response if isinstance(self.fixed_response, str)
                else json.dumps(self.fixed_response, ensure_ascii=False)
            )
        marker = PROMPT_INPUT_MARKER
        if marker not in prompt:
            raise ValueError(f"prompt missing {marker}")
        payload = extract_json_object(prompt.split(marker, 1)[1])
        want = str(payload["want"])
        scores = []
        for c in payload.get("candidate_objects", []):
            name = str(c.get("name") or c.get("category") or "")
            mapped = map_object_to_want(name)
            if mapped == want:
                m, e, reason = 5, 4, f"{name} directly supports {want}."
            elif mapped == "use" and want == "use":
                m, e, reason = 4, 3, f"{name} is a plausible use target."
            else:
                m, e, reason = 1, 2, f"{name} is not a primary fit."
            scores.append({"object_id": c["object_id"], "match": m, "ease": e, "reasoning_trace": reason})
        return json.dumps({"sample_id": payload["sample_id"], "want": want, "semantic_scores": scores})


__all__ = [
    "MockSemanticLLMClient",
    "PROMPT_INPUT_MARKER",
    "build_semantic_prompt",
    "candidate_prompt_payload",
    "extract_json_object",
    "parse_semantic_response",
    "score_candidates_with_llm",
    "validate_semantic_result_for_sample",
]
