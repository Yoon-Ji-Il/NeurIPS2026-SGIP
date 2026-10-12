from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from object_selector.common.selector_interfaces import (
    ObjectCandidate,
    SceneGraph,
    SemanticScore,
    SemanticScoringResult,
    TeacherInputSample,
)
from object_selector.common.want_taxonomy import map_object_to_want, normalize_object_name


PROMPT_INPUT_MARKER = "INPUT_JSON:"
MAX_REASONING_WORDS = 24


def build_semantic_prompt(sample: TeacherInputSample) -> str:
    payload = {
        "sample_id": sample.sample_id,
        "want": sample.want,
        "supporting_actions": list(sample.supporting_actions),
        "scene_graph_summary": scene_graph_summary(sample.scene_graph),
        "candidate_objects": [candidate_prompt_payload(candidate) for candidate in sample.object_candidates],
        "canonical_policy": {
            "object_to_want": (
                "Each candidate includes canonical_object_want from the project taxonomy. "
                "Use it as a strong semantic prior for match scoring."
            ),
            "priority": (
                "Candidates whose canonical_object_want equals the requested want should usually "
                "receive a higher match score than candidates mapped to a different want."
            ),
            "fallback_use": "The use want is fallback-only; specific wants have priority over use.",
        },
        "rubric": {
            "match": "Integer 1 to 5. How well the object satisfies the high-level want.",
            "ease": "Integer 1 to 5. How easy and natural the object is for that want in the scene context.",
            "semantic_total": "Do not output this field. It will be computed as match + ease after parsing.",
            "reasoning_trace": "One short sentence. No chain-of-thought.",
        },
        "output_schema": {
            "sample_id": sample.sample_id,
            "want": sample.want,
            "semantic_scores": [
                {
                    "object_id": "candidate object id",
                    "match": "integer 1..5",
                    "ease": "integer 1..5",
                    "reasoning_trace": "one short sentence",
                }
            ],
        },
    }
    return (
        "You are a semantic scoring teacher for human-object target selection.\n"
        "Score every candidate object with respect to the high-level want.\n"
        "Score primarily based on the want.\n"
        "Use supporting_actions only as semantic hints for interpreting the want.\n"
        "Use canonical_object_want as the main object taxonomy prior for semantic match.\n"
        "Do not overrule a specific canonical object mapping just because another object can be loosely associated with an action.\n"
        "Use scene_graph context only to understand object identity and scene context.\n"
        "Do not perform geometric tie-breaking.\n"
        "Do not use face direction, root direction, torso direction, distances, or positions to resolve final ties.\n"
        "Keep reasoning_trace concise: one short sentence per object, no hidden chain-of-thought.\n"
        "Return strict JSON only. Do not wrap the JSON in markdown.\n"
        f"{PROMPT_INPUT_MARKER}\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_semantic_response(text: str) -> SemanticScoringResult:
    payload = extract_json_object(text)
    sample_id = require_text(payload, "sample_id")
    want = require_text(payload, "want")
    raw_scores = payload.get("semantic_scores")
    if raw_scores is None:
        raise ValueError("semantic_scores is required")

    semantic_scores: dict[str, SemanticScore] = {}
    if isinstance(raw_scores, list):
        score_items = raw_scores
    elif isinstance(raw_scores, Mapping):
        score_items = []
        for object_id, value in raw_scores.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"semantic_scores[{object_id!r}] must be an object")
            score_items.append({"object_id": object_id, **dict(value)})
    else:
        raise ValueError("semantic_scores must be a list or object")

    for index, raw_score in enumerate(score_items):
        if not isinstance(raw_score, Mapping):
            raise ValueError(f"semantic_scores[{index}] must be an object")
        object_id = require_text(raw_score, "object_id")
        if object_id in semantic_scores:
            raise ValueError(f"Duplicate semantic score for object_id {object_id!r}")
        match = validate_integer_score(raw_score.get("match"), f"{object_id}.match")
        ease = validate_integer_score(raw_score.get("ease"), f"{object_id}.ease")
        reasoning_trace = parse_reasoning_trace(raw_score.get("reasoning_trace"), object_id)
        semantic_scores[object_id] = SemanticScore(
            object_id=object_id,
            match=match,
            ease=ease,
            semantic_total=float(match + ease),
            reasoning_trace=(reasoning_trace,),
        )

    return SemanticScoringResult(
        sample_id=sample_id,
        want=want,
        semantic_scores=semantic_scores,
        metadata={"semantic_total_source": "computed_match_plus_ease"},
    )


def score_candidates_with_llm(sample: TeacherInputSample, llm_client: Any) -> SemanticScoringResult:
    prompt = build_semantic_prompt(sample)
    response_text = call_llm_client(
        llm_client,
        prompt,
        candidate_ids=[candidate.object_id for candidate in sample.object_candidates],
    )
    result = parse_semantic_response(response_text)
    validate_semantic_result_for_sample(result, sample)
    return result


def get_semantic_tie_set(semantic_scores: Mapping[str, SemanticScore] | SemanticScoringResult, delta: float = 0) -> list[str]:
    scores = semantic_scores.semantic_scores if isinstance(semantic_scores, SemanticScoringResult) else semantic_scores
    if not scores:
        return []
    totals = {object_id: score.semantic_total for object_id, score in scores.items()}
    top_total = max(totals.values())
    return [
        object_id
        for object_id, total in totals.items()
        if top_total - total <= delta
    ]


def validate_semantic_result_for_sample(result: SemanticScoringResult, sample: TeacherInputSample) -> None:
    if result.sample_id != sample.sample_id:
        raise ValueError(f"Response sample_id {result.sample_id!r} does not match {sample.sample_id!r}")
    if result.want != sample.want:
        raise ValueError(f"Response want {result.want!r} does not match {sample.want!r}")

    expected_ids = [candidate.object_id for candidate in sample.object_candidates]
    expected_set = set(expected_ids)
    actual_set = set(result.semantic_scores)
    missing = [object_id for object_id in expected_ids if object_id not in actual_set]
    extra = sorted(actual_set - expected_set)
    if missing:
        raise ValueError(f"Semantic response missing candidate object scores: {missing}")
    if extra:
        raise ValueError(f"Semantic response contains scores for non-candidate objects: {extra}")

    for object_id, score in result.semantic_scores.items():
        expected_total = float(score.match + score.ease)
        if score.semantic_total != expected_total:
            raise ValueError(
                f"semantic_total for {object_id!r} must be computed as match + ease"
            )
        for trace in score.reasoning_trace:
            validate_reasoning_trace(trace, object_id)


def semantic_response_json_schema(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sample_id", "want", "semantic_scores"],
        "properties": {
            "sample_id": {"type": "string"},
            "want": {"type": "string"},
            "semantic_scores": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object_id", "match", "ease", "reasoning_trace"],
                    "properties": {
                        "object_id": {"type": "string", "enum": candidate_ids},
                        "match": {"type": "integer", "minimum": 1, "maximum": 5},
                        "ease": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reasoning_trace": {"type": "string"},
                    },
                },
            },
        },
    }


@dataclass(frozen=True)
class MockSemanticLLMClient:
    fixed_response: str | Mapping[str, Any] | None = None

    def complete(self, prompt: str) -> str:
        if self.fixed_response is not None:
            if isinstance(self.fixed_response, str):
                return self.fixed_response
            return json.dumps(self.fixed_response, ensure_ascii=False)
        payload = extract_prompt_input_payload(prompt)
        want = str(payload["want"])
        scores = []
        for candidate in payload.get("candidate_objects", []):
            object_id = str(candidate["object_id"])
            name = str(candidate.get("name") or candidate.get("category") or "")
            mapped_want = map_object_to_want(name)
            if mapped_want == want:
                match = 5
                ease = 4
                reason = f"{name} directly supports the want."
            elif mapped_want == "use" and want == "use":
                match = 4
                ease = 3
                reason = f"{name} is a plausible use target."
            else:
                match = 1
                ease = 2
                reason = f"{name} is not the main semantic fit."
            scores.append(
                {
                    "object_id": object_id,
                    "match": match,
                    "ease": ease,
                    "reasoning_trace": reason,
                }
            )
        return json.dumps(
            {
                "sample_id": payload["sample_id"],
                "want": want,
                "semantic_scores": scores,
            },
            ensure_ascii=False,
        )


def scene_graph_summary(scene_graph: SceneGraph) -> dict[str, Any]:
    object_nodes = [
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "name": node.name,
            "category": node.metadata.get("category"),
        }
        for node in scene_graph.nodes
        if node.node_type == "object"
    ]
    edges = [
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation": edge.relation,
        }
        for edge in scene_graph.edges
    ]
    return {
        "metadata": {
            key: value
            for key, value in scene_graph.metadata.items()
            if key in {"scene_name", "sequence_name", "coordinate_frame"}
        },
        "object_nodes": object_nodes,
        "edges": edges,
        "geometry_note": "Numeric positions and orientation metrics are intentionally excluded from semantic scoring.",
    }


def candidate_prompt_payload(candidate: ObjectCandidate) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in candidate.metadata.items()
        if key in {"semantic_label_id"}
    }
    return {
        "object_id": candidate.object_id,
        "name": candidate.name,
        "category": candidate.category,
        "normalized_name": normalize_object_name(candidate.name),
        "canonical_object_want": map_object_to_want(candidate.name),
        "metadata": metadata,
    }


def call_llm_client(llm_client: Any, prompt: str, *, candidate_ids: list[str] | None = None) -> str:
    if hasattr(llm_client, "complete"):
        response = llm_client.complete(prompt)
    elif hasattr(llm_client, "generate"):
        response = llm_client.generate(prompt)
    elif hasattr(llm_client, "chat"):
        response = llm_client.chat(prompt)
    elif hasattr(llm_client, "complete_json"):
        response = llm_client.complete_json(
            instructions="Return semantic scoring JSON only.",
            prompt=prompt,
            json_schema=semantic_response_json_schema(candidate_ids or []),
            schema_name="semantic_scoring_result",
        )
    elif callable(llm_client):
        response = llm_client(prompt)
    else:
        raise TypeError("llm_client must expose complete, generate, chat, complete_json, or be callable")

    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, Mapping):
        return json.dumps(response, ensure_ascii=False)
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    return str(response)


def extract_prompt_input_payload(prompt: str) -> dict[str, Any]:
    if PROMPT_INPUT_MARKER not in prompt:
        raise ValueError(f"Prompt is missing {PROMPT_INPUT_MARKER}")
    text = prompt.split(PROMPT_INPUT_MARKER, 1)[1].strip()
    payload = extract_json_object(text)
    return dict(payload)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(str(text).strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Semantic response must be a JSON object")
    return payload


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} is required")
    return text


def parse_reasoning_trace(value: Any, object_id: str) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"reasoning_trace for {object_id!r} must be one short sentence")
        value = value[0]
    trace = str(value).strip() if value is not None else ""
    validate_reasoning_trace(trace, object_id)
    return trace


def validate_reasoning_trace(trace: str, object_id: str) -> None:
    if not trace:
        raise ValueError(f"reasoning_trace for {object_id!r} must be non-empty")
    if len(trace.split()) > MAX_REASONING_WORDS:
        raise ValueError(
            f"reasoning_trace for {object_id!r} is too long; max {MAX_REASONING_WORDS} words"
        )


def validate_integer_score(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer in [1, 5]")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str) and re.fullmatch(r"[1-5]", value.strip()):
        score = int(value.strip())
    else:
        raise ValueError(f"{field_name} must be an integer in [1, 5]")
    if not 1 <= score <= 5:
        raise ValueError(f"{field_name} must be an integer in [1, 5]")
    return score
