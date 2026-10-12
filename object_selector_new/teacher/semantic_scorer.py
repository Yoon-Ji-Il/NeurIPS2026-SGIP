"""Semantic scoring (LLM-based) of candidate objects vs the requested want.

The teacher LLM produces a four-criterion rubric per candidate:
  - want_affordance       (1-40)
  - want_state_viability  (1-28)
  - action_priority       (1-15)
  - scene_grounded_support(1-17)
  rubric = sum of the four = 4..100.

Geometry signals (root_horizontal_alignment, face_normal_alignment, etc.) are
exposed inside the prompt so the LLM can reason about the geometric alignment
when scoring `scene_grounded_support`. The set of exposed signals is governed
by a PromptGeometrySpec passed in by the caller.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from object_selector_new.common.selector_interfaces import (
    ObjectCandidate,
    PromptGeometrySpec,
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
MAX_REASONING_WORDS_PER_KEY = 30
MAX_CANDIDATES_PER_LLM_CALL = 8

REASONING_KEYS = ("affordance", "state_viability", "action_priority", "scene_support")

# Score ranges per criterion.
SCORE_RANGES: dict[str, tuple[int, int]] = {
    "want_affordance": (1, 40),
    "want_state_viability": (1, 28),
    "action_priority": (1, 15),
    "scene_grounded_support": (1, 17),
}
RUBRIC_MIN = sum(low for low, _ in SCORE_RANGES.values())  # 4
RUBRIC_MAX = sum(high for _, high in SCORE_RANGES.values())  # 100


# Default geometry spec used when caller does not supply one. Matches the
# "currently-best student feature combination": root_horizontal_alignment +
# face_normal_alignment as primary signals, with classical 3D alignment and
# distance as auxiliary context.
DEFAULT_PROMPT_GEOMETRY = PromptGeometrySpec(
    primary_columns=("root_horizontal_alignment", "face_normal_alignment"),
    auxiliary_columns=("root_alignment", "torso_alignment", "distance"),
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _candidate_geometry_payload(
    candidate: ObjectCandidate,
    geometry_spec: PromptGeometrySpec,
) -> dict[str, dict[str, float]]:
    """Pull the configured geometry columns out of candidate.metadata['metrics']."""  # noqa: E501
    metrics = dict(candidate.metadata.get("metrics", {}))
    primary = {col: float(metrics.get(col, 0.0)) for col in geometry_spec.primary_columns}
    auxiliary = {col: float(metrics.get(col, 0.0)) for col in geometry_spec.auxiliary_columns}
    return {"primary": primary, "auxiliary": auxiliary}


def scene_graph_summary(scene_graph: SceneGraph) -> dict[str, Any]:
    object_nodes = [
        {
            "node_id": n.node_id,
            "node_type": n.node_type,
            "name": n.name,
            "category": n.metadata.get("category"),
            "state": n.metadata.get("state"),
            "occupancy": n.metadata.get("occupancy"),
        }
        for n in scene_graph.nodes
        if n.node_type == "object"
    ]
    return {
        "metadata": {
            k: v
            for k, v in scene_graph.metadata.items()
            if k in {"scene_name", "sequence_name", "coordinate_frame", "context"}
        },
        "object_nodes": object_nodes,
        "edges": [
            {"source_id": e.source_id, "target_id": e.target_id, "relation": e.relation}
            for e in scene_graph.edges
        ],
    }


def candidate_prompt_payload(
    candidate: ObjectCandidate,
    geometry_spec: PromptGeometrySpec,
) -> dict[str, Any]:
    return {
        "object_id": candidate.object_id,
        "name": candidate.name,
        "category": candidate.category,
        "normalized_name": normalize_object_name(candidate.name),
        "canonical_object_want": map_object_to_want(candidate.name),
        "geometry_signals": _candidate_geometry_payload(candidate, geometry_spec),
        "metadata": {
            k: v for k, v in candidate.metadata.items() if k in {"semantic_label_id", "state", "occupancy"}
        },
    }


def candidate_overview_payload(candidate: ObjectCandidate) -> dict[str, Any]:
    return {
        "object_id": candidate.object_id,
        "name": candidate.name,
        "category": candidate.category,
        "normalized_name": normalize_object_name(candidate.name),
        "canonical_object_want": map_object_to_want(candidate.name),
    }


SYSTEM_PREAMBLE = (
    "You are a semantic scoring teacher for human-object target selection.\n"
    "Given a desired action (`want`), a scene graph, and a list of candidate\n"
    "objects, score every candidate on a four-criterion rubric. The rubric is\n"
    "designed to weight semantic want-aware reasoning much more than raw\n"
    "geometric alignment (want : scene = 5 : 1).\n"
    "Return exactly one JSON object and nothing else.\n"
    "Do not output reasoning, markdown, code fences, or any text before/after the JSON.\n"
    "The JSON must follow output_schema exactly.\n"
)

RUBRIC_INSTRUCTIONS = (
    "Rubric (per candidate, all integers):\n"
    "  1. want_affordance        (1-40):\n"
    "       Is the object's primary purpose this `want`?\n"
    "       35-40: prototypical (sit→chair, drink→cup, sleep→bed)\n"
    "       25-34: frequently used but secondary\n"
    "       15-24: incidental surface or related tool\n"
    "       5-14:  accidentally enables the action\n"
    "       1-4:   not a target of this action\n"
    "  2. want_state_viability   (1-28):\n"
    "       Given the object's current state and scene context, can the action\n"
    "       actually be completed as intended?\n"
    "       23-28: immediately usable, no obstruction\n"
    "       15-22: usable after one trivial preparatory action\n"
    "       8-14:  state degrades the action's intent (e.g. cluttered seat)\n"
    "       1-7:   state blocks the action\n"
    "  3. action_priority        (1-15):\n"
    "       When several candidates afford the same want, which would a person\n"
    "       most naturally pick FIRST in this scene context?\n"
    "       13-15: highest-priority pick given context\n"
    "       9-12:  natural but not the top pick\n"
    "       5-8:   possible but a different candidate is clearly preferred\n"
    "       1-4:   only chosen if no better candidate exists\n"
    "  4. scene_grounded_support (1-17):\n"
    "       Do the geometric alignment signals from the scene graph point at\n"
    "       this candidate? Look at the geometry_signals.primary fields most,\n"
    "       and use auxiliary fields as context only.\n"
    "       14-17: primary signals strongly positive (≥0.7)\n"
    "       9-13:  weak but positive\n"
    "       5-8:   weak alignment or distance is awkward\n"
    "       1-4:   negative alignment (human facing away)\n"
    "\n"
    "rubric = want_affordance + want_state_viability + action_priority + scene_grounded_support.\n"
    "rubric must equal that sum exactly.\n"
)


def _geometry_instructions(geometry_spec: PromptGeometrySpec) -> str:
    primary = ", ".join(geometry_spec.primary_columns) or "(none)"
    auxiliary = ", ".join(geometry_spec.auxiliary_columns) or "(none)"
    return (
        "Geometry signals exposed per candidate:\n"
        f"  primary   (drive scene_grounded_support):   {primary}\n"
        f"  auxiliary (context only, do not anchor on): {auxiliary}\n"
        "Each value is a precomputed scalar already normalized into a\n"
        "comparable range across candidates. Use these to compare candidates\n"
        "geometrically; do not invent additional geometry from positions.\n"
    )


def build_semantic_prompt(
    sample: TeacherInputSample,
    *,
    geometry_spec: PromptGeometrySpec | None = None,
    candidate_pool: tuple[ObjectCandidate, ...] | None = None,
) -> str:
    geometry_spec = geometry_spec or DEFAULT_PROMPT_GEOMETRY
    pool = candidate_pool or sample.object_candidates
    payload = {
        "sample_id": sample.sample_id,
        "want": sample.want,
        "supporting_actions": list(sample.supporting_actions),
        "scene_graph_summary": scene_graph_summary(sample.scene_graph),
        "candidate_objects": [
            candidate_prompt_payload(c, geometry_spec) for c in sample.object_candidates
        ],
        "rubric_policy": {
            "score_ranges": {k: list(v) for k, v in SCORE_RANGES.items()},
            "rubric_total_range": [RUBRIC_MIN, RUBRIC_MAX],
            "want_to_scene_ratio": "5:1 (want components 83/100, scene component 17/100)",
        },
        "required_scoring_object_ids": [c.object_id for c in sample.object_candidates],
        "scoring_scope_rule": (
            "Return exactly one semantic_scores entry for every ID in "
            "required_scoring_object_ids, in the same order. Do not score IDs "
            "that appear only in candidate_pool_overview. Copy each object_id exactly."
        ),
        "output_schema": {
            "sample_id": sample.sample_id,
            "want": sample.want,
            "semantic_scores": [
                {
                    "object_id": candidate.object_id,
                    "want_affordance": "1..40",
                    "want_state_viability": "1..28",
                    "action_priority": "1..15",
                    "scene_grounded_support": "1..17",
                    "rubric": "4..100",
                }
                for candidate in sample.object_candidates
            ],
        },
    }
    if len(pool) != len(sample.object_candidates):
        payload["candidate_pool_overview"] = [
            candidate_overview_payload(c) for c in pool
        ]
        payload["candidate_scoring_scope"] = (
            "Score ONLY candidate_objects. candidate_pool_overview is global context only."
        )
    return (
        SYSTEM_PREAMBLE
        + "\n"
        + RUBRIC_INSTRUCTIONS
        + "\n"
        + _geometry_instructions(geometry_spec)
        + "\n"
        + f"{PROMPT_INPUT_MARKER}\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------
def strip_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def _repair_json_text(text: str) -> str:
    repaired = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return re.sub(r",(\s*[}\]])", r"\1", repaired)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(str(text).strip())
    candidates = [cleaned]
    s = cleaned.find("{")
    e = cleaned.rfind("}")
    if s >= 0 and e >= s:
        candidates.append(cleaned[s : e + 1])

    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        for variant in (candidate, _repair_json_text(candidate)):
            if variant in seen:
                continue
            seen.add(variant)
            try:
                payload = json.loads(variant)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            continue
        break
    else:
        if last_error is not None:
            raise last_error
        raise ValueError("Semantic response must contain a JSON object")
    if not isinstance(payload, dict):
        raise ValueError("Semantic response must be a JSON object")
    return payload


def require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} is required")
    return text


def _validate_int_in_range(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int in [{low},{high}]")
    if isinstance(value, int):
        v = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        v = int(value.strip())
    else:
        raise ValueError(f"{name} must be int in [{low},{high}]")
    if not low <= v <= high:
        raise ValueError(f"{name} must be int in [{low},{high}], got {v}")
    return v


def _validate_reasoning_trace(value: Any, object_id: str) -> dict[str, str]:
    if value is None:
        return {key: "omitted" for key in REASONING_KEYS}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"reasoning_trace for {object_id!r} must be an object with keys " f"{REASONING_KEYS}"
        )
    out: dict[str, str] = {}
    for key in REASONING_KEYS:
        text = str(value.get(key) or "").strip()
        if not text:
            out[key] = "omitted"
            continue
        if len(text.split()) > MAX_REASONING_WORDS_PER_KEY:
            raise ValueError(
                f"reasoning_trace[{key}] for {object_id!r} exceeds "
                f"{MAX_REASONING_WORDS_PER_KEY} words"
            )
        out[key] = text
    return out


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

        components: dict[str, int] = {}
        for crit, (low, high) in SCORE_RANGES.items():
            components[crit] = _validate_int_in_range(r.get(crit), f"{oid}.{crit}", low, high)

        computed_rubric = sum(components.values())
        # Self-correct: if LLM reported a rubric, prefer the recomputed sum so
        # downstream code never has to deal with arithmetic mistakes.
        reported = r.get("rubric")
        if reported is not None:
            try:
                _validate_int_in_range(reported, f"{oid}.rubric", RUBRIC_MIN, RUBRIC_MAX)
            except ValueError:
                # ignore reported value, use computed
                pass

        trace = _validate_reasoning_trace(r.get("reasoning_trace"), oid)

        out[oid] = SemanticScore(
            object_id=oid,
            want_affordance=components["want_affordance"],
            want_state_viability=components["want_state_viability"],
            action_priority=components["action_priority"],
            scene_grounded_support=components["scene_grounded_support"],
            rubric=computed_rubric,
            reasoning_trace=trace,
        )

    return SemanticScoringResult(
        sample_id=sample_id,
        want=want,
        semantic_scores=out,
        metadata={"rubric_source": "computed_from_components"},
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


def _sample_with_candidates(
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


def _score_sample_once(
    sample: TeacherInputSample,
    llm_client: Any,
    *,
    geometry_spec: PromptGeometrySpec | None = None,
    candidate_pool: tuple[ObjectCandidate, ...] | None = None,
) -> SemanticScoringResult:
    prompt = build_semantic_prompt(
        sample,
        geometry_spec=geometry_spec,
        candidate_pool=candidate_pool,
    )
    text = call_llm_client(llm_client, prompt)
    result = parse_semantic_response(text)
    # During the per-candidate fallback some small models occasionally return
    # the right single score under a copied/hallucinated object ID.  There is
    # no ambiguity when both the request and response contain exactly one
    # candidate, so bind that score to the requested bookkeeping ID.  Never
    # coerce multi-score responses because their correspondence is ambiguous.
    if len(sample.object_candidates) == 1 and len(result.semantic_scores) == 1:
        expected_id = sample.object_candidates[0].object_id
        if expected_id not in result.semantic_scores:
            original_id, original_score = next(iter(result.semantic_scores.items()))
            rebound_score = SemanticScore(
                object_id=expected_id,
                want_affordance=original_score.want_affordance,
                want_state_viability=original_score.want_state_viability,
                action_priority=original_score.action_priority,
                scene_grounded_support=original_score.scene_grounded_support,
                rubric=original_score.rubric,
                reasoning_trace=original_score.reasoning_trace,
            )
            result = SemanticScoringResult(
                sample_id=result.sample_id,
                want=result.want,
                semantic_scores={expected_id: rebound_score},
                metadata={
                    **dict(result.metadata),
                    "object_id_rebound_from": original_id,
                },
            )
    validate_semantic_result_for_sample(result, sample)
    return result


def _score_candidates_individually(
    sample: TeacherInputSample,
    llm_client: Any,
    *,
    geometry_spec: PromptGeometrySpec | None = None,
    candidate_pool: tuple[ObjectCandidate, ...] | None = None,
) -> SemanticScoringResult:
    merged_scores: dict[str, SemanticScore] = {}
    for candidate in sample.object_candidates:
        single_sample = _sample_with_candidates(sample, (candidate,))
        single_result = _score_sample_once(
            single_sample,
            llm_client,
            geometry_spec=geometry_spec,
            candidate_pool=candidate_pool,
        )
        merged_scores.update(single_result.semantic_scores)
    merged_result = SemanticScoringResult(
        sample_id=sample.sample_id,
        want=sample.want,
        semantic_scores=merged_scores,
        metadata={"candidate_fallback": "single_candidate_calls"},
    )
    validate_semantic_result_for_sample(merged_result, sample)
    return merged_result


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


def score_candidates_with_llm(
    sample: TeacherInputSample,
    llm_client: Any,
    *,
    geometry_spec: PromptGeometrySpec | None = None,
) -> SemanticScoringResult:
    candidates = tuple(sample.object_candidates)
    if len(candidates) <= MAX_CANDIDATES_PER_LLM_CALL:
        try:
            return _score_sample_once(sample, llm_client, geometry_spec=geometry_spec)
        except Exception:
            if len(candidates) == 1:
                raise
            return _score_candidates_individually(
                sample,
                llm_client,
                geometry_spec=geometry_spec,
                candidate_pool=candidates,
            )

    merged_scores: dict[str, SemanticScore] = {}
    num_chunks = 0
    for start in range(0, len(candidates), MAX_CANDIDATES_PER_LLM_CALL):
        chunk = candidates[start : start + MAX_CANDIDATES_PER_LLM_CALL]
        chunk_sample = _sample_with_candidates(sample, chunk)
        try:
            chunk_result = _score_sample_once(
                chunk_sample,
                llm_client,
                geometry_spec=geometry_spec,
                candidate_pool=candidates,
            )
        except Exception:
            chunk_result = _score_candidates_individually(
                chunk_sample,
                llm_client,
                geometry_spec=geometry_spec,
                candidate_pool=candidates,
            )
        merged_scores.update(chunk_result.semantic_scores)
        num_chunks += 1

    merged_result = SemanticScoringResult(
        sample_id=sample.sample_id,
        want=sample.want,
        semantic_scores=merged_scores,
        metadata={
            "chunked": True,
            "num_chunks": num_chunks,
            "chunk_size": MAX_CANDIDATES_PER_LLM_CALL,
        },
    )
    validate_semantic_result_for_sample(merged_result, sample)
    return merged_result


# ---------------------------------------------------------------------------
# Mock client (for unit tests)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MockSemanticLLMClient:
    """Deterministic mock that uses canonical_object_want to emulate scoring.

    Produces 4-criterion rubric scores with reasonable defaults so end-to-end
    pipelines can be exercised without an actual LLM.
    """

    fixed_response: str | Mapping[str, Any] | None = None

    def complete(self, prompt: str) -> str:
        if self.fixed_response is not None:
            return (
                self.fixed_response
                if isinstance(self.fixed_response, str)
                else json.dumps(self.fixed_response, ensure_ascii=False)
            )
        if PROMPT_INPUT_MARKER not in prompt:
            raise ValueError(f"prompt missing {PROMPT_INPUT_MARKER}")
        payload = extract_json_object(prompt.split(PROMPT_INPUT_MARKER, 1)[1])
        want = str(payload["want"])
        scores = []
        for c in payload.get("candidate_objects", []):
            name = str(c.get("name") or c.get("category") or "")
            mapped = c.get("canonical_object_want") or map_object_to_want(name)
            primary = (c.get("geometry_signals") or {}).get("primary", {})
            geom_avg = (
                sum(float(v) for v in primary.values()) / max(len(primary), 1) if primary else 0.0
            )

            if mapped == want:
                aff, sv, ap = 36, 24, 12
                trace_aff = f"{name} is a prototypical target for {want}."
                trace_sv = f"{name} appears unobstructed in this scene."
                trace_ap = f"{name} is a natural first pick for {want}."
            elif mapped == "use" and want == "use":
                aff, sv, ap = 28, 20, 9
                trace_aff = f"{name} is a plausible 'use' target."
                trace_sv = f"{name} is accessible in the scene."
                trace_ap = f"{name} is a reasonable but not top pick."
            else:
                aff, sv, ap = 4, 10, 3
                trace_aff = f"{name} is not a primary target for {want}."
                trace_sv = f"{name} is not in a state to satisfy {want}."
                trace_ap = f"{name} would only be picked as a last resort."

            # Map geom_avg in roughly [-1,1] to 1..17
            scaled = int(round(((geom_avg + 1.0) / 2.0) * 16.0)) + 1
            sgs = max(1, min(17, scaled))
            trace_sgs = (
                f"primary geometry mean {geom_avg:.2f}; "
                "positive values indicate the human is oriented toward this candidate."
            )

            rubric = aff + sv + ap + sgs
            scores.append(
                {
                    "object_id": c["object_id"],
                    "want_affordance": aff,
                    "want_state_viability": sv,
                    "action_priority": ap,
                    "scene_grounded_support": sgs,
                    "rubric": rubric,
                    "reasoning_trace": {
                        "affordance": trace_aff,
                        "state_viability": trace_sv,
                        "action_priority": trace_ap,
                        "scene_support": trace_sgs,
                    },
                }
            )
        return json.dumps({"sample_id": payload["sample_id"], "want": want, "semantic_scores": scores})


__all__ = [
    "DEFAULT_PROMPT_GEOMETRY",
    "MockSemanticLLMClient",
    "PROMPT_INPUT_MARKER",
    "REASONING_KEYS",
    "RUBRIC_MAX",
    "RUBRIC_MIN",
    "SCORE_RANGES",
    "build_semantic_prompt",
    "candidate_prompt_payload",
    "extract_json_object",
    "parse_semantic_response",
    "score_candidates_with_llm",
    "validate_semantic_result_for_sample",
]
