from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping

from .motivation_vocab import MOTIVATION_FIELDS, MotivationVocab
from .reasoning_factors import ReasoningFactorSpec


INSTANCE_RE = re.compile(r"instance_(\d+)$")


@dataclass(frozen=True)
class MotivationCode:
    need_state: str
    posture_mode: str
    support_type: str
    duration_type: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], vocab: MotivationVocab) -> "MotivationCode":
        validated = vocab.validate_code({field: str(payload.get(field, "")).strip() for field in MOTIVATION_FIELDS})
        return cls(**validated)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningFactors:
    scores: dict[str, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], spec: ReasoningFactorSpec) -> "ReasoningFactors":
        return cls(spec.validate_scores({name: float(payload[name]) for name in spec.names}))

    def to_dict(self) -> dict[str, float]:
        return dict(self.scores)


@dataclass(frozen=True)
class TeacherOutput:
    selected_object: str
    object_posterior: dict[str, float]
    motivation_code: MotivationCode
    reasoning_factors: ReasoningFactors
    rationale_short: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        candidate_ids: list[str],
        vocab: MotivationVocab,
        factor_spec: ReasoningFactorSpec,
        metadata_motivation: Mapping[str, str] | None = None,
        motivation_source: str = "metadata",
    ) -> tuple["TeacherOutput", list[str]]:
        warnings: list[str] = []
        candidate_set = set(candidate_ids)
        selected_object = str(payload.get("selected_object", "")).strip()
        if selected_object not in candidate_set:
            raise ValueError(f"selected_object {selected_object!r} is not in candidate ids")

        raw_posterior = payload.get("object_posterior", {})
        if not isinstance(raw_posterior, Mapping):
            raise ValueError("object_posterior must be an object keyed by candidate id")
        posterior = normalize_posterior(raw_posterior, candidate_ids, selected_object, warnings)

        if motivation_source == "metadata":
            if metadata_motivation is None:
                raise ValueError("metadata_motivation is required when motivation_source='metadata'")
            motivation_payload = metadata_motivation
        elif motivation_source == "llm":
            motivation_payload = payload.get("motivation_code", {})
        else:
            raise ValueError(f"Unknown motivation_source: {motivation_source!r}")
        motivation_code = MotivationCode.from_mapping(motivation_payload, vocab)

        reasoning_payload = payload.get("reasoning_factors", {})
        if not isinstance(reasoning_payload, Mapping):
            raise ValueError("reasoning_factors must be an object")
        reasoning_factors = ReasoningFactors.from_mapping(reasoning_payload, factor_spec)

        rationale_short = str(payload.get("rationale_short", "")).strip()
        if len(rationale_short.split()) > 40:
            warnings.append("rationale_short is longer than 40 words")

        top_object = max(posterior.items(), key=lambda item: item[1])[0]
        if top_object != selected_object:
            warnings.append(f"selected_object {selected_object} differs from posterior top1 {top_object}")

        return (
            cls(
                selected_object=selected_object,
                object_posterior=posterior,
                motivation_code=motivation_code,
                reasoning_factors=reasoning_factors,
                rationale_short=rationale_short,
            ),
            warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_object": self.selected_object,
            "object_posterior": dict(self.object_posterior),
            "motivation_code": self.motivation_code.to_dict(),
            "reasoning_factors": self.reasoning_factors.to_dict(),
            "rationale_short": self.rationale_short,
        }


def normalize_gt_instance(value: Any) -> str:
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


def normalize_posterior(
    raw_posterior: Mapping[str, Any],
    candidate_ids: list[str],
    selected_object: str,
    warnings: list[str],
) -> dict[str, float]:
    candidate_set = set(candidate_ids)
    posterior: dict[str, float] = {}
    extra = [key for key in raw_posterior if key not in candidate_set]
    if extra:
        warnings.append(f"Dropped posterior keys outside candidates: {extra}")
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in raw_posterior]
    if missing:
        warnings.append(f"Filled missing posterior candidates with 0: {missing}")
    for candidate_id in candidate_ids:
        value = raw_posterior.get(candidate_id, 0.0)
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
            warnings.append(f"Non-numeric posterior for {candidate_id}; set to 0")
        if not math.isfinite(score):
            score = 0.0
            warnings.append(f"Non-finite posterior for {candidate_id}; set to 0")
        posterior[candidate_id] = max(score, 0.0)
    total = sum(posterior.values())
    if total <= 1.0e-8:
        warnings.append("Posterior sum was zero; repaired to one-hot selected_object")
        posterior = {candidate_id: 0.0 for candidate_id in candidate_ids}
        posterior[selected_object] = 1.0
        return posterior
    return {candidate_id: posterior[candidate_id] / total for candidate_id in candidate_ids}


def teacher_output_json_schema(
    *,
    candidate_ids: list[str],
    vocab: MotivationVocab,
    factor_spec: ReasoningFactorSpec,
) -> dict[str, Any]:
    posterior_properties = {
        candidate_id: {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": f"Posterior probability for {candidate_id}.",
        }
        for candidate_id in candidate_ids
    }
    motivation_properties = {
        field: {
            "type": "string",
            "enum": list(vocab.allowed(field)),
        }
        for field in MOTIVATION_FIELDS
    }
    factor_properties = {
        name: {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": factor_spec.definitions[name],
        }
        for name in factor_spec.names
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "selected_object",
            "object_posterior",
            "motivation_code",
            "reasoning_factors",
            "rationale_short",
        ],
        "properties": {
            "selected_object": {
                "type": "string",
                "enum": candidate_ids,
            },
            "object_posterior": {
                "type": "object",
                "additionalProperties": False,
                "required": candidate_ids,
                "properties": posterior_properties,
            },
            "motivation_code": {
                "type": "object",
                "additionalProperties": False,
                "required": list(MOTIVATION_FIELDS),
                "properties": motivation_properties,
            },
            "reasoning_factors": {
                "type": "object",
                "additionalProperties": False,
                "required": list(factor_spec.names),
                "properties": factor_properties,
            },
            "rationale_short": {
                "type": "string",
                "description": "One short debugging sentence. Do not include hidden chain-of-thought.",
            },
        },
    }
