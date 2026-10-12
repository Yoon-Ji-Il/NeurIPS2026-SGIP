"""Teacher fusion scorer.

semantic_total ∈ [2, 10] is renormalized to [0, 1] so it sits on the same
scale as face/body alignment. Default weights treat the three signals
equally. Final fusion_total ∈ roughly [0, 3].
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from object_selector_new.common.selector_interfaces import (
    CandidateFusionScore,
    SemanticScore,
    SemanticScoringResult,
    TeacherInputSample,
    Vector3,
    validate_finite_float,
)
from object_selector_new.teacher.body_tiebreak import (
    DEFAULT_ROOT_WEIGHT,
    DEFAULT_TORSO_WEIGHT,
    compute_body_alignment_score,
)
from object_selector_new.teacher.face_tiebreak import compute_face_alignment_score


DEFAULT_FUSION_FEATURE_WEIGHTS: dict[str, float] = {
    "semantic_total_norm": 1.0,
    "face_alignment_norm": 1.0,
    "body_alignment_norm": 1.0,
}


SEMANTIC_TOTAL_MIN = 2.0
SEMANTIC_TOTAL_MAX = 10.0
SEMANTIC_TOTAL_RANGE = SEMANTIC_TOTAL_MAX - SEMANTIC_TOTAL_MIN  # 8.0


def normalize_semantic_total(value: float) -> float:
    v = (float(value) - SEMANTIC_TOTAL_MIN) / SEMANTIC_TOTAL_RANGE
    return min(1.0, max(0.0, v))


def normalize_cosine_score(value: float) -> float:
    v = (validate_finite_float(value, "cosine_score") + 1.0) / 2.0
    return min(1.0, max(0.0, v))


@dataclass(frozen=True)
class FusionScoringConfig:
    feature_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FUSION_FEATURE_WEIGHTS))
    temperature: float = 1.0
    body_root_weight: float = DEFAULT_ROOT_WEIGHT
    body_torso_weight: float = DEFAULT_TORSO_WEIGHT
    use_body_distance: bool = False

    def __post_init__(self) -> None:
        weights = {
            str(name): validate_finite_float(value, f"feature_weights[{name}]")
            for name, value in dict(self.feature_weights).items()
        }
        object.__setattr__(self, "feature_weights", weights)
        T = validate_finite_float(self.temperature, "temperature")
        if T <= 0:
            raise ValueError("temperature must be positive")
        object.__setattr__(self, "temperature", T)
        a = validate_finite_float(self.body_root_weight, "body_root_weight")
        b = validate_finite_float(self.body_torso_weight, "body_torso_weight")
        if a < 0 or b < 0:
            raise ValueError("body weights must be non-negative")
        if a == 0 and b == 0:
            raise ValueError("at least one body weight must be positive")
        object.__setattr__(self, "body_root_weight", a)
        object.__setattr__(self, "body_torso_weight", b)
        object.__setattr__(self, "use_body_distance", bool(self.use_body_distance))


def select_face_origin(sample: TeacherInputSample) -> Vector3:
    if sample.head_position is not None:
        return sample.head_position
    if sample.pelvis_position is not None:
        return sample.pelvis_position
    if sample.face_normal_ray.origin is not None:
        return sample.face_normal_ray.origin
    raise ValueError("Cannot choose face origin")


def _semantic_score_map(
    semantic_scores: Mapping[str, SemanticScore] | SemanticScoringResult,
) -> Mapping[str, SemanticScore]:
    return semantic_scores.semantic_scores if isinstance(semantic_scores, SemanticScoringResult) else semantic_scores


def compute_fusion_feature_scores(
    sample: TeacherInputSample,
    semantic_scores: Mapping[str, SemanticScore] | SemanticScoringResult,
    *,
    config: FusionScoringConfig | None = None,
) -> dict[str, dict[str, float]]:
    config = config or FusionScoringConfig()
    scores = _semantic_score_map(semantic_scores)
    cand_by_id = {c.object_id: c for c in sample.object_candidates}
    missing = [oid for oid in cand_by_id if oid not in scores]
    if missing:
        raise ValueError(f"semantic_scores missing candidates: {missing}")
    extra = sorted(set(scores) - set(cand_by_id))
    if extra:
        raise ValueError(f"semantic_scores has non-candidate ids: {extra}")

    face_origin = select_face_origin(sample)
    out: dict[str, dict[str, float]] = {}
    for c in sample.object_candidates:
        face_raw = compute_face_alignment_score(sample.face_normal_ray, face_origin, c.position)
        body_raw = compute_body_alignment_score(
            sample.root_direction,
            sample.torso_direction,
            sample.pelvis_position,
            c.position,
            use_distance=config.use_body_distance,
            alpha=config.body_root_weight,
            beta=config.body_torso_weight,
        )
        sem = scores[c.object_id]
        out[c.object_id] = {
            "semantic_total": sem.semantic_total,
            "semantic_total_norm": normalize_semantic_total(sem.semantic_total),
            "match": float(sem.match),
            "ease": float(sem.ease),
            "face_alignment_raw": face_raw,
            "face_alignment_norm": normalize_cosine_score(face_raw),
            "body_alignment_raw": body_raw,
            "body_alignment_norm": normalize_cosine_score(body_raw),
        }
    return out


def weighted_feature_sum(features: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(
        validate_finite_float(features.get(name, 0.0), f"feature[{name}]") * weight
        for name, weight in weights.items()
    )


def softmax_scores(scores: Mapping[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    T = validate_finite_float(temperature, "temperature")
    if T <= 0:
        raise ValueError("temperature must be positive")
    ids = list(scores)
    logits = [validate_finite_float(scores[i], f"scores[{i}]") / T for i in ids]
    m = max(logits)
    exp_v = [math.exp(l - m) for l in logits]
    total = sum(exp_v)
    return {i: e / total for i, e in zip(ids, exp_v, strict=True)}


def compute_fusion_scores(
    sample: TeacherInputSample,
    semantic_scores: Mapping[str, SemanticScore] | SemanticScoringResult,
    *,
    config: FusionScoringConfig | None = None,
) -> dict[str, CandidateFusionScore]:
    config = config or FusionScoringConfig()
    scores = _semantic_score_map(semantic_scores)
    feats = compute_fusion_feature_scores(sample, scores, config=config)
    finals = {oid: weighted_feature_sum(f, config.feature_weights) for oid, f in feats.items()}
    posteriors = softmax_scores(finals, temperature=config.temperature)
    return {
        oid: CandidateFusionScore(
            object_id=oid,
            final_score=finals[oid],
            posterior=posteriors[oid],
            feature_scores=feats[oid],
            metadata={
                "temperature": config.temperature,
                "feature_weights": dict(config.feature_weights),
                "semantic_normalization": "linear_2_to_10_to_0_1",
            },
        )
        for oid in finals
    }


def select_final_object_from_fusion_scores(fusion_scores: Mapping[str, CandidateFusionScore]) -> str | None:
    if not fusion_scores:
        return None
    return max(
        fusion_scores,
        key=lambda oid: (fusion_scores[oid].posterior, fusion_scores[oid].final_score, oid),
    )


__all__ = [
    "DEFAULT_FUSION_FEATURE_WEIGHTS",
    "FusionScoringConfig",
    "compute_fusion_feature_scores",
    "compute_fusion_scores",
    "normalize_cosine_score",
    "normalize_semantic_total",
    "select_face_origin",
    "select_final_object_from_fusion_scores",
    "softmax_scores",
    "weighted_feature_sum",
]
