"""Teacher fusion scorer (rubric softmax + motion tie-break).

Pipeline:
  1. Compute per-candidate rubric (already done by semantic_scorer).
  2. Convert rubric scalars to a distribution via softmax with temperature
     T_T:
         q_rubric_j = softmax(rubric_j / T_T)_j
  3. Detect "tie set": all candidates whose rubric is within `tie_threshold`
     of the top rubric. If |tie_set| >= 2, the rubric alone is too uncertain
     to commit, and we fall back to a motion cue.
  4. For tie cases, compute motion score per candidate from a configurable
     MotionCueSpec (which columns to combine, how to aggregate, how to
     normalize). Add the normalized motion score into the rubric_posterior
     for tie members only:
         final_j = q_rubric_j + motion_weight * motion_norm_j   if j in tie_set
         final_j = q_rubric_j                                    otherwise
     Renormalize `final` so it sums to 1 -> final_posterior.
  5. Winner = argmax(final_posterior).

Motion cue and rubric softmax are independently configurable so that the
caller can ablate prompt geometry, motion definition, temperature and tie
threshold separately.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from object_selector_new.common.selector_interfaces import (
    CandidateFusionScore,
    MotionCueSpec,
    ObjectCandidate,
    SemanticScore,
    SemanticScoringResult,
    TeacherInputSample,
    validate_finite_float,
)


# Default motion cue: same two signals exposed as primary in the prompt.
DEFAULT_MOTION_CUE = MotionCueSpec(
    columns=("root_horizontal_alignment", "face_normal_alignment"),
    aggregation="sum",
    normalize="minmax_within_tieset",
)


@dataclass(frozen=True)
class FusionScoringConfig:
    rubric_temperature: float = 20.0
    tie_threshold: int = 5
    motion_weight: float = 1.0
    motion_cue: MotionCueSpec = field(default_factory=lambda: DEFAULT_MOTION_CUE)

    def __post_init__(self) -> None:
        T = validate_finite_float(self.rubric_temperature, "rubric_temperature")
        if T <= 0:
            raise ValueError("rubric_temperature must be positive")
        object.__setattr__(self, "rubric_temperature", T)

        if not isinstance(self.tie_threshold, int) or self.tie_threshold < 0:
            raise ValueError("tie_threshold must be a non-negative integer")

        mw = validate_finite_float(self.motion_weight, "motion_weight")
        if mw < 0:
            raise ValueError("motion_weight must be non-negative")
        object.__setattr__(self, "motion_weight", mw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _semantic_score_map(
    semantic_scores: Mapping[str, SemanticScore] | SemanticScoringResult,
) -> Mapping[str, SemanticScore]:
    return (
        semantic_scores.semantic_scores
        if isinstance(semantic_scores, SemanticScoringResult)
        else semantic_scores
    )


def softmax_scores(scores: Mapping[str, float], *, temperature: float) -> dict[str, float]:
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


def _candidate_metric(candidate: ObjectCandidate, column: str) -> float:
    metrics = candidate.metadata.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        return 0.0
    return float(metrics.get(column, 0.0))


def _aggregate_motion(spec: MotionCueSpec, values: list[float]) -> float:
    if spec.aggregation == "sum":
        return float(sum(values))
    if spec.aggregation == "mean":
        return float(sum(values) / max(len(values), 1))
    if spec.aggregation == "weighted_sum":
        weights = list(spec.weights or ())
        return float(sum(w * v for w, v in zip(weights, values, strict=True)))
    raise ValueError(f"unknown aggregation {spec.aggregation!r}")


def _compute_motion_raw(
    sample: TeacherInputSample,
    spec: MotionCueSpec,
) -> dict[str, float]:
    """Raw (un-normalized) motion score per candidate."""
    out: dict[str, float] = {}
    for c in sample.object_candidates:
        values = [_candidate_metric(c, col) for col in spec.columns]
        out[c.object_id] = _aggregate_motion(spec, values)
    return out


def _normalize_motion(
    raw: Mapping[str, float],
    spec: MotionCueSpec,
    tie_set: set[str],
) -> dict[str, float]:
    """Normalize motion scores. By default, only candidates inside the tie set
    contribute to the min/max scaling, and only those candidates receive a
    nonzero normalized score."""
    if spec.normalize == "none":
        return {oid: float(raw[oid]) for oid in raw}

    if spec.normalize == "minmax_within_tieset":
        if not tie_set:
            return {oid: 0.0 for oid in raw}
        relevant = [raw[oid] for oid in tie_set]
        lo, hi = min(relevant), max(relevant)
        span = hi - lo
        out: dict[str, float] = {}
        for oid in raw:
            if oid not in tie_set:
                out[oid] = 0.0
            elif span <= 1.0e-12:
                out[oid] = 0.0
            else:
                out[oid] = (raw[oid] - lo) / span
        return out

    if spec.normalize == "minmax_global":
        lo, hi = min(raw.values()), max(raw.values())
        span = hi - lo
        if span <= 1.0e-12:
            return {oid: 0.0 for oid in raw}
        return {oid: (raw[oid] - lo) / span for oid in raw}

    raise ValueError(f"unknown normalize {spec.normalize!r}")


def _detect_tie_set(rubrics: Mapping[str, int], threshold: int) -> set[str]:
    if not rubrics:
        return set()
    top = max(rubrics.values())
    return {oid for oid, r in rubrics.items() if (top - r) <= threshold}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def compute_fusion_scores(
    sample: TeacherInputSample,
    semantic_scores: Mapping[str, SemanticScore] | SemanticScoringResult,
    *,
    config: FusionScoringConfig | None = None,
) -> dict[str, CandidateFusionScore]:
    config = config or FusionScoringConfig()
    scores = _semantic_score_map(semantic_scores)

    cand_by_id = {c.object_id: c for c in sample.object_candidates}
    missing = [oid for oid in cand_by_id if oid not in scores]
    if missing:
        raise ValueError(f"semantic_scores missing candidates: {missing}")
    extra = sorted(set(scores) - set(cand_by_id))
    if extra:
        raise ValueError(f"semantic_scores has non-candidate ids: {extra}")

    # 1) rubric posterior
    rubrics = {oid: int(scores[oid].rubric) for oid in cand_by_id}
    rubric_posterior = softmax_scores(
        {oid: float(rubrics[oid]) for oid in cand_by_id},
        temperature=config.rubric_temperature,
    )

    # 2) tie detection
    tie_set = _detect_tie_set(rubrics, config.tie_threshold)
    is_tie = len(tie_set) >= 2

    # 3) motion cue (raw + normalized)
    raw_motion = _compute_motion_raw(sample, config.motion_cue)
    motion_norm = _normalize_motion(raw_motion, config.motion_cue, tie_set if is_tie else set())

    # 4) final = rubric_posterior + motion_weight * motion_norm (only on tie members)
    final_unnorm: dict[str, float] = {}
    for oid in cand_by_id:
        bonus = config.motion_weight * motion_norm[oid] if is_tie else 0.0
        final_unnorm[oid] = rubric_posterior[oid] + bonus

    total = sum(final_unnorm.values())
    if total <= 0:
        # degenerate; fall back to rubric posterior
        final_posterior = dict(rubric_posterior)
    else:
        final_posterior = {oid: v / total for oid, v in final_unnorm.items()}

    out: dict[str, CandidateFusionScore] = {}
    for oid in cand_by_id:
        out[oid] = CandidateFusionScore(
            object_id=oid,
            final_score=final_unnorm[oid],
            posterior=final_posterior[oid],
            feature_scores={
                "rubric": float(rubrics[oid]),
                "rubric_posterior": rubric_posterior[oid],
                "motion_raw": raw_motion[oid],
                "motion_norm": motion_norm[oid],
            },
            rubric=rubrics[oid],
            rubric_posterior=rubric_posterior[oid],
            motion_score=motion_norm[oid] if is_tie else 0.0,
            is_in_tie_set=(oid in tie_set) if is_tie else False,
            metadata={
                "rubric_temperature": config.rubric_temperature,
                "tie_threshold": config.tie_threshold,
                "motion_weight": config.motion_weight,
                "motion_cue_columns": list(config.motion_cue.columns),
                "motion_cue_aggregation": config.motion_cue.aggregation,
                "motion_cue_normalize": config.motion_cue.normalize,
                "is_tie": is_tie,
                "tie_set_size": len(tie_set) if is_tie else 0,
            },
        )
    return out


def select_final_object_from_fusion_scores(
    fusion_scores: Mapping[str, CandidateFusionScore],
) -> str | None:
    if not fusion_scores:
        return None
    return max(
        fusion_scores,
        key=lambda oid: (fusion_scores[oid].posterior, fusion_scores[oid].final_score, oid),
    )


def determine_winner_stage(fusion_scores: Mapping[str, CandidateFusionScore]) -> str:
    """Return 'tie_break_motion' if the chosen winner sat inside a tie set,
    otherwise 'rubric_argmax'."""
    if not fusion_scores:
        return "rubric_argmax"
    any_tie = any(fs.is_in_tie_set for fs in fusion_scores.values())
    return "tie_break_motion" if any_tie else "rubric_argmax"


__all__ = [
    "DEFAULT_MOTION_CUE",
    "FusionScoringConfig",
    "compute_fusion_scores",
    "determine_winner_stage",
    "select_final_object_from_fusion_scores",
    "softmax_scores",
]
