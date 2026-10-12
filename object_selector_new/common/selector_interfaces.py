from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from object_selector_new import selector_interfaces as _base
from object_selector_new.selector_interfaces import *  # noqa: F401,F403


@dataclass(frozen=True)
class SemanticScore:
    object_id: str
    want_affordance: int
    want_state_viability: int
    action_priority: int
    scene_grounded_support: int
    rubric: int
    reasoning_trace: Mapping[str, str]
    # Keys: "affordance", "state_viability", "action_priority", "scene_support"


@dataclass(frozen=True)
class SemanticScoringResult:
    sample_id: str
    want: str
    semantic_scores: Mapping[str, SemanticScore]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateFusionScore:
    object_id: str
    final_score: float
    posterior: float
    feature_scores: Mapping[str, float]
    rubric: int
    rubric_posterior: float
    motion_score: float
    is_in_tie_set: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionCueSpec:
    columns: tuple[str, ...]
    aggregation: str = "sum"
    weights: tuple[float, ...] | None = None
    normalize: str = "minmax_within_tieset"

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("MotionCueSpec.columns must be non-empty")
        if self.aggregation not in ("sum", "mean", "weighted_sum"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if self.aggregation == "weighted_sum":
            if self.weights is None or len(self.weights) != len(self.columns):
                raise ValueError("weighted_sum requires weights of same length as columns")
        if self.normalize not in ("minmax_within_tieset", "minmax_global", "none"):
            raise ValueError(f"unknown normalize {self.normalize!r}")


@dataclass(frozen=True)
class PromptGeometrySpec:
    primary_columns: tuple[str, ...]
    auxiliary_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.primary_columns:
            raise ValueError("PromptGeometrySpec.primary_columns must be non-empty")
        overlap = set(self.primary_columns) & set(self.auxiliary_columns)
        if overlap:
            raise ValueError(f"primary and auxiliary cannot overlap: {sorted(overlap)}")


# WinnerStage values: "rubric_argmax" | "tie_break_motion"
# (기존의 "fusion_score" | "semantic_argmax" | "face_tiebreak" | "body_tiebreak" 는 사용하지 않음)
WinnerStage = str


@dataclass(frozen=True)
class TeacherSelectionResult:
    sample_id: str
    want: str
    supporting_actions: tuple[str, ...]
    semantic_scores: Mapping[str, SemanticScore]
    tie_set: tuple[str, ...]
    face_tiebreak_result: FaceTieBreakResult | None
    body_tiebreak_result: BodyTieBreakResult | None
    final_object: str
    winner_stage: WinnerStage
    fusion_scores: Mapping[str, CandidateFusionScore] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [*getattr(_base, "__all__", ())]
for _name in ("MotionCueSpec", "PromptGeometrySpec"):
    if _name not in __all__:
        __all__.append(_name)
