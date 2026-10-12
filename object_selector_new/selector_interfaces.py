"""Selector dataclass interfaces — pure-Python (no torch).

These mirror the structures used by the teacher pipeline and the CSV exporter.
Coordinate vectors are 3-tuples; positions, directions, and ray origins are
all in the same world frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


Vector3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_finite_float(value: Any, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v


def validate_vector3(value: Sequence[Any], name: str) -> Vector3:
    if value is None:
        raise ValueError(f"{name} is required")
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"{name} must have length 3, got {len(seq)}")
    return (
        validate_finite_float(seq[0], f"{name}[0]"),
        validate_finite_float(seq[1], f"{name}[1]"),
        validate_finite_float(seq[2], f"{name}[2]"),
    )


def validate_required_field(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} is required")
    return value


def validate_score_range(value: float, name: str, low: float, high: float) -> float:
    v = validate_finite_float(value, name)
    if not low <= v <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {v}")
    return v


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ray3D:
    origin: Vector3
    direction: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", validate_vector3(self.origin, "Ray3D.origin"))
        object.__setattr__(self, "direction", validate_vector3(self.direction, "Ray3D.direction"))


# ---------------------------------------------------------------------------
# Scene graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneGraphNode:
    node_id: str
    node_type: str           # "object" | "human" | other
    name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraphEdge:
    source_id: str
    target_id: str
    relation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraph:
    nodes: tuple[SceneGraphNode, ...]
    edges: tuple[SceneGraphEdge, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Object candidates and teacher input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectCandidate:
    object_id: str
    name: str
    category: str
    position: Vector3
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", validate_vector3(self.position, "ObjectCandidate.position"))


@dataclass(frozen=True)
class PastMotion:
    pelvis_trajectory: tuple[Vector3, ...] = ()
    head_trajectory: tuple[Vector3, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherInputSample:
    sample_id: str
    want: str
    supporting_actions: tuple[str, ...]
    scene_graph: SceneGraph
    object_candidates: tuple[ObjectCandidate, ...]
    face_normal_ray: Ray3D
    root_direction: Vector3
    torso_direction: Vector3
    pelvis_position: Vector3
    head_position: Vector3 | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Semantic scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SemanticScore:
    object_id: str
    match: int
    ease: int
    semantic_total: float
    reasoning_trace: tuple[str, ...]


@dataclass(frozen=True)
class SemanticScoringResult:
    sample_id: str
    want: str
    semantic_scores: Mapping[str, SemanticScore]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fusion scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateFusionScore:
    object_id: str
    final_score: float
    posterior: float
    feature_scores: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tie-break results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaceTieBreakResult:
    tie_set: tuple[str, ...]
    face_scores: Mapping[str, float]
    top_candidates: tuple[str, ...]
    still_tied: bool
    winner: str | None
    reasoning_trace: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BodyTieBreakResult:
    tie_set: tuple[str, ...]
    body_scores: Mapping[str, float]
    top_candidates: tuple[str, ...]
    still_tied: bool
    winner: str | None
    reasoning_trace: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Final teacher result
# ---------------------------------------------------------------------------
WinnerStage = str  # "fusion_score" | "semantic_argmax" | "face_tiebreak" | "body_tiebreak"


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


__all__ = [
    "BodyTieBreakResult",
    "CandidateFusionScore",
    "FaceTieBreakResult",
    "ObjectCandidate",
    "PastMotion",
    "Ray3D",
    "SceneGraph",
    "SceneGraphEdge",
    "SceneGraphNode",
    "SemanticScore",
    "SemanticScoringResult",
    "TeacherInputSample",
    "TeacherSelectionResult",
    "Vector3",
    "WinnerStage",
    "validate_finite_float",
    "validate_required_field",
    "validate_score_range",
    "validate_vector3",
]
