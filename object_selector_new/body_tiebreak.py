"""Body-direction tie-breaking using root + torso cosine alignment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from object_selector_new.common.selector_interfaces import (
    Vector3,
    validate_vector3,
)


DEFAULT_BODY_TIE_EPSILON = 1.0e-6
DEFAULT_ROOT_WEIGHT = 0.5
DEFAULT_TORSO_WEIGHT = 0.5


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vector3) -> float:
    return math.sqrt(_dot(v, v))


def _cosine_similarity(a: Sequence[Any], b: Sequence[Any]) -> float:
    a_vec = validate_vector3(a, "direction")
    b_vec = validate_vector3(b, "object_vector")
    a_norm = _norm(a_vec)
    b_norm = _norm(b_vec)
    if a_norm <= 1.0e-12:
        raise ValueError("body direction must be non-zero")
    if b_norm <= 1.0e-12:
        return 0.0
    return _dot(a_vec, b_vec) / (a_norm * b_norm)


def _distance_weight(vector: Vector3) -> float:
    distance = _norm(vector)
    if distance <= 1.0e-12:
        return 1.0
    return 1.0 / (1.0 + distance)


def compute_body_alignment_score(
    root_direction: Sequence[Any],
    torso_direction: Sequence[Any],
    origin: Sequence[Any],
    object_position: Sequence[Any],
    use_distance: bool = False,
    *,
    alpha: float = DEFAULT_ROOT_WEIGHT,
    beta: float = DEFAULT_TORSO_WEIGHT,
) -> float:
    root_vec = validate_vector3(root_direction, "root_direction")
    torso_vec = validate_vector3(torso_direction, "torso_direction")
    origin_vec = validate_vector3(origin, "origin")
    object_vec = validate_vector3(object_position, "object_position")
    to_object = (
        object_vec[0] - origin_vec[0],
        object_vec[1] - origin_vec[1],
        object_vec[2] - origin_vec[2],
    )
    root_alignment = _cosine_similarity(root_vec, to_object)
    torso_alignment = _cosine_similarity(torso_vec, to_object)
    score = alpha * root_alignment + beta * torso_alignment
    if use_distance:
        score *= _distance_weight(to_object)
    return score


__all__ = [
    "DEFAULT_BODY_TIE_EPSILON",
    "DEFAULT_ROOT_WEIGHT",
    "DEFAULT_TORSO_WEIGHT",
    "compute_body_alignment_score",
]
