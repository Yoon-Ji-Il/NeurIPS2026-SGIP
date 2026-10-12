"""Face-direction tie-breaking based on cosine alignment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from object_selector_new.common.selector_interfaces import (
    Ray3D,
    Vector3,
    validate_vector3,
)


DEFAULT_FACE_TIE_EPSILON = 1.0e-6


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vector3) -> float:
    return math.sqrt(_dot(v, v))


def _cosine_similarity(a: Vector3, b: Vector3) -> float:
    a_norm = _norm(a)
    b_norm = _norm(b)
    if a_norm <= 1.0e-12:
        raise ValueError("face direction must be non-zero")
    if b_norm <= 1.0e-12:
        return 0.0
    return _dot(a, b) / (a_norm * b_norm)


def _face_direction(face_normal_ray: Ray3D | Sequence[Any]) -> Vector3:
    if isinstance(face_normal_ray, Ray3D):
        return validate_vector3(face_normal_ray.direction, "face_normal_ray.direction")
    return validate_vector3(face_normal_ray, "face_normal_ray")


def compute_face_alignment_score(
    face_normal_ray: Ray3D | Sequence[Any],
    origin: Sequence[Any],
    object_position: Sequence[Any],
) -> float:
    face_direction = _face_direction(face_normal_ray)
    origin_vec = validate_vector3(origin, "origin")
    object_vec = validate_vector3(object_position, "object_position")
    to_object = (
        object_vec[0] - origin_vec[0],
        object_vec[1] - origin_vec[1],
        object_vec[2] - origin_vec[2],
    )
    return _cosine_similarity(face_direction, to_object)


__all__ = ["DEFAULT_FACE_TIE_EPSILON", "compute_face_alignment_score"]
