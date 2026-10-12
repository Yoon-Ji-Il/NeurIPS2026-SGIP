"""Derived geometry-only candidate features.

These features are built strictly from existing per-candidate geometry columns,
with no privileged labels, wants, names, or future-frame information.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping


DerivedFeatureFn = Callable[[Mapping[str, float]], float]


def _positive(value: float) -> float:
    return float(value) if float(value) > 0.0 else 0.0


def _nonnegative(value: float) -> float:
    return float(value) if float(value) > 0.0 else 0.0


def face_root_product(features: Mapping[str, float]) -> float:
    return _positive(features["root_horizontal_alignment"]) * _positive(features["face_normal_alignment"])


def face_root_consensus(features: Mapping[str, float]) -> float:
    return min(_positive(features["root_horizontal_alignment"]), _positive(features["face_normal_alignment"]))


def root_frontness_over_xz(features: Mapping[str, float]) -> float:
    return _positive(features["root_horizontal_alignment"]) / (1.0 + _nonnegative(features["root_line_distance_xz"]))


def face_frontness_over_xz(features: Mapping[str, float]) -> float:
    return _positive(features["face_normal_horizontal_alignment"]) / (
        1.0 + _nonnegative(features["face_normal_line_distance_xz"])
    )


def face_root_disagreement(features: Mapping[str, float]) -> float:
    return abs(float(features["root_horizontal_alignment"]) - float(features["face_normal_alignment"]))


def body_fallback_alignment(features: Mapping[str, float]) -> float:
    return max(float(features["root_horizontal_alignment"]), float(features["torso_horizontal_alignment"]))


DERIVED_GEOMETRY_FEATURES: dict[str, DerivedFeatureFn] = {
    "derived_face_root_product": face_root_product,
    "derived_face_root_consensus": face_root_consensus,
    "derived_root_frontness_over_xz": root_frontness_over_xz,
    "derived_face_frontness_over_xz": face_frontness_over_xz,
    "derived_face_root_disagreement": face_root_disagreement,
    "derived_body_fallback_alignment": body_fallback_alignment,
}


__all__ = ["DERIVED_GEOMETRY_FEATURES", "DerivedFeatureFn"]
