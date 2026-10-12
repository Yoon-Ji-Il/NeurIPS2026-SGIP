"""Shared geometry feature definitions for teacher export and student training."""

from __future__ import annotations


DEFAULT_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "distance",
    "root_centroid_distance_3d",
    "root_centroid_distance_xz",
    "head_centroid_distance_3d",
    "head_centroid_distance_xz",
    "root_min_point_distance_3d",
    "head_min_point_distance_3d",
    "face_normal_alignment",
    "face_normal_horizontal_alignment",
    "face_normal_raw_alignment",
    "face_normal_line_distance_xz",
    "root_alignment",
    "root_horizontal_alignment",
    "torso_alignment",
    "torso_horizontal_alignment",
    "root_torso_alignment",
    "root_torso_horizontal_alignment",
    "root_line_distance_xz",
    "torso_line_distance_xz",
)


__all__ = ["DEFAULT_GEOMETRY_COLUMNS"]
