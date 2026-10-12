from __future__ import annotations

import json
from typing import Any, Mapping


GAZE_TEACHER_INSTRUCTIONS = """You are a gaze-guided teacher for human-object target selection.

Goal:
- Select the object candidate that is most consistent with the 3D gaze annotations.
- Treat gaze evidence as the primary signal.
- Use scene-graph labels and scene context only as secondary tie-breakers.
- Return only the strict JSON object requested by the output schema.

Primary evidence:
- gaze_annotation.points_3d
- candidate_objects[].gaze_metrics.bbox_hit_ratio
- candidate_objects[].gaze_metrics.mean_bbox_distance
- candidate_objects[].gaze_metrics.last_bbox_distance
- candidate_objects[].gaze_metrics.mean_centroid_distance
- candidate_objects[].gaze_metrics.prompt_score

Rules:
- Prefer objects whose bounding box is hit by gaze points or is very close to them.
- If gaze evidence is ambiguous, use object label and scene context as tie-breakers.
- Do not let auxiliary metadata overrule clear gaze evidence.
- Score every candidate object.
- selected_object must match the highest-scoring candidate in semantic_scores.
- reasoning_trace must be one short sentence per object.
- Return JSON only. No markdown.
"""


def _round_number(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _round_vector(values: Any, digits: int = 4) -> Any:
    if not isinstance(values, (list, tuple)):
        return values
    rounded: list[Any] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            rounded.append(_round_vector(value, digits=digits))
        else:
            rounded.append(_round_number(value, digits=digits))
    return rounded


def compact_scene_graph(scene_graph: Mapping[str, Any]) -> dict[str, Any]:
    human = scene_graph.get("human_final_frame", {})
    objects = []
    for obj in scene_graph.get("objects", []):
        metrics = obj.get("metrics", {})
        objects.append(
            {
                "object_id": obj.get("object_id", ""),
                "label": obj.get("label", ""),
                "semantic_label_id": obj.get("semantic_label_id"),
                "centroid": _round_vector(obj.get("centroid")),
                "bbox_min": _round_vector(obj.get("bbox_min")),
                "bbox_max": _round_vector(obj.get("bbox_max")),
                "metrics": {
                    "distance": _round_number(metrics.get("distance")),
                    "head_centroid_distance_3d": _round_number(metrics.get("head_centroid_distance_3d")),
                    "root_alignment": _round_number(metrics.get("root_alignment")),
                    "torso_alignment": _round_number(metrics.get("torso_alignment")),
                    "face_normal_alignment": _round_number(metrics.get("face_normal_alignment")),
                    "root_torso_alignment": _round_number(metrics.get("root_torso_alignment")),
                },
            }
        )
    return {
        "sample_id": scene_graph.get("sample_id", ""),
        "scene_name": scene_graph.get("scene_name", ""),
        "sequence_name": scene_graph.get("sequence_name", ""),
        "coordinate_frame": scene_graph.get("coordinate_frame", ""),
        "human_final_frame": {
            "pelvis": _round_vector(human.get("pelvis")),
            "head": _round_vector(human.get("head")),
            "eye_mid": _round_vector(human.get("eye_mid")),
            "root_direction": _round_vector(human.get("root_direction")),
            "torso_direction": _round_vector(human.get("torso_direction")),
            "face_normal_direction": _round_vector(human.get("face_normal_direction")),
            "valid": human.get("valid"),
        },
        "objects": objects,
    }


def build_gaze_teacher_prompt(
    *,
    scene_graph: Mapping[str, Any],
    metadata: Mapping[str, str],
    gaze_annotation: Mapping[str, Any],
    candidate_payloads: list[Mapping[str, Any]],
) -> str:
    payload = {
        "task": "Choose the target object primarily from 3D gaze evidence.",
        "primary_selection_policy": {
            "priority": "gaze evidence first, scene/object semantics second",
            "prefer": "high bbox_hit_ratio, low bbox distance, low centroid distance, high prompt_score",
            "tie_breaker": "object label and scene context only when gaze evidence is similar",
            "forbidden": "do not use auxiliary metadata to override clear gaze alignment",
        },
        "auxiliary_metadata": {
            "action": metadata.get("action_norm") or metadata.get("action") or "",
            "target_object_name": metadata.get("object_norm") or metadata.get("object") or "",
            "want_text": metadata.get("want_text", ""),
        },
        "gaze_annotation": {
            "selected_frame_indices": gaze_annotation.get("selected_frame_indices", []),
            "selected_frame_weights": _round_vector(gaze_annotation.get("selected_frame_weights", [])),
            "points_3d": _round_vector(gaze_annotation.get("points_3d", [])),
            "note": "Each point is a 3D gaze annotation in the same coordinate frame as the scene graph.",
        },
        "scene_graph": compact_scene_graph(scene_graph),
        "candidate_objects": candidate_payloads,
        "output_schema": {
            "sample_id": scene_graph.get("sample_id", ""),
            "selected_object": "candidate object id with the strongest gaze support",
            "semantic_scores": [
                {
                    "object_id": "candidate object id",
                    "match": "integer 1..5 for gaze consistency",
                    "ease": "integer 1..5 for overall plausibility given gaze and scene context",
                    "reasoning_trace": "one short sentence",
                }
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
