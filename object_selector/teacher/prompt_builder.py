from __future__ import annotations

import json
from typing import Any, Mapping

from object_selector.common.motivation_vocab import MOTIVATION_FIELDS, MotivationVocab
from object_selector.common.reasoning_factors import ReasoningFactorSpec


TEACHER_INSTRUCTIONS = """You are a structured Teacher for human-object target selection.

Goal:
- Select the best target object candidate for the human's next interaction.
- Use the scene graph, final human pose/orientation cues, object relation metrics, and privileged want context.
- Return only the structured JSON object requested by the schema.

Rules:
- Do not output long chain-of-thought.
- The posterior must assign probability to every candidate object id.
- The posterior should sum to approximately 1.0.
- selected_object should be the highest-probability object.
- Motivation code must use the provided compact vocabulary.
- Reasoning factor scores must be continuous values in [0, 1].
- rationale_short is for logging only and must be brief.
"""


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
                "raw_num_points": obj.get("raw_num_points"),
                "centroid": obj.get("centroid"),
                "bbox_min": obj.get("bbox_min"),
                "bbox_max": obj.get("bbox_max"),
                "metrics": {
                    "distance": metrics.get("distance"),
                    "root_centroid_distance_3d": metrics.get("root_centroid_distance_3d"),
                    "head_centroid_distance_3d": metrics.get("head_centroid_distance_3d"),
                    "root_min_point_distance_3d": metrics.get("root_min_point_distance_3d"),
                    "head_min_point_distance_3d": metrics.get("head_min_point_distance_3d"),
                    "face_normal_alignment": metrics.get("face_normal_alignment"),
                    "face_normal_horizontal_alignment": metrics.get("face_normal_horizontal_alignment"),
                    "root_alignment": metrics.get("root_alignment"),
                    "root_horizontal_alignment": metrics.get("root_horizontal_alignment"),
                    "torso_alignment": metrics.get("torso_alignment"),
                    "torso_horizontal_alignment": metrics.get("torso_horizontal_alignment"),
                    "root_torso_alignment": metrics.get("root_torso_alignment"),
                    "root_line_distance_xz": metrics.get("root_line_distance_xz"),
                    "torso_line_distance_xz": metrics.get("torso_line_distance_xz"),
                },
            }
        )
    return {
        "sample_id": scene_graph.get("sample_id"),
        "scene_name": scene_graph.get("scene_name"),
        "sequence_name": scene_graph.get("sequence_name"),
        "coordinate_frame": scene_graph.get("coordinate_frame"),
        "human_final_frame": {
            "pelvis": human.get("pelvis"),
            "head": human.get("head"),
            "eye_mid": human.get("eye_mid"),
            "root_direction": human.get("root_direction"),
            "torso_direction": human.get("torso_direction"),
            "face_normal_direction": human.get("face_normal_direction"),
            "valid": human.get("valid"),
        },
        "objects": objects,
    }


def build_teacher_prompt(
    *,
    scene_graph: Mapping[str, Any],
    metadata: Mapping[str, str],
    vocab: MotivationVocab,
    factor_spec: ReasoningFactorSpec,
    include_gt_in_prompt: bool = False,
) -> str:
    motivation_hint = {
        field: str(metadata.get(field, "")).strip()
        for field in MOTIVATION_FIELDS
    }
    privileged_context: dict[str, Any] = {
        "action": metadata.get("action_norm") or metadata.get("action") or "",
        "target_object_name": metadata.get("object_norm") or metadata.get("object") or "",
        "want_text": metadata.get("want_text", ""),
        "motivation_code_hint": motivation_hint,
        "mapping_confidence": metadata.get("mapping_confidence", ""),
    }
    if include_gt_in_prompt:
        privileged_context["gt_instance_for_oracle_experiment"] = metadata.get("gt_object_id", "")

    payload = {
        "task": "Choose the target object and produce structured distillation targets.",
        "privileged_context": privileged_context,
        "motivation_vocabulary": vocab.prompt_payload(),
        "reasoning_factor_rubric": factor_spec.prompt_payload(),
        "scene_graph": compact_scene_graph(scene_graph),
        "posterior_guidance": {
            "high_probability": "object best satisfies function, posture, orientation, distance, and feasibility",
            "medium_probability": "plausible alternative or same semantic family but weaker geometry",
            "low_probability": "candidate is present but unlikely for the want",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
