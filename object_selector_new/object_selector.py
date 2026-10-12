"""Teacher orchestrator.

Takes a TeacherInputSample, runs the LLM-based semantic scorer, fuses with
geometric alignment scores, and emits a TeacherSelectionResult containing
fusion_total + posterior + semantic scores per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from object_selector_new.common.selector_interfaces import (
    CandidateFusionScore,
    ObjectCandidate,
    SemanticScore,
    TeacherInputSample,
    TeacherSelectionResult,
)
from object_selector_new.common.supporting_action_taxonomy import (
    get_supporting_actions_for_want,
)
from object_selector_new.common.want_taxonomy import normalize_object_name
from object_selector_new.teacher.fusion_scorer import (
    FusionScoringConfig,
    compute_fusion_scores,
    select_final_object_from_fusion_scores,
)
from object_selector_new.teacher.semantic_scorer import score_candidates_with_llm


@dataclass
class TeacherObjectSelector:
    llm_client: Any
    fusion_config: FusionScoringConfig = field(default_factory=FusionScoringConfig)

    def select(self, sample: TeacherInputSample) -> TeacherSelectionResult:
        sample = self._prepare_sample(sample)
        semantic_result = score_candidates_with_llm(sample, self.llm_client)
        fusion_scores = compute_fusion_scores(
            sample, semantic_result, config=self.fusion_config
        )
        final_object = select_final_object_from_fusion_scores(fusion_scores)
        if final_object is None:
            raise ValueError("fusion produced no scores")

        return TeacherSelectionResult(
            sample_id=sample.sample_id,
            want=sample.want,
            supporting_actions=sample.supporting_actions,
            semantic_scores=semantic_result.semantic_scores,
            tie_set=(),
            face_tiebreak_result=None,
            body_tiebreak_result=None,
            final_object=final_object,
            winner_stage="fusion_score",
            fusion_scores=fusion_scores,
            metadata={
                "selection_mode": "fusion_score",
                "fusion_config": {
                    "feature_weights": dict(self.fusion_config.feature_weights),
                    "temperature": self.fusion_config.temperature,
                },
            },
        )

    def _prepare_sample(self, sample: TeacherInputSample) -> TeacherInputSample:
        supporting = tuple(get_supporting_actions_for_want(sample.want))
        normalized = tuple(
            ObjectCandidate(
                object_id=c.object_id,
                name=normalize_object_name(c.name) or c.name,
                category=normalize_object_name(c.category) or c.category,
                position=c.position,
                metadata={
                    **c.metadata,
                    "raw_name": c.metadata.get("raw_name", c.name),
                    "raw_category": c.metadata.get("raw_category", c.category),
                },
            )
            for c in sample.object_candidates
        )
        return TeacherInputSample(
            sample_id=sample.sample_id,
            want=sample.want,
            supporting_actions=supporting,
            scene_graph=sample.scene_graph,
            object_candidates=normalized,
            face_normal_ray=sample.face_normal_ray,
            root_direction=sample.root_direction,
            torso_direction=sample.torso_direction,
            pelvis_position=sample.pelvis_position,
            head_position=sample.head_position,
            metadata={**sample.metadata, "supporting_actions_source": "canonical_want_taxonomy"},
        )


__all__ = ["TeacherObjectSelector"]
