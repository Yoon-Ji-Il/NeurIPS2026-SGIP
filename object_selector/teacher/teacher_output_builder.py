from __future__ import annotations

from typing import Any, Mapping

from object_selector.common.motivation_vocab import MOTIVATION_FIELDS, MotivationVocab
from object_selector.common.reasoning_factors import ReasoningFactorSpec
from object_selector.common.schemas import TeacherOutput


class TeacherOutputBuilder:
    def __init__(
        self,
        *,
        vocab: MotivationVocab,
        factor_spec: ReasoningFactorSpec,
        motivation_source: str = "metadata",
    ) -> None:
        self.vocab = vocab
        self.factor_spec = factor_spec
        self.motivation_source = motivation_source

    def build(
        self,
        *,
        raw_output: Mapping[str, Any],
        scene_graph: Mapping[str, Any],
        metadata: Mapping[str, str],
    ) -> tuple[TeacherOutput, list[str]]:
        candidate_ids = [str(obj["object_id"]) for obj in scene_graph.get("objects", [])]
        metadata_motivation = {field: str(metadata.get(field, "")).strip() for field in MOTIVATION_FIELDS}
        return TeacherOutput.from_mapping(
            raw_output,
            candidate_ids=candidate_ids,
            vocab=self.vocab,
            factor_spec=self.factor_spec,
            metadata_motivation=metadata_motivation,
            motivation_source=self.motivation_source,
        )
