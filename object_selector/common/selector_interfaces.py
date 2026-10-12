from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, cast

from object_selector.common.want_taxonomy import validate_canonical_want


Vector3 = tuple[float, float, float]
MotionFrame = tuple[float, ...]
PastMotion = tuple[MotionFrame, ...]
WinnerStage = Literal["none", "semantic", "face_tiebreak", "body_tiebreak", "manual", "fusion_score"]


@dataclass(frozen=True)
class Ray3D:
    direction: Vector3
    origin: Vector3 | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", validate_vector3(self.direction, "face_normal_ray.direction"))
        if self.origin is not None:
            object.__setattr__(self, "origin", validate_vector3(self.origin, "face_normal_ray.origin"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Ray3D":
        return cls(
            origin=payload.get("origin"),
            direction=require_mapping_field(payload, "direction"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": vector_to_list(self.origin) if self.origin is not None else None,
            "direction": vector_to_list(self.direction),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ObjectCandidate:
    object_id: str
    name: str
    category: str
    position: Vector3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", validate_required_text(self.object_id, "object_id"))
        object.__setattr__(self, "name", validate_required_text(self.name, "name"))
        object.__setattr__(self, "category", validate_required_text(self.category, "category"))
        object.__setattr__(self, "position", validate_vector3(self.position, "position"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectCandidate":
        name = payload.get("name") or payload.get("label") or payload.get("semantic_label_name") or ""
        category = payload.get("category") or payload.get("label") or name
        return cls(
            object_id=payload.get("object_id") or payload.get("instance_id") or "",
            name=name,
            category=category,
            position=payload.get("position") or payload.get("centroid"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_scene_graph_object(cls, payload: Mapping[str, Any]) -> "ObjectCandidate":
        label = str(payload.get("label", "")).strip()
        metadata = {
            "semantic_label_id": payload.get("semantic_label_id"),
            "bbox_min": payload.get("bbox_min"),
            "bbox_max": payload.get("bbox_max"),
            "centroid": payload.get("centroid"),
            "raw_num_points": payload.get("raw_num_points"),
            "sampled_num_points_for_metrics": payload.get("sampled_num_points_for_metrics"),
            "source_ply_path": payload.get("source_ply_path"),
            "metrics": payload.get("metrics"),
        }
        return cls(
            object_id=payload.get("object_id", ""),
            name=label,
            category=label,
            position=payload.get("centroid"),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "category": self.category,
            "position": vector_to_list(self.position),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SceneGraphNode:
    node_id: str
    node_type: str
    name: str | None = None
    position: Vector3 | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", validate_required_text(self.node_id, "node_id"))
        object.__setattr__(self, "node_type", validate_required_text(self.node_type, "node_type"))
        if self.name is not None:
            object.__setattr__(self, "name", str(self.name).strip() or None)
        if self.position is not None:
            object.__setattr__(self, "position", validate_vector3(self.position, "position"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SceneGraphNode":
        return cls(
            node_id=payload.get("node_id") or payload.get("id") or payload.get("object_id") or "",
            node_type=payload.get("node_type") or payload.get("type") or "object",
            name=payload.get("name") or payload.get("label"),
            position=payload.get("position") or payload.get("centroid"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_object_candidate(cls, candidate: ObjectCandidate) -> "SceneGraphNode":
        return cls(
            node_id=candidate.object_id,
            node_type="object",
            name=candidate.name,
            position=candidate.position,
            metadata={"category": candidate.category, **dict(candidate.metadata)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "position": vector_to_list(self.position) if self.position is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SceneGraphEdge:
    source_id: str
    target_id: str
    relation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", validate_required_text(self.source_id, "source_id"))
        object.__setattr__(self, "target_id", validate_required_text(self.target_id, "target_id"))
        object.__setattr__(self, "relation", validate_required_text(self.relation, "relation"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SceneGraphEdge":
        return cls(
            source_id=payload.get("source_id") or payload.get("source") or "",
            target_id=payload.get("target_id") or payload.get("target") or "",
            relation=payload.get("relation") or payload.get("edge_type") or "",
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SceneGraph:
    nodes: tuple[SceneGraphNode, ...]
    edges: tuple[SceneGraphEdge, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = tuple(node if isinstance(node, SceneGraphNode) else SceneGraphNode.from_mapping(node) for node in self.nodes)
        edges = tuple(edge if isinstance(edge, SceneGraphEdge) else SceneGraphEdge.from_mapping(edge) for edge in self.edges)
        node_ids = validate_unique_ids((node.node_id for node in nodes), "scene graph node")
        node_id_set = set(node_ids)
        for edge in edges:
            # TODO: relax this if future graphs intentionally reference external nodes.
            if edge.source_id not in node_id_set or edge.target_id not in node_id_set:
                raise ValueError(
                    f"edge {edge.source_id!r}->{edge.target_id!r} references a missing scene graph node"
                )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SceneGraph":
        return cls(
            nodes=tuple(SceneGraphNode.from_mapping(node) for node in payload.get("nodes", ())),
            edges=tuple(SceneGraphEdge.from_mapping(edge) for edge in payload.get("edges", ())),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_scene_graph_cache(cls, payload: Mapping[str, Any]) -> "SceneGraph":
        candidates = tuple(ObjectCandidate.from_scene_graph_object(obj) for obj in payload.get("objects", ()))
        nodes = tuple(SceneGraphNode.from_object_candidate(candidate) for candidate in candidates)
        metadata = {
            "schema_version": payload.get("schema_version"),
            "split": payload.get("split"),
            "sample_index": payload.get("sample_index"),
            "sample_id": payload.get("sample_id"),
            "scene_name": payload.get("scene_name"),
            "sequence_name": payload.get("sequence_name"),
            "start_frame": payload.get("start_frame"),
            "end_frame": payload.get("end_frame"),
            "coordinate_frame": payload.get("coordinate_frame"),
            "human_final_frame": payload.get("human_final_frame"),
            # TODO: add explicit object-object and human-object relation edges when the relation ontology is finalized.
        }
        return cls(
            nodes=nodes,
            edges=(),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }


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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", validate_required_text(self.sample_id, "sample_id"))
        object.__setattr__(self, "want", validate_canonical_want(self.want))
        object.__setattr__(self, "supporting_actions", validate_text_tuple(self.supporting_actions, "supporting_actions"))
        scene_graph = self.scene_graph if isinstance(self.scene_graph, SceneGraph) else SceneGraph.from_mapping(self.scene_graph)
        object.__setattr__(self, "scene_graph", scene_graph)
        candidates = tuple(
            candidate if isinstance(candidate, ObjectCandidate) else ObjectCandidate.from_mapping(candidate)
            for candidate in self.object_candidates
        )
        validate_unique_ids((candidate.object_id for candidate in candidates), "object candidate")
        object.__setattr__(self, "object_candidates", candidates)
        face_normal_ray = self.face_normal_ray if isinstance(self.face_normal_ray, Ray3D) else Ray3D.from_mapping(self.face_normal_ray)
        object.__setattr__(self, "face_normal_ray", face_normal_ray)
        object.__setattr__(self, "root_direction", validate_vector3(self.root_direction, "root_direction"))
        object.__setattr__(self, "torso_direction", validate_vector3(self.torso_direction, "torso_direction"))
        object.__setattr__(self, "pelvis_position", validate_vector3(self.pelvis_position, "pelvis_position"))
        if self.head_position is not None:
            object.__setattr__(self, "head_position", validate_vector3(self.head_position, "head_position"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TeacherInputSample":
        return cls(
            sample_id=require_mapping_field(payload, "sample_id"),
            want=require_mapping_field(payload, "want"),
            supporting_actions=tuple(payload.get("supporting_actions") or ()),
            scene_graph=SceneGraph.from_mapping(require_mapping_field(payload, "scene_graph")),
            object_candidates=tuple(
                ObjectCandidate.from_mapping(candidate)
                for candidate in require_mapping_field(payload, "object_candidates")
            ),
            face_normal_ray=Ray3D.from_mapping(require_mapping_field(payload, "face_normal_ray")),
            root_direction=require_mapping_field(payload, "root_direction"),
            torso_direction=require_mapping_field(payload, "torso_direction"),
            pelvis_position=require_mapping_field(payload, "pelvis_position"),
            head_position=payload.get("head_position"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_scene_graph_and_dataset_row(
        cls,
        scene_graph_payload: Mapping[str, Any],
        dataset_row: Mapping[str, Any],
    ) -> "TeacherInputSample":
        human = scene_graph_payload.get("human_final_frame", {})
        return cls(
            sample_id=scene_graph_payload.get("sample_id") or sample_id_from_dataset_row(dataset_row),
            want=want_from_dataset_row(dataset_row),
            supporting_actions=supporting_actions_from_dataset_row(dataset_row),
            scene_graph=SceneGraph.from_scene_graph_cache(scene_graph_payload),
            object_candidates=tuple(
                ObjectCandidate.from_scene_graph_object(candidate)
                for candidate in scene_graph_payload.get("objects", ())
            ),
            face_normal_ray=Ray3D(
                origin=human.get("eye_mid"),
                direction=human.get("face_normal_direction"),
            ),
            root_direction=human.get("root_direction"),
            torso_direction=human.get("torso_direction"),
            pelvis_position=human.get("pelvis"),
            head_position=human.get("head"),
            metadata={
                "scene_name": scene_graph_payload.get("scene_name"),
                "sequence_name": scene_graph_payload.get("sequence_name"),
                "start_frame": scene_graph_payload.get("start_frame"),
                "end_frame": scene_graph_payload.get("end_frame"),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "want": self.want,
            "supporting_actions": list(self.supporting_actions),
            "scene_graph": self.scene_graph.to_dict(),
            "object_candidates": [candidate.to_dict() for candidate in self.object_candidates],
            "face_normal_ray": self.face_normal_ray.to_dict(),
            "root_direction": vector_to_list(self.root_direction),
            "torso_direction": vector_to_list(self.torso_direction),
            "pelvis_position": vector_to_list(self.pelvis_position),
            "head_position": vector_to_list(self.head_position) if self.head_position is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StudentInputSample:
    sample_id: str
    scene_graph: SceneGraph
    object_candidates: tuple[ObjectCandidate, ...]
    past_motion: PastMotion
    face_normal_ray: Ray3D | None = None
    root_direction: Vector3 | None = None
    torso_direction: Vector3 | None = None
    pelvis_position: Vector3 | None = None
    head_position: Vector3 | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", validate_required_text(self.sample_id, "sample_id"))
        scene_graph = self.scene_graph if isinstance(self.scene_graph, SceneGraph) else SceneGraph.from_mapping(self.scene_graph)
        object.__setattr__(self, "scene_graph", scene_graph)
        candidates = tuple(
            candidate if isinstance(candidate, ObjectCandidate) else ObjectCandidate.from_mapping(candidate)
            for candidate in self.object_candidates
        )
        validate_unique_ids((candidate.object_id for candidate in candidates), "object candidate")
        object.__setattr__(self, "object_candidates", candidates)
        object.__setattr__(self, "past_motion", validate_past_motion(self.past_motion, "past_motion"))
        if self.face_normal_ray is not None and not isinstance(self.face_normal_ray, Ray3D):
            object.__setattr__(self, "face_normal_ray", Ray3D.from_mapping(self.face_normal_ray))
        if self.root_direction is not None:
            object.__setattr__(self, "root_direction", validate_vector3(self.root_direction, "root_direction"))
        if self.torso_direction is not None:
            object.__setattr__(self, "torso_direction", validate_vector3(self.torso_direction, "torso_direction"))
        if self.pelvis_position is not None:
            object.__setattr__(self, "pelvis_position", validate_vector3(self.pelvis_position, "pelvis_position"))
        if self.head_position is not None:
            object.__setattr__(self, "head_position", validate_vector3(self.head_position, "head_position"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StudentInputSample":
        return cls(
            sample_id=require_mapping_field(payload, "sample_id"),
            scene_graph=SceneGraph.from_mapping(require_mapping_field(payload, "scene_graph")),
            object_candidates=tuple(
                ObjectCandidate.from_mapping(candidate)
                for candidate in require_mapping_field(payload, "object_candidates")
            ),
            past_motion=require_mapping_field(payload, "past_motion"),
            face_normal_ray=Ray3D.from_mapping(payload["face_normal_ray"]) if payload.get("face_normal_ray") else None,
            root_direction=payload.get("root_direction"),
            torso_direction=payload.get("torso_direction"),
            pelvis_position=payload.get("pelvis_position"),
            head_position=payload.get("head_position"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_scene_graph(
        cls,
        scene_graph_payload: Mapping[str, Any],
        *,
        past_motion: Sequence[Sequence[Any]],
        include_orientation_features: bool = True,
    ) -> "StudentInputSample":
        human = scene_graph_payload.get("human_final_frame", {}) if include_orientation_features else {}
        return cls(
            sample_id=scene_graph_payload.get("sample_id", ""),
            scene_graph=SceneGraph.from_scene_graph_cache(scene_graph_payload),
            object_candidates=tuple(
                ObjectCandidate.from_scene_graph_object(candidate)
                for candidate in scene_graph_payload.get("objects", ())
            ),
            past_motion=past_motion,
            face_normal_ray=(
                Ray3D(origin=human.get("eye_mid"), direction=human.get("face_normal_direction"))
                if human.get("face_normal_direction") is not None
                else None
            ),
            root_direction=human.get("root_direction"),
            torso_direction=human.get("torso_direction"),
            pelvis_position=human.get("pelvis"),
            head_position=human.get("head"),
            metadata={
                "scene_name": scene_graph_payload.get("scene_name"),
                "sequence_name": scene_graph_payload.get("sequence_name"),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scene_graph": self.scene_graph.to_dict(),
            "object_candidates": [candidate.to_dict() for candidate in self.object_candidates],
            "past_motion": [list(frame) for frame in self.past_motion],
            "face_normal_ray": self.face_normal_ray.to_dict() if self.face_normal_ray is not None else None,
            "root_direction": vector_to_list(self.root_direction) if self.root_direction is not None else None,
            "torso_direction": vector_to_list(self.torso_direction) if self.torso_direction is not None else None,
            "pelvis_position": vector_to_list(self.pelvis_position) if self.pelvis_position is not None else None,
            "head_position": vector_to_list(self.head_position) if self.head_position is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SemanticScore:
    object_id: str
    match: int
    ease: int
    semantic_total: float
    reasoning_trace: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", validate_required_text(self.object_id, "object_id"))
        object.__setattr__(self, "match", validate_score_range(self.match, "match", minimum=1, maximum=5))
        object.__setattr__(self, "ease", validate_score_range(self.ease, "ease", minimum=1, maximum=5))
        object.__setattr__(self, "semantic_total", validate_finite_float(self.semantic_total, "semantic_total"))
        object.__setattr__(self, "reasoning_trace", validate_text_tuple(self.reasoning_trace, "reasoning_trace", allow_empty=True))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SemanticScore":
        return cls(
            object_id=require_mapping_field(payload, "object_id"),
            match=require_mapping_field(payload, "match"),
            ease=require_mapping_field(payload, "ease"),
            semantic_total=require_mapping_field(payload, "semantic_total"),
            reasoning_trace=tuple(payload.get("reasoning_trace") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "match": self.match,
            "ease": self.ease,
            "semantic_total": self.semantic_total,
            "reasoning_trace": list(self.reasoning_trace),
        }


@dataclass(frozen=True)
class SemanticScoringResult:
    sample_id: str
    want: str
    semantic_scores: dict[str, SemanticScore]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", validate_required_text(self.sample_id, "sample_id"))
        object.__setattr__(self, "want", validate_canonical_want(self.want))
        scores = normalize_semantic_score_map(self.semantic_scores)
        object.__setattr__(self, "semantic_scores", scores)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SemanticScoringResult":
        return cls(
            sample_id=require_mapping_field(payload, "sample_id"),
            want=require_mapping_field(payload, "want"),
            semantic_scores={
                object_id: SemanticScore.from_mapping(score)
                for object_id, score in require_mapping_field(payload, "semantic_scores").items()
            },
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "want": self.want,
            "semantic_scores": {
                object_id: score.to_dict()
                for object_id, score in self.semantic_scores.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CandidateFusionScore:
    object_id: str
    final_score: float
    posterior: float
    feature_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", validate_required_text(self.object_id, "object_id"))
        object.__setattr__(self, "final_score", validate_finite_float(self.final_score, "final_score"))
        object.__setattr__(self, "posterior", validate_probability(self.posterior, "posterior"))
        object.__setattr__(self, "feature_scores", validate_float_map(self.feature_scores, "feature_scores"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateFusionScore":
        return cls(
            object_id=require_mapping_field(payload, "object_id"),
            final_score=require_mapping_field(payload, "final_score"),
            posterior=require_mapping_field(payload, "posterior"),
            feature_scores=dict(payload.get("feature_scores") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "final_score": self.final_score,
            "posterior": self.posterior,
            "feature_scores": dict(self.feature_scores),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FaceTieBreakResult:
    tie_set: tuple[str, ...]
    face_scores: dict[str, float]
    winner: str | None
    top_candidates: tuple[str, ...] = field(default_factory=tuple)
    still_tied: bool = False
    reasoning_trace: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tie_set = validate_unique_ids(self.tie_set, "tie_set")
        object.__setattr__(self, "tie_set", tie_set)
        scores = validate_tiebreak_scores(self.face_scores, tie_set, "face_scores")
        object.__setattr__(self, "face_scores", scores)
        if self.winner is not None:
            winner = validate_required_text(self.winner, "winner")
            if winner not in tie_set:
                raise ValueError(f"winner {winner!r} is not in tie_set")
            object.__setattr__(self, "winner", winner)
        top_candidates = tuple(validate_unique_ids(self.top_candidates, "top_candidates")) if self.top_candidates else tuple()
        unknown_top_ids = set(top_candidates) - set(tie_set)
        if unknown_top_ids:
            raise ValueError(f"top_candidates has ids not present in tie_set: {sorted(unknown_top_ids)}")
        if top_candidates and self.winner is not None and top_candidates != (self.winner,):
            raise ValueError("top_candidates must contain only winner when winner is unique")
        if self.still_tied and len(top_candidates) == 1:
            raise ValueError("still_tied cannot be true with a single top candidate")
        if not self.still_tied and len(top_candidates) > 1:
            raise ValueError("still_tied must be true with multiple top candidates")
        object.__setattr__(self, "top_candidates", top_candidates)
        object.__setattr__(self, "still_tied", bool(self.still_tied))
        object.__setattr__(self, "reasoning_trace", validate_text_tuple(self.reasoning_trace, "reasoning_trace", allow_empty=True))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FaceTieBreakResult":
        return cls(
            tie_set=tuple(require_mapping_field(payload, "tie_set")),
            face_scores=dict(require_mapping_field(payload, "face_scores")),
            winner=payload.get("winner"),
            top_candidates=tuple(payload.get("top_candidates") or ()),
            still_tied=bool(payload.get("still_tied", False)),
            reasoning_trace=tuple(payload.get("reasoning_trace") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tie_set": list(self.tie_set),
            "face_scores": dict(self.face_scores),
            "top_candidates": list(self.top_candidates),
            "still_tied": self.still_tied,
            "winner": self.winner,
            "reasoning_trace": list(self.reasoning_trace),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BodyTieBreakResult:
    tie_set: tuple[str, ...]
    body_scores: dict[str, float]
    winner: str | None
    top_candidates: tuple[str, ...] = field(default_factory=tuple)
    still_tied: bool = False
    reasoning_trace: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tie_set = validate_unique_ids(self.tie_set, "tie_set")
        object.__setattr__(self, "tie_set", tie_set)
        scores = validate_tiebreak_scores(self.body_scores, tie_set, "body_scores")
        object.__setattr__(self, "body_scores", scores)
        if self.winner is not None:
            winner = validate_required_text(self.winner, "winner")
            if winner not in tie_set:
                raise ValueError(f"winner {winner!r} is not in tie_set")
            object.__setattr__(self, "winner", winner)
        top_candidates = tuple(validate_unique_ids(self.top_candidates, "top_candidates")) if self.top_candidates else tuple()
        unknown_top_ids = set(top_candidates) - set(tie_set)
        if unknown_top_ids:
            raise ValueError(f"top_candidates has ids not present in tie_set: {sorted(unknown_top_ids)}")
        if top_candidates and self.winner is not None and top_candidates != (self.winner,):
            raise ValueError("top_candidates must contain only winner when winner is unique")
        if self.still_tied and len(top_candidates) == 1:
            raise ValueError("still_tied cannot be true with a single top candidate")
        if not self.still_tied and len(top_candidates) > 1:
            raise ValueError("still_tied must be true with multiple top candidates")
        object.__setattr__(self, "top_candidates", top_candidates)
        object.__setattr__(self, "still_tied", bool(self.still_tied))
        object.__setattr__(self, "reasoning_trace", validate_text_tuple(self.reasoning_trace, "reasoning_trace", allow_empty=True))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BodyTieBreakResult":
        return cls(
            tie_set=tuple(require_mapping_field(payload, "tie_set")),
            body_scores=dict(require_mapping_field(payload, "body_scores")),
            winner=payload.get("winner"),
            top_candidates=tuple(payload.get("top_candidates") or ()),
            still_tied=bool(payload.get("still_tied", False)),
            reasoning_trace=tuple(payload.get("reasoning_trace") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tie_set": list(self.tie_set),
            "body_scores": dict(self.body_scores),
            "top_candidates": list(self.top_candidates),
            "still_tied": self.still_tied,
            "winner": self.winner,
            "reasoning_trace": list(self.reasoning_trace),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TeacherSelectionResult:
    sample_id: str
    want: str
    supporting_actions: tuple[str, ...]
    semantic_scores: dict[str, SemanticScore]
    tie_set: tuple[str, ...]
    face_tiebreak_result: FaceTieBreakResult | None
    body_tiebreak_result: BodyTieBreakResult | None
    final_object: str | None
    winner_stage: WinnerStage
    fusion_scores: dict[str, CandidateFusionScore] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", validate_required_text(self.sample_id, "sample_id"))
        object.__setattr__(self, "want", validate_canonical_want(self.want))
        object.__setattr__(self, "supporting_actions", validate_text_tuple(self.supporting_actions, "supporting_actions"))
        scores = normalize_semantic_score_map(self.semantic_scores)
        object.__setattr__(self, "semantic_scores", scores)
        fusion_scores = normalize_fusion_score_map(self.fusion_scores)
        unknown_fusion_ids = set(fusion_scores) - set(scores)
        if unknown_fusion_ids:
            raise ValueError(f"fusion_scores has ids not present in semantic_scores: {sorted(unknown_fusion_ids)}")
        object.__setattr__(self, "fusion_scores", fusion_scores)
        tie_set = validate_unique_ids(self.tie_set, "tie_set") if self.tie_set else tuple()
        unknown_tie_ids = set(tie_set) - set(scores)
        if unknown_tie_ids:
            raise ValueError(f"tie_set has ids not present in semantic_scores: {sorted(unknown_tie_ids)}")
        object.__setattr__(self, "tie_set", tie_set)

        if self.face_tiebreak_result is not None and not isinstance(self.face_tiebreak_result, FaceTieBreakResult):
            object.__setattr__(self, "face_tiebreak_result", FaceTieBreakResult.from_mapping(self.face_tiebreak_result))
        if self.body_tiebreak_result is not None and not isinstance(self.body_tiebreak_result, BodyTieBreakResult):
            object.__setattr__(self, "body_tiebreak_result", BodyTieBreakResult.from_mapping(self.body_tiebreak_result))
        if self.final_object is not None:
            final_object = validate_required_text(self.final_object, "final_object")
            if final_object not in scores:
                raise ValueError(f"final_object {final_object!r} is not present in semantic_scores")
            object.__setattr__(self, "final_object", final_object)
        if self.winner_stage not in {"none", "semantic", "face_tiebreak", "body_tiebreak", "manual", "fusion_score"}:
            raise ValueError(f"Unknown winner_stage: {self.winner_stage!r}")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TeacherSelectionResult":
        return cls(
            sample_id=require_mapping_field(payload, "sample_id"),
            want=require_mapping_field(payload, "want"),
            supporting_actions=tuple(require_mapping_field(payload, "supporting_actions")),
            semantic_scores={
                object_id: SemanticScore.from_mapping(score)
                for object_id, score in require_mapping_field(payload, "semantic_scores").items()
            },
            tie_set=tuple(payload.get("tie_set") or ()),
            face_tiebreak_result=(
                FaceTieBreakResult.from_mapping(payload["face_tiebreak_result"])
                if payload.get("face_tiebreak_result") is not None
                else None
            ),
            body_tiebreak_result=(
                BodyTieBreakResult.from_mapping(payload["body_tiebreak_result"])
                if payload.get("body_tiebreak_result") is not None
                else None
            ),
            final_object=payload.get("final_object"),
            winner_stage=require_mapping_field(payload, "winner_stage"),
            fusion_scores=fusion_scores_from_payload(payload),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "want": self.want,
            "supporting_actions": list(self.supporting_actions),
            "semantic_scores": {
                object_id: score.to_dict()
                for object_id, score in self.semantic_scores.items()
            },
            "tie_set": list(self.tie_set),
            "face_tiebreak_result": (
                self.face_tiebreak_result.to_dict()
                if self.face_tiebreak_result is not None
                else None
            ),
            "body_tiebreak_result": (
                self.body_tiebreak_result.to_dict()
                if self.body_tiebreak_result is not None
                else None
            ),
            "final_object": self.final_object,
            "winner_stage": self.winner_stage,
            "fusion_scores": {
                object_id: score.to_dict()
                for object_id, score in self.fusion_scores.items()
            },
            "metadata": dict(self.metadata),
        }


def validate_required_field(value: Any, field_name: str) -> Any:
    if value is None:
        raise ValueError(f"{field_name} is required")
    return value


def require_mapping_field(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in payload or payload[field_name] is None:
        raise ValueError(f"{field_name} is required")
    return payload[field_name]


def validate_required_text(value: Any, field_name: str) -> str:
    validate_required_field(value, field_name)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def validate_vector_length(value: Any, expected_length: int, field_name: str) -> tuple[float, ...]:
    validate_required_field(value, field_name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of {expected_length} numeric values")
    if len(value) != expected_length:
        raise ValueError(f"{field_name} must have length {expected_length}, got {len(value)}")
    return tuple(validate_finite_float(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def validate_vector3(value: Any, field_name: str) -> Vector3:
    return cast(Vector3, validate_vector_length(value, 3, field_name))


def validate_score_range(value: Any, field_name: str, *, minimum: int = 1, maximum: int = 5) -> int:
    validate_required_field(value, field_name)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer in [{minimum}, {maximum}]")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer in [{minimum}, {maximum}]") from exc
    if str(value).strip() not in {str(number), f"{number}.0"}:
        try:
            if float(value) != float(number):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer in [{minimum}, {maximum}]") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{field_name} must be in [{minimum}, {maximum}], got {number}")
    return number


def validate_probability(value: Any, field_name: str) -> float:
    number = validate_finite_float(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")
    return number


def validate_finite_float(value: Any, field_name: str) -> float:
    validate_required_field(value, field_name)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def validate_text_tuple(values: Iterable[Any], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if values is None:
        raise ValueError(f"{field_name} is required")
    raw_values = (values,) if isinstance(values, (str, bytes)) else values
    texts = tuple(str(value).strip() for value in raw_values if str(value).strip())
    if not texts and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one non-empty string")
    return texts


def validate_unique_ids(values: Iterable[Any], field_name: str) -> tuple[str, ...]:
    raw_values = (values,) if isinstance(values, (str, bytes)) else values
    ids = tuple(validate_required_text(value, field_name) for value in raw_values)
    if not ids:
        raise ValueError(f"{field_name} must contain at least one id")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate ids: {duplicates}")
    return ids


def validate_past_motion(value: Any, field_name: str) -> PastMotion:
    validate_required_field(value, field_name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of per-frame numeric feature sequences")
    frames: list[MotionFrame] = []
    for frame_index, frame in enumerate(value):
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
            raise ValueError(f"{field_name}[{frame_index}] must be a numeric feature sequence")
        frames.append(tuple(validate_finite_float(item, f"{field_name}[{frame_index}][{item_index}]") for item_index, item in enumerate(frame)))
    if not frames:
        raise ValueError(f"{field_name} must contain at least one frame")
    # TODO: replace flat per-frame features with the final joint/tensor shape once the student input contract is fixed.
    return tuple(frames)


def validate_tiebreak_scores(values: Mapping[str, Any], tie_set: tuple[str, ...], field_name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping keyed by object id")
    scores = {
        validate_required_text(object_id, f"{field_name} key"): validate_finite_float(score, f"{field_name}[{object_id}]")
        for object_id, score in values.items()
    }
    missing = [object_id for object_id in tie_set if object_id not in scores]
    if missing:
        raise ValueError(f"{field_name} missing tie-set object ids: {missing}")
    extra = sorted(set(scores) - set(tie_set))
    if extra:
        raise ValueError(f"{field_name} has ids outside tie_set: {extra}")
    return scores


def validate_float_map(values: Mapping[str, Any], field_name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return {
        validate_required_text(key, f"{field_name} key"): validate_finite_float(value, f"{field_name}[{key}]")
        for key, value in values.items()
    }


def normalize_semantic_score_map(values: Mapping[str, Any]) -> dict[str, SemanticScore]:
    if not isinstance(values, Mapping):
        raise ValueError("semantic_scores must be a mapping keyed by object id")
    scores = {
        validate_required_text(object_id, "semantic_scores key"): (
            score if isinstance(score, SemanticScore) else SemanticScore.from_mapping(score)
        )
        for object_id, score in values.items()
    }
    validate_unique_ids(scores.keys(), "semantic_scores")
    mismatches = [object_id for object_id, score in scores.items() if object_id != score.object_id]
    if mismatches:
        raise ValueError(f"semantic_scores keys do not match score.object_id for: {mismatches}")
    return scores


def normalize_fusion_score_map(values: Mapping[str, Any] | None) -> dict[str, CandidateFusionScore]:
    if not values:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("fusion_scores must be a mapping keyed by object id")
    scores = {
        validate_required_text(object_id, "fusion_scores key"): (
            score if isinstance(score, CandidateFusionScore) else CandidateFusionScore.from_mapping(score)
        )
        for object_id, score in values.items()
    }
    validate_unique_ids(scores.keys(), "fusion_scores")
    mismatches = [object_id for object_id, score in scores.items() if object_id != score.object_id]
    if mismatches:
        raise ValueError(f"fusion_scores keys do not match score.object_id for: {mismatches}")
    # TODO: decide whether every export must enforce posterior sum == 1.0.
    return scores


def fusion_scores_from_payload(payload: Mapping[str, Any]) -> dict[str, CandidateFusionScore]:
    if payload.get("fusion_scores") is not None:
        return {
            object_id: CandidateFusionScore.from_mapping(score)
            for object_id, score in dict(payload["fusion_scores"]).items()
        }

    final_scores = payload.get("candidate_final_scores") or {}
    posteriors = payload.get("candidate_posteriors") or payload.get("candidate_posterior") or {}
    feature_scores = payload.get("fusion_feature_scores") or {}
    if not final_scores and not posteriors:
        return {}

    object_ids = set(final_scores) | set(posteriors)
    output: dict[str, CandidateFusionScore] = {}
    for object_id in object_ids:
        object_id_text = str(object_id)
        output[object_id_text] = CandidateFusionScore(
            object_id=object_id_text,
            final_score=final_scores.get(object_id, 0.0),
            posterior=posteriors.get(object_id, 0.0),
            feature_scores=dict(feature_scores.get(object_id, {})) if isinstance(feature_scores, Mapping) else {},
        )
    return output


def vector_to_list(vector: Vector3) -> list[float]:
    return [float(value) for value in vector]


def normalize_frame_value(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def sample_id_from_dataset_row(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scene", "")).strip(),
            str(row.get("sequence_path", "")).strip(),
            normalize_frame_value(row.get("start_frame", "")),
            normalize_frame_value(row.get("end_frame", "")),
        ]
    )


def want_from_dataset_row(row: Mapping[str, Any]) -> str:
    return validate_canonical_want(str(row.get("want", "")).strip())


def supporting_actions_from_dataset_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("supporting_actions") or row.get("actions") or row.get("action") or ""
    actions = tuple(part.strip() for part in re.split(r"[,;|]", str(raw)) if part.strip())
    if not actions:
        raise ValueError("supporting_actions/action is required")
    return actions


def validate_teacher_input_sample(sample: TeacherInputSample | Mapping[str, Any]) -> TeacherInputSample:
    return sample if isinstance(sample, TeacherInputSample) else TeacherInputSample.from_mapping(sample)


def validate_student_input_sample(sample: StudentInputSample | Mapping[str, Any]) -> StudentInputSample:
    return sample if isinstance(sample, StudentInputSample) else StudentInputSample.from_mapping(sample)


def validate_teacher_selection_result(
    result: TeacherSelectionResult | Mapping[str, Any],
) -> TeacherSelectionResult:
    return result if isinstance(result, TeacherSelectionResult) else TeacherSelectionResult.from_mapping(result)


def minimal_scene_graph_example() -> SceneGraph:
    candidate = ObjectCandidate(
        object_id="instance_0003",
        name="chair",
        category="chair",
        position=(1.0, 0.45, 2.0),
    )
    return SceneGraph(
        nodes=(
            SceneGraphNode(node_id="human", node_type="human", position=(0.0, 1.0, 0.0)),
            SceneGraphNode.from_object_candidate(candidate),
        ),
        edges=(
            SceneGraphEdge(source_id="human", target_id="instance_0003", relation="candidate_object"),
        ),
        metadata={"coordinate_frame": "example_frame"},
    )


def minimal_object_candidates_example() -> tuple[ObjectCandidate, ...]:
    return (
        ObjectCandidate(
            object_id="instance_0003",
            name="chair",
            category="chair",
            position=(1.0, 0.45, 2.0),
        ),
    )


def minimal_teacher_input_example() -> TeacherInputSample:
    return TeacherInputSample(
        sample_id="bedroom0122|2022-01-21-194925|242|528",
        want="sit",
        supporting_actions=("sit",),
        scene_graph=minimal_scene_graph_example(),
        object_candidates=minimal_object_candidates_example(),
        face_normal_ray=Ray3D(origin=(0.0, 1.6, 0.0), direction=(0.0, 0.0, 1.0)),
        root_direction=(0.0, 0.0, 1.0),
        torso_direction=(0.0, 0.0, 1.0),
        pelvis_position=(0.0, 1.0, 0.0),
        head_position=(0.0, 1.6, 0.0),
    )


def minimal_student_input_example() -> StudentInputSample:
    return StudentInputSample(
        sample_id="bedroom0122|2022-01-21-194925|242|528",
        scene_graph=minimal_scene_graph_example(),
        object_candidates=minimal_object_candidates_example(),
        past_motion=((0.0, 1.0, 0.0, 0.0, 1.6, 0.0),),
        root_direction=(0.0, 0.0, 1.0),
        torso_direction=(0.0, 0.0, 1.0),
    )


def minimal_teacher_selection_result_example() -> TeacherSelectionResult:
    return TeacherSelectionResult(
        sample_id="bedroom0122|2022-01-21-194925|242|528",
        want="sit",
        supporting_actions=("sit",),
        semantic_scores={
            "instance_0003": SemanticScore(
                object_id="instance_0003",
                match=5,
                ease=4,
                semantic_total=9.0,
                reasoning_trace=("chair is a direct sitting target",),
            ),
        },
        tie_set=(),
        face_tiebreak_result=None,
        body_tiebreak_result=None,
        final_object="instance_0003",
        winner_stage="fusion_score",
        fusion_scores={
            "instance_0003": CandidateFusionScore(
                object_id="instance_0003",
                final_score=9.75,
                posterior=1.0,
                feature_scores={
                    "semantic_total": 9.0,
                    "face_alignment": 0.5,
                    "body_alignment": 0.25,
                },
            ),
        },
    )
