from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from object_selector.common.want_taxonomy import CANONICAL_WANTS, map_object_to_want, normalize_object_name


DEFAULT_TEACHER_DIR = (
    Path(".")
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "past_motion_fusion_score_final"
)
DEFAULT_TEACHER_CANDIDATE_CSV = DEFAULT_TEACHER_DIR / "teacher_candidate_scores.csv"
DEFAULT_GT_EVAL_CSV = DEFAULT_TEACHER_DIR / "gt_eval_all.csv"
DEFAULT_SCENE_GRAPH_ROOT = Path("./object_selector/scene_graphs")
DEFAULT_SLICES_ROOT = Path("data/GIMO/SLICES_888s")


@dataclass(frozen=True)
class FreshStudentSelectorSample:
    sample_id: str
    split: str
    candidate_ids: list[str]
    candidate_names: list[str]
    label_ids: torch.Tensor
    candidate_features: torch.Tensor
    context_features: torch.Tensor
    motion_tokens: torch.Tensor
    semantic_target: torch.Tensor
    final_target: torch.Tensor
    component_target: torch.Tensor
    selected_idx: int
    gt_idx: int
    want_idx: int
    gt_object_id: str
    gt_object_name: str


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        output[clean_key] = str(value).strip() if value is not None else ""
    return output


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [_clean_row(row) for row in csv.DictReader(file)]


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _softmax(scores: list[float], temperature: float) -> torch.Tensor:
    if not scores:
        return torch.zeros((0,), dtype=torch.float32)
    raw = torch.tensor(scores, dtype=torch.float32)
    logits = raw / max(float(temperature), 1.0e-6)
    return torch.softmax(logits, dim=0)


def _normalize_probability(values: list[float], temperature: float = 1.0) -> torch.Tensor:
    if not values:
        return torch.zeros((0,), dtype=torch.float32)
    raw = torch.tensor([max(float(value), 0.0) for value in values], dtype=torch.float32)
    total = raw.sum()
    if float(total.item()) <= 0.0:
        return torch.full_like(raw, 1.0 / max(raw.numel(), 1))
    probs = raw / total
    if abs(float(temperature) - 1.0) <= 1.0e-8:
        return probs
    positive = probs > 0.0
    if not positive.any():
        return probs
    scaled = torch.zeros_like(probs)
    scaled[positive] = torch.softmax(probs[positive].clamp_min(1.0e-8).log() / max(float(temperature), 1.0e-6), dim=0)
    return scaled


def _load_scene_samples(scene_graph_root: Path, split: str) -> list[dict[str, Any]]:
    path = scene_graph_root / split / "samples.jsonl"
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _group_candidate_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        object_id = row.get("object_id", "")
        if not sample_id or not object_id:
            continue
        current = groups.setdefault(sample_id, {}).get(object_id)
        if current is None or _parse_float(row.get("fusion_total")) > _parse_float(current.get("fusion_total")):
            groups[sample_id][object_id] = row
    return groups


def _gt_lookup(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None or not Path(path).exists():
        return {}
    return {row["sample_id"]: row for row in _read_csv(path) if row.get("sample_id")}


def _horizontal_unit(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    horizontal = vector[..., [0, 2]]
    return horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(eps)


def _unit(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _bbox_distance(points: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    below = bbox_min - points
    above = points - bbox_max
    outside = torch.maximum(torch.maximum(below, above), torch.zeros_like(points))
    return outside.norm(dim=-1)


def _safe_joint(joints: torch.Tensor, index: int) -> torch.Tensor:
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints [T, J, 3], got {tuple(joints.shape)}")
    if index >= joints.shape[1]:
        return torch.zeros((3,), dtype=joints.dtype)
    return joints[-1, index]


def _motion_context(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pelvis = _safe_joint(joints, 0)
    head = _safe_joint(joints, 15)
    pelvis_start = joints[0, 0] if joints.shape[1] > 0 else torch.zeros((3,), dtype=joints.dtype)
    root_delta = pelvis - pelvis_start
    root_speed = root_delta.norm().view(1)
    root_dir = root_delta / root_delta.norm().clamp_min(1.0e-8)
    context = torch.cat([pelvis / 5.0, head / 5.0, root_dir, root_speed / 5.0], dim=0)
    return pelvis, head, root_dir, context


def _joint_sequence(joints: torch.Tensor, index: int) -> torch.Tensor:
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints [T, J, 3], got {tuple(joints.shape)}")
    if index >= joints.shape[1]:
        return torch.zeros((joints.shape[0], 3), dtype=joints.dtype)
    return joints[:, index]


def _motion_tokens(joints: torch.Tensor) -> torch.Tensor:
    pelvis = _joint_sequence(joints, 0)
    head = _joint_sequence(joints, 15)
    left_index = 20 if joints.shape[1] > 20 else 18
    right_index = 21 if joints.shape[1] > 21 else 19
    left_hand = _joint_sequence(joints, left_index)
    right_hand = _joint_sequence(joints, right_index)
    hand_center = (left_hand + right_hand) * 0.5

    zeros = torch.zeros((1, 3), dtype=joints.dtype)
    pelvis_step = torch.cat([zeros, pelvis[1:] - pelvis[:-1]], dim=0)
    head_step = torch.cat([zeros, head[1:] - head[:-1]], dim=0)
    hand_step = torch.cat([zeros, hand_center[1:] - hand_center[:-1]], dim=0)
    time = torch.linspace(0.0, 1.0, steps=joints.shape[0], dtype=joints.dtype).view(-1, 1)

    return torch.cat(
        [
            pelvis / 5.0,
            head / 5.0,
            hand_center / 5.0,
            (pelvis - pelvis[:1]) / 5.0,
            (head - head[:1]) / 5.0,
            (hand_center - hand_center[:1]) / 5.0,
            pelvis_step,
            head_step,
            hand_step,
            time,
        ],
        dim=-1,
    )


def _candidate_feature(
    *,
    scene_object: dict[str, Any],
    pelvis: torch.Tensor,
    head: torch.Tensor,
    root_dir: torch.Tensor,
    joints: torch.Tensor,
) -> torch.Tensor:
    centroid = torch.tensor(scene_object.get("centroid") or [0.0, 0.0, 0.0], dtype=torch.float32)
    bbox_min = torch.tensor(scene_object.get("bbox_min") or centroid.tolist(), dtype=torch.float32)
    bbox_max = torch.tensor(scene_object.get("bbox_max") or centroid.tolist(), dtype=torch.float32)
    bbox_size = (bbox_max - bbox_min).clamp_min(0.0)
    raw_num_points = _parse_float(scene_object.get("raw_num_points"), 0.0)

    rel_pelvis = centroid - pelvis
    rel_head = centroid - head
    pelvis_dist = rel_pelvis.norm().view(1)
    head_dist = rel_head.norm().view(1)
    root_align = (_horizontal_unit(rel_pelvis) * _horizontal_unit(root_dir)).sum().view(1)

    hand_indices = [index for index in (20, 21) if index < joints.shape[1]]
    if not hand_indices:
        hand_indices = [index for index in (18, 19) if index < joints.shape[1]]
    if hand_indices:
        past_hands = joints[:, hand_indices, :].reshape(-1, 3)
        last_hands = joints[-1, hand_indices, :]
        first_hand_center = joints[0, hand_indices, :].mean(dim=0)
        last_hand_center = last_hands.mean(dim=0)
        hand_delta = last_hand_center - first_hand_center
        rel_hand = centroid - last_hand_center
        last_hand_min_dist = (last_hands - centroid).norm(dim=-1).min().view(1)
        past_hand_bbox_dist = _bbox_distance(past_hands, bbox_min, bbox_max).min().view(1)
        last_hand_bbox_dist = _bbox_distance(last_hands, bbox_min, bbox_max).min().view(1)
        hand_motion_align = (_unit(hand_delta) * _unit(rel_hand)).sum().view(1)
    else:
        last_hand_min_dist = torch.zeros((1,), dtype=torch.float32)
        past_hand_bbox_dist = torch.zeros((1,), dtype=torch.float32)
        last_hand_bbox_dist = torch.zeros((1,), dtype=torch.float32)
        hand_motion_align = torch.zeros((1,), dtype=torch.float32)

    return torch.cat(
        [
            centroid / 5.0,
            bbox_size / 5.0,
            torch.tensor([math.log1p(max(raw_num_points, 0.0)) / 10.0], dtype=torch.float32),
            rel_pelvis / 5.0,
            pelvis_dist / 5.0,
            rel_head / 5.0,
            head_dist / 5.0,
            root_align,
            last_hand_min_dist / 5.0,
            last_hand_bbox_dist / 5.0,
            past_hand_bbox_dist / 5.0,
            hand_motion_align,
        ],
        dim=0,
    )


SCENE_GRAPH_METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "distance": (
        "root_centroid_distance_3d",
        "root_centroid_distance_xz",
        "head_centroid_distance_3d",
        "root_min_point_distance_3d",
        "head_min_point_distance_3d",
    ),
    "alignment": (
        "face_normal_alignment",
        "face_normal_horizontal_alignment",
        "root_alignment",
        "root_horizontal_alignment",
        "torso_alignment",
        "torso_horizontal_alignment",
    ),
    "line": (
        "root_line_distance_xz",
        "torso_line_distance_xz",
        "face_normal_line_distance_xz",
    ),
}
SCENE_GRAPH_METRIC_GROUPS["all"] = (
    *SCENE_GRAPH_METRIC_GROUPS["distance"],
    *SCENE_GRAPH_METRIC_GROUPS["alignment"],
    *SCENE_GRAPH_METRIC_GROUPS["line"],
)


def _scene_graph_metric_feature(scene_object: dict[str, Any], mode: str = "all") -> torch.Tensor:
    metrics = scene_object.get("metrics") or {}
    if mode not in SCENE_GRAPH_METRIC_GROUPS:
        raise ValueError(f"Unsupported scene graph metric mode: {mode}")
    values: list[float] = []
    for column in SCENE_GRAPH_METRIC_GROUPS[mode]:
        value = _parse_float(metrics.get(column), 0.0)
        if "distance" in column:
            value /= 5.0
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


def _scene_context(candidate_positions: torch.Tensor, valid_count: int) -> torch.Tensor:
    if candidate_positions.numel() == 0:
        return torch.zeros((10,), dtype=torch.float32)
    mean = candidate_positions.mean(dim=0)
    std = candidate_positions.std(dim=0, unbiased=False)
    extent = candidate_positions.max(dim=0).values - candidate_positions.min(dim=0).values
    return torch.cat([mean / 5.0, std / 5.0, extent / 5.0, torch.tensor([valid_count / 100.0])], dim=0)


def _normalize_component_value(field: str, value: float) -> float:
    if field in {"match", "ease"}:
        return value / 5.0
    if field == "semantic_total":
        return value / 10.0
    if field in {"fusion_total", "masked_fusion_total"}:
        return value / 15.0
    return value


def _label_from_row(row: dict[str, str], scene_object: dict[str, Any]) -> str:
    name = row.get("object_name") or row.get("name") or scene_object.get("label") or "unknown"
    return normalize_object_name(name) or str(name).strip().lower() or "unknown"


def _find_gt_candidate_idx(
    *,
    candidate_ids: list[str],
    candidate_names: list[str],
    gt_row: dict[str, str],
) -> int:
    gt_object_id = str(gt_row.get("gt_object_id", "")).strip()
    if gt_object_id:
        for index, object_id in enumerate(candidate_ids):
            if object_id == gt_object_id:
                return index
    gt_name = normalize_object_name(gt_row.get("gt_object_name", "")) or str(gt_row.get("gt_object_name", "")).strip()
    if gt_name:
        for index, name in enumerate(candidate_names):
            if normalize_object_name(name) == gt_name:
                return index
    return -1


def build_label_vocab(
    *,
    teacher_candidate_csv: str | Path = DEFAULT_TEACHER_CANDIDATE_CSV,
    scene_graph_root: str | Path = DEFAULT_SCENE_GRAPH_ROOT,
) -> dict[str, int]:
    rows = _read_csv(teacher_candidate_csv)
    labels = {"unknown"}
    for row in rows:
        labels.add(normalize_object_name(row.get("object_name") or row.get("name") or "unknown") or "unknown")
    for split in ("train", "test"):
        path = Path(scene_graph_root) / split / "samples.jsonl"
        if not path.exists():
            continue
        for sample in _load_scene_samples(Path(scene_graph_root), split):
            for item in sample.get("objects", []):
                labels.add(normalize_object_name(str(item.get("label", "unknown"))) or "unknown")
    return {label: index for index, label in enumerate(sorted(labels))}


class FreshStudentSelectorDataset(Dataset[FreshStudentSelectorSample]):
    def __init__(
        self,
        *,
        split: str,
        label_vocab: dict[str, int],
        teacher_candidate_csv: str | Path = DEFAULT_TEACHER_CANDIDATE_CSV,
        scene_graph_root: str | Path = DEFAULT_SCENE_GRAPH_ROOT,
        slices_root: str | Path = DEFAULT_SLICES_ROOT,
        gt_eval_csv: str | Path | None = DEFAULT_GT_EVAL_CSV,
        final_temperature: float = 1.0,
        semantic_temperature: float = 1.0,
        final_score_field: str = "fusion_total",
        final_target_mode: str = "score_softmax",
        semantic_score_field: str = "semantic_total",
        include_scene_graph_metrics: bool = False,
        scene_graph_metric_mode: str = "all",
        component_score_fields: tuple[str, ...] | list[str] | None = None,
        gt_calibration_weight: float = 0.0,
        gt_calibration_split: str = "train",
    ) -> None:
        self.split = split
        self.label_vocab = label_vocab
        self.want_vocab = {want: index for index, want in enumerate(CANONICAL_WANTS)}
        self.scene_graph_root = Path(scene_graph_root)
        self.slices_root = Path(slices_root)
        self.final_score_field = final_score_field
        self.final_target_mode = final_target_mode
        self.semantic_score_field = semantic_score_field
        self.include_scene_graph_metrics = include_scene_graph_metrics
        self.scene_graph_metric_mode = scene_graph_metric_mode
        self.component_score_fields = tuple(component_score_fields or ())
        self.gt_calibration_weight = float(gt_calibration_weight)
        self.gt_calibration_split = str(gt_calibration_split)
        if self.scene_graph_metric_mode not in SCENE_GRAPH_METRIC_GROUPS:
            raise ValueError(f"Unsupported scene_graph_metric_mode: {self.scene_graph_metric_mode}")
        if self.final_target_mode not in {"score_softmax", "probability"}:
            raise ValueError(f"Unsupported final_target_mode: {self.final_target_mode}")
        candidate_groups = _group_candidate_rows(_read_csv(teacher_candidate_csv))
        gt_rows = _gt_lookup(gt_eval_csv)
        scene_samples = _load_scene_samples(self.scene_graph_root, split)
        joints_input = torch.load(
            self.slices_root / split / "joints_input.pth",
            map_location="cpu",
            weights_only=False,
        ).float()
        if len(scene_samples) != int(joints_input.shape[0]):
            raise ValueError(
                f"{split}: scene sample count {len(scene_samples)} does not match joints_input {tuple(joints_input.shape)}"
            )
        self.samples = [
            self._build_sample(
                split=split,
                sample=sample,
                joints=joints_input[index],
                candidate_group=candidate_groups.get(str(sample["sample_id"]), {}),
                gt_row=gt_rows.get(str(sample["sample_id"]), {}),
            final_temperature=final_temperature,
            semantic_temperature=semantic_temperature,
            final_score_field=self.final_score_field,
            final_target_mode=self.final_target_mode,
            semantic_score_field=self.semantic_score_field,
            include_scene_graph_metrics=self.include_scene_graph_metrics,
            scene_graph_metric_mode=self.scene_graph_metric_mode,
            component_score_fields=self.component_score_fields,
            gt_calibration_weight=self.gt_calibration_weight,
            gt_calibration_split=self.gt_calibration_split,
        )
            for index, sample in enumerate(scene_samples)
        ]

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def num_wants(self) -> int:
        return len(self.want_vocab)

    @property
    def candidate_feature_dim(self) -> int:
        return int(self.samples[0].candidate_features.shape[-1])

    @property
    def context_feature_dim(self) -> int:
        return int(self.samples[0].context_features.shape[-1])

    @property
    def component_target_dim(self) -> int:
        return int(self.samples[0].component_target.shape[-1])

    @property
    def motion_token_dim(self) -> int:
        return int(self.samples[0].motion_tokens.shape[-1])

    def _build_sample(
        self,
        *,
        split: str,
        sample: dict[str, Any],
        joints: torch.Tensor,
        candidate_group: dict[str, dict[str, str]],
        gt_row: dict[str, str],
        final_temperature: float,
        semantic_temperature: float,
        final_score_field: str,
        final_target_mode: str,
        semantic_score_field: str,
        include_scene_graph_metrics: bool,
        scene_graph_metric_mode: str,
        component_score_fields: tuple[str, ...],
        gt_calibration_weight: float,
        gt_calibration_split: str,
    ) -> FreshStudentSelectorSample:
        sample_id = str(sample["sample_id"])
        pelvis, head, root_dir, motion_context = _motion_context(joints)
        motion_tokens = _motion_tokens(joints)
        object_by_id = {str(item.get("object_id", "")): item for item in sample.get("objects", [])}
        rows: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        for object_id in sorted(object_by_id):
            if object_id in candidate_group:
                rows.append((object_id, object_by_id[object_id], candidate_group[object_id]))
        if not rows:
            raise ValueError(f"{split} {sample_id}: no candidate score rows matched scene graph objects")

        candidate_ids: list[str] = []
        candidate_names: list[str] = []
        label_ids: list[int] = []
        candidate_features: list[torch.Tensor] = []
        candidate_positions: list[torch.Tensor] = []
        final_scores: list[float] = []
        semantic_scores: list[float] = []
        component_scores: list[list[float]] = []
        selected_idx = -1
        want = "unknown"
        for local_idx, (object_id, scene_object, row) in enumerate(rows):
            label = _label_from_row(row, scene_object)
            object_want = map_object_to_want(label)
            object_want_idx = self.want_vocab.get(object_want, self.want_vocab["unknown"])
            object_want_one_hot = torch.zeros((len(self.want_vocab),), dtype=torch.float32)
            object_want_one_hot[object_want_idx] = 1.0
            candidate_ids.append(object_id)
            candidate_names.append(label)
            label_ids.append(self.label_vocab.get(label, self.label_vocab["unknown"]))
            feature_parts = [
                _candidate_feature(
                    scene_object=scene_object,
                    pelvis=pelvis,
                    head=head,
                    root_dir=root_dir,
                    joints=joints,
                )
            ]
            if include_scene_graph_metrics:
                feature_parts.append(_scene_graph_metric_feature(scene_object, scene_graph_metric_mode))
            feature_parts.append(object_want_one_hot)
            candidate_features.append(torch.cat(feature_parts, dim=0))
            candidate_positions.append(
                torch.tensor(scene_object.get("centroid") or [0.0, 0.0, 0.0], dtype=torch.float32)
            )
            final_scores.append(_parse_float(row.get(final_score_field), _parse_float(row.get("fusion_total"), 0.0)))
            semantic_scores.append(
                _parse_float(row.get(semantic_score_field), _parse_float(row.get("semantic_total"), 0.0))
            )
            component_scores.append(
                [
                    _normalize_component_value(field, _parse_float(row.get(field), 0.0))
                    for field in component_score_fields
                ]
            )
            if _to_bool(row.get("selected")) or _to_bool(row.get("is_final")):
                selected_idx = local_idx
            if row.get("want"):
                want = row["want"]
        if selected_idx < 0:
            selected_idx = int(torch.tensor(final_scores).argmax().item())

        candidate_tensor = torch.stack(candidate_features, dim=0)
        position_tensor = torch.stack(candidate_positions, dim=0)
        context_features = torch.cat([motion_context, _scene_context(position_tensor, len(rows))], dim=0)
        semantic_target = _softmax(semantic_scores, semantic_temperature)
        if final_target_mode == "probability":
            final_target = _normalize_probability(final_scores, final_temperature)
        else:
            final_target = _softmax(final_scores, final_temperature)
        gt_idx = _find_gt_candidate_idx(
            candidate_ids=candidate_ids,
            candidate_names=candidate_names,
            gt_row=gt_row,
        )
        if gt_calibration_weight > 0.0 and split == gt_calibration_split:
            if 0 <= gt_idx < final_target.shape[0]:
                one_hot = torch.zeros_like(final_target)
                one_hot[gt_idx] = 1.0
                weight = max(0.0, min(float(gt_calibration_weight), 1.0))
                final_target = (1.0 - weight) * final_target + weight * one_hot
        component_target = (
            torch.tensor(component_scores, dtype=torch.float32)
            if component_score_fields
            else torch.zeros((len(rows), 0), dtype=torch.float32)
        )
        want_idx = self.want_vocab.get(want, self.want_vocab["unknown"])
        return FreshStudentSelectorSample(
            sample_id=sample_id,
            split=split,
            candidate_ids=candidate_ids,
            candidate_names=candidate_names,
            label_ids=torch.tensor(label_ids, dtype=torch.long),
            candidate_features=candidate_tensor,
            context_features=context_features,
            motion_tokens=motion_tokens,
            semantic_target=semantic_target,
            final_target=final_target,
            component_target=component_target,
            selected_idx=selected_idx,
            gt_idx=gt_idx,
            want_idx=want_idx,
            gt_object_id=gt_row.get("gt_object_id", ""),
            gt_object_name=normalize_object_name(gt_row.get("gt_object_name", "")) or gt_row.get("gt_object_name", ""),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FreshStudentSelectorSample:
        return self.samples[index]


def collate_fresh_student_samples(batch: list[FreshStudentSelectorSample]) -> dict[str, Any]:
    batch_size = len(batch)
    max_candidates = max(sample.candidate_features.shape[0] for sample in batch)
    candidate_dim = batch[0].candidate_features.shape[-1]
    context_dim = batch[0].context_features.shape[-1]

    candidate_features = torch.zeros((batch_size, max_candidates, candidate_dim), dtype=torch.float32)
    label_ids = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    semantic_target = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    final_target = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    component_dim = batch[0].component_target.shape[-1]
    component_target = torch.zeros((batch_size, max_candidates, component_dim), dtype=torch.float32)
    context_features = torch.zeros((batch_size, context_dim), dtype=torch.float32)
    max_motion_tokens = max(sample.motion_tokens.shape[0] for sample in batch)
    motion_dim = batch[0].motion_tokens.shape[-1]
    motion_tokens = torch.zeros((batch_size, max_motion_tokens, motion_dim), dtype=torch.float32)
    motion_token_mask = torch.zeros((batch_size, max_motion_tokens), dtype=torch.bool)
    selected_idx = torch.full((batch_size,), -1, dtype=torch.long)
    gt_idx = torch.full((batch_size,), -1, dtype=torch.long)
    want_idx = torch.zeros((batch_size,), dtype=torch.long)

    for row, sample in enumerate(batch):
        count = sample.candidate_features.shape[0]
        candidate_features[row, :count] = sample.candidate_features
        label_ids[row, :count] = sample.label_ids
        valid_mask[row, :count] = True
        semantic_target[row, :count] = sample.semantic_target
        final_target[row, :count] = sample.final_target
        if component_dim > 0:
            component_target[row, :count] = sample.component_target
        context_features[row] = sample.context_features
        token_count = sample.motion_tokens.shape[0]
        motion_tokens[row, :token_count] = sample.motion_tokens
        motion_token_mask[row, :token_count] = True
        selected_idx[row] = sample.selected_idx
        gt_idx[row] = sample.gt_idx
        want_idx[row] = sample.want_idx

    return {
        "candidate_features": candidate_features,
        "label_ids": label_ids,
        "valid_mask": valid_mask,
        "semantic_target": semantic_target,
        "final_target": final_target,
        "component_target": component_target,
        "context_features": context_features,
        "motion_tokens": motion_tokens,
        "motion_token_mask": motion_token_mask,
        "selected_idx": selected_idx,
        "gt_idx": gt_idx,
        "want_idx": want_idx,
        "sample_ids": [sample.sample_id for sample in batch],
        "candidate_ids": [sample.candidate_ids for sample in batch],
        "candidate_names": [sample.candidate_names for sample in batch],
        "gt_object_ids": [sample.gt_object_id for sample in batch],
        "gt_object_names": [sample.gt_object_name for sample in batch],
    }
