#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.teacher.prompt_builder_gaze import (  # noqa: E402
    GAZE_TEACHER_INSTRUCTIONS,
    build_gaze_teacher_prompt,
)


PROMPT_INPUT_MARKER = "INPUT_JSON:"
MAX_REASONING_WORDS = 24
TOP1_TEACHER_INSTRUCTIONS = """You select exactly one target object from a noisy 3D scene graph.

Use want_text as the intended human goal. Combine it with the candidate labels,
human-relative scene-graph metrics, and 3D gaze evidence. The segmentation may be
noisy, so choose among the candidate IDs exactly as provided.

Return JSON only, with exactly these two keys:
{"sample_id":"the provided sample id","selected_object":"one provided candidate object id"}

Do not return scores, explanations, markdown, or any additional keys.
"""
SOFT_LOGIT_TEACHER_INSTRUCTIONS = """You select the intended target object from a noisy 3D scene graph.

Use want_text as the human goal. Combine it with each candidate's semantic label,
human-relative geometry, and 3D gaze evidence. Segmentation may delete, merge, split,
or partially corrupt instances, so judge only the candidates shown.

Return exactly one uppercase choice letter and no other text. The model's normalized
next-token logits over the listed choice letters will be used as a soft teacher
distribution; no ground-truth object identity is included in this prompt.
"""
DEFAULT_GIMO_ROOT = Path("data/GIMO")
DEFAULT_SLICE_ROOT = DEFAULT_GIMO_ROOT / "SLICES_888s"
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_METADATA_CSV = REPO_ROOT / "dataset" / "combined_dataset.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "object_selector" / "teacher_targets" / "qwen3_14b_gaze_scene_graph_v1"


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        row[clean_key] = value.strip() if isinstance(value, str) else str(value)
    return row


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                rows = [clean_row(row) for row in csv.DictReader(file)]
            if not rows:
                raise RuntimeError(f"No rows found in {path}")
            return rows
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Could not decode {path}; tried encodings: {errors}",
    )


def norm_value(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def build_sample_id(scene_name: str, sequence_name: str, start_frame: str, end_frame: str) -> str:
    return "|".join(
        (
            norm_value(scene_name),
            norm_value(sequence_name),
            norm_value(start_frame),
            norm_value(end_frame),
        )
    )


def split_name_from_row(row: dict[str, str]) -> str:
    training = norm_value(row.get("training", "0")).lower()
    if training in {"1", "true", "train"}:
        return "train"
    return "test"


def load_scene_graphs(scene_graph_root: Path, split: str) -> dict[str, dict[str, Any]]:
    jsonl_path = scene_graph_root / split / "samples.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing scene-graph jsonl: {jsonl_path}")
    lookup: dict[str, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            graph = json.loads(line)
            sample_uid = str(
                graph.get("sample_uid") or graph.get("sample_id", "")
            ).strip()
            if sample_uid:
                if sample_uid in lookup:
                    raise ValueError(
                        f"Duplicate scene-graph sample key: {sample_uid}"
                    )
                lookup[sample_uid] = graph
    return lookup


def load_gaze_tensor(slice_root: Path, split: str):
    import torch

    path = slice_root / split / "gazes.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing gaze tensor: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)!r}")
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected gaze tensor [N, T, G, 3], got {tuple(tensor.shape)}")
    return tensor.detach().cpu().float()


def select_gaze_frames(
    gaze_sequence: np.ndarray,
    *,
    last_k_frames: int,
    temporal_decay: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    frame_count = int(gaze_sequence.shape[0])
    if frame_count <= 0:
        raise ValueError("gaze_sequence must contain at least one frame")
    if last_k_frames <= 0 or last_k_frames >= frame_count:
        start = 0
    else:
        start = frame_count - last_k_frames
    frame_indices = list(range(start, frame_count))
    selected = gaze_sequence[frame_indices]
    weights = np.asarray(
        [float(temporal_decay) ** (len(frame_indices) - 1 - offset) for offset in range(len(frame_indices))],
        dtype=np.float32,
    )
    weights /= np.clip(weights.sum(), 1.0e-8, None)
    return selected, weights, frame_indices


def point_distance_scores(distances: np.ndarray, scale: float) -> np.ndarray:
    safe_scale = max(float(scale), 1.0e-6)
    return np.exp(-distances / safe_scale).astype(np.float32)


def bbox_hits(gaze_points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float) -> np.ndarray:
    lower = bbox_min[None, :] - float(margin)
    upper = bbox_max[None, :] + float(margin)
    return np.all((gaze_points >= lower) & (gaze_points <= upper), axis=1).astype(np.float32)


def bbox_distances(gaze_points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    clamped = np.minimum(np.maximum(gaze_points, bbox_min[None, :]), bbox_max[None, :])
    return np.linalg.norm(gaze_points - clamped, axis=1).astype(np.float32)


class LocalQwenSemanticClient:
    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-14B",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        quantization: str = "4bit",
        max_new_tokens: int = 2048,
    ) -> None:
        import importlib.util

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype!r}")
        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError(f"Unsupported quantization: {quantization!r}")

        compute_dtype = dtype_map[torch_dtype]
        if compute_dtype == "auto":
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        dtype_arg_name = "dtype" if int(transformers.__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
        model_kwargs[dtype_arg_name] = dtype_map[torch_dtype]

        if quantization != "none":
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError("bitsandbytes is required for 4bit/8bit Qwen loading.")
            if importlib.util.find_spec("accelerate") is None:
                raise ImportError("accelerate is required for 4bit/8bit Qwen loading.")
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto" if device == "auto" else {"": device}
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.input_device = self.model.device
        else:
            use_device_map = device == "auto" and importlib.util.find_spec("accelerate") is not None
            if use_device_map:
                model_kwargs["device_map"] = "auto"
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                self.input_device = self.model.device
            else:
                target_device = device
                if target_device == "auto":
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                self.model.to(torch.device(target_device))
                self.input_device = torch.device(target_device)

        self.model.eval()
        self.model.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, attr):
                setattr(self.model.generation_config, attr, None)
        self.max_new_tokens = int(max_new_tokens)

    def complete(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        encoded = self.tokenizer([chat_text], return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = int(encoded["input_ids"].shape[-1])
        output_ids = generated[0][input_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def choice_logits(self, prompt: str, choice_tokens: list[str]) -> dict[str, float]:
        """Return raw next-token logits for a closed set of one-token choices."""
        import torch

        if not choice_tokens:
            raise ValueError("choice_tokens must not be empty")
        token_ids: list[int] = []
        for choice in choice_tokens:
            encoded_choice = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded_choice) != 1:
                raise ValueError(
                    f"Soft-logit choice {choice!r} must tokenize to exactly one token, "
                    f"got {encoded_choice}"
                )
            token_ids.append(int(encoded_choice[0]))

        messages = [{"role": "user", "content": prompt}]
        try:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer([chat_text], return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.no_grad():
            next_token_logits = self.model(**encoded).logits[0, -1]
        return {
            choice: float(next_token_logits[token_id].float().item())
            for choice, token_id in zip(choice_tokens, token_ids)
        }


class MockGazeLLMClient:
    def complete(self, prompt: str) -> str:
        payload = extract_prompt_input_payload(prompt)
        scores = []
        for candidate in payload.get("candidate_objects", []):
            gaze_metrics = candidate.get("gaze_metrics", {})
            prompt_score = float(gaze_metrics.get("prompt_score") or 0.0)
            hit_ratio = float(gaze_metrics.get("bbox_hit_ratio") or 0.0)
            mean_dist = float(gaze_metrics.get("mean_bbox_distance") or 99.0)
            match = 5 if hit_ratio > 0.5 else 4 if prompt_score > 0.7 else 3 if prompt_score > 0.4 else 2
            ease = 5 if mean_dist < 0.15 else 4 if mean_dist < 0.35 else 3 if mean_dist < 0.75 else 2
            scores.append(
                {
                    "object_id": candidate["object_id"],
                    "match": int(max(1, min(5, match))),
                    "ease": int(max(1, min(5, ease))),
                    "reasoning_trace": f"{candidate.get('label', 'object')} is closest to the gaze points.",
                }
            )
        scores.sort(key=lambda item: (item["match"] + item["ease"], item["object_id"]), reverse=True)
        sample_id = str(
            payload.get("sample_id")
            or payload.get("scene_graph", {}).get("sample_id", "")
        )
        return json.dumps(
            {
                "sample_id": sample_id,
                "selected_object": scores[0]["object_id"],
                "semantic_scores": scores,
            },
            ensure_ascii=False,
        )

    def choice_logits(self, prompt: str, choice_tokens: list[str]) -> dict[str, float]:
        payload = extract_prompt_input_payload(prompt)
        candidates = payload.get("candidate_objects", [])
        if len(candidates) != len(choice_tokens):
            raise ValueError("Mock soft-logit choices do not match candidate count")
        output: dict[str, float] = {}
        for candidate in candidates:
            choice = str(candidate["choice"])
            gaze_metrics = candidate.get("gaze_metrics", {})
            hit_ratio = float(gaze_metrics.get("bbox_hit_ratio") or 0.0)
            prompt_score = float(gaze_metrics.get("prompt_score") or 0.0)
            mean_dist = float(gaze_metrics.get("mean_bbox_distance") or 99.0)
            output[choice] = 2.0 * hit_ratio + prompt_score - 0.5 * mean_dist
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a new gaze-aware Qwen selector without modifying existing training code. "
            "The script writes selector CSVs and can optionally build object point .pth files."
        )
    )
    parser.add_argument("--gimo-root", type=Path, default=DEFAULT_GIMO_ROOT)
    parser.add_argument("--gimo-dataset-csv", type=Path, default=DEFAULT_GIMO_ROOT / "dataset.csv")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--openins3d-root", type=Path, default=Path("data/OpenIns3D/output/results_demo"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sample-id", type=str, default="")
    parser.add_argument("--backend", choices=("qwen", "mock"), default="qwen")
    parser.add_argument(
        "--selection-mode",
        choices=("full_scores", "top1", "soft_logits"),
        default="full_scores",
        help=(
            "full_scores preserves the original per-candidate scoring prompt; "
            "top1 asks the same model for only one selected object; soft_logits "
            "normalizes Qwen next-token choice logits into a soft distribution."
        ),
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"), default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=4.0,
        help="Temperature applied to Qwen candidate logits in soft_logits mode.",
    )
    parser.add_argument("--last-k-frames", type=int, default=3)
    parser.add_argument("--temporal-decay", type=float, default=0.7)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    parser.add_argument("--point-distance-scale", type=float, default=0.20)
    parser.add_argument("--centroid-distance-scale", type=float, default=0.75)
    parser.add_argument("--bbox-hit-weight", type=float, default=1.5)
    parser.add_argument("--point-score-weight", type=float, default=1.0)
    parser.add_argument("--centroid-score-weight", type=float, default=0.35)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--build-points", action="store_true")
    parser.add_argument("--num-object-points", type=int, default=1024)
    parser.add_argument(
        "--object-output-name",
        type=str,
        default="final_object_qwen3_14b_gaze_scene_graph_points_1024.pth",
    )
    return parser.parse_args()


def normalize_number_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def sample_id_from_row(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scene", "")).strip(),
            str(row.get("sequence_path", "")).strip(),
            normalize_number_text(row.get("start_frame", "")),
            normalize_number_text(row.get("end_frame", "")),
        ]
    )


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    rows = load_csv_rows(path)
    for row in rows:
        if str(row.get("source", "data")).strip() not in {"", "data"}:
            continue
        if not str(row.get("sequence_path", "")).strip():
            continue
        sample_id = sample_id_from_row(row)
        cleaned = {key: str(value).strip() for key, value in row.items() if key is not None}
        metadata.setdefault(sample_id, cleaned)
    return metadata


def build_qwen_client(args: argparse.Namespace) -> Any:
    if args.backend == "mock":
        return MockGazeLLMClient()
    return LocalQwenSemanticClient(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
    )


def prompt_score_from_metrics(
    *,
    frame_weights: np.ndarray,
    point_distances: np.ndarray,
    centroid_distances: np.ndarray,
    bbox_hit_values: np.ndarray,
    args: argparse.Namespace,
) -> float:
    return float(
        np.sum(
            frame_weights
            * (
                args.bbox_hit_weight * bbox_hit_values
                + args.point_score_weight * point_distance_scores(point_distances, args.point_distance_scale)
                + args.centroid_score_weight
                * point_distance_scores(centroid_distances, args.centroid_distance_scale)
            )
        )
    )


def build_gaze_candidate_payloads(
    scene_graph: dict[str, Any],
    gaze_sequence: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_gaze_points, frame_weights, selected_frame_indices = select_gaze_frames(
        gaze_sequence,
        last_k_frames=args.last_k_frames,
        temporal_decay=args.temporal_decay,
    )
    gaze_annotation = {
        "selected_frame_indices": selected_frame_indices,
        "selected_frame_weights": [float(value) for value in frame_weights.tolist()],
        "points_3d": [[float(v) for v in point.tolist()] for point in selected_gaze_points],
    }

    candidate_payloads: list[dict[str, Any]] = []
    candidate_analysis_rows: list[dict[str, Any]] = []
    for object_row in scene_graph.get("objects", []):
        object_id = str(object_row.get("object_id", "")).strip()
        if not object_id:
            continue
        centroid = np.asarray(object_row.get("centroid", []), dtype=np.float32)
        bbox_min = np.asarray(object_row.get("bbox_min", []), dtype=np.float32)
        bbox_max = np.asarray(object_row.get("bbox_max", []), dtype=np.float32)
        if centroid.shape != (3,) or bbox_min.shape != (3,) or bbox_max.shape != (3,):
            continue

        point_distances = bbox_distances(selected_gaze_points, bbox_min, bbox_max)
        centroid_distances = np.linalg.norm(
            selected_gaze_points - centroid[None, :],
            axis=1,
        ).astype(np.float32)
        bbox_hit_values = bbox_hits(selected_gaze_points, bbox_min, bbox_max, args.bbox_margin)
        prompt_score = prompt_score_from_metrics(
            frame_weights=frame_weights,
            point_distances=point_distances,
            centroid_distances=centroid_distances,
            bbox_hit_values=bbox_hit_values,
            args=args,
        )
        metrics = object_row.get("metrics", {})
        payload = {
            "object_id": object_id,
            "label": object_row.get("label", ""),
            "semantic_label_id": object_row.get("semantic_label_id"),
            "centroid": [float(v) for v in centroid.tolist()],
            "bbox_min": [float(v) for v in bbox_min.tolist()],
            "bbox_max": [float(v) for v in bbox_max.tolist()],
            "scene_graph_metrics": {
                "distance": metrics.get("distance"),
                "head_centroid_distance_3d": metrics.get("head_centroid_distance_3d"),
                "root_alignment": metrics.get("root_alignment"),
                "torso_alignment": metrics.get("torso_alignment"),
                "face_normal_alignment": metrics.get("face_normal_alignment"),
                "root_torso_alignment": metrics.get("root_torso_alignment"),
            },
            "gaze_metrics": {
                "mean_bbox_distance": float(np.sum(frame_weights * point_distances)),
                "last_bbox_distance": float(point_distances[-1]),
                "mean_centroid_distance": float(np.sum(frame_weights * centroid_distances)),
                "last_centroid_distance": float(centroid_distances[-1]),
                "bbox_hit_ratio": float(np.sum(frame_weights * bbox_hit_values)),
                "last_bbox_hit": float(bbox_hit_values[-1]),
                "prompt_score": float(prompt_score),
            },
        }
        candidate_payloads.append(payload)
        candidate_analysis_rows.append(payload)

    candidate_payloads.sort(
        key=lambda item: (
            -float(item["gaze_metrics"]["prompt_score"]),
            float(item["gaze_metrics"]["mean_bbox_distance"]),
            item["object_id"],
        )
    )
    return gaze_annotation, candidate_payloads, candidate_analysis_rows


def build_prompt(
    *,
    scene_graph: dict[str, Any],
    metadata: dict[str, str],
    gaze_annotation: dict[str, Any],
    candidate_payloads: list[dict[str, Any]],
) -> str:
    payload = build_gaze_teacher_prompt(
        scene_graph=scene_graph,
        metadata=metadata,
        gaze_annotation=gaze_annotation,
        candidate_payloads=candidate_payloads,
    )
    return f"{GAZE_TEACHER_INSTRUCTIONS}\n{PROMPT_INPUT_MARKER}\n{payload}"


def rounded_payload(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [rounded_payload(item, digits=digits) for item in value]
    if isinstance(value, dict):
        return {
            str(key): rounded_payload(item, digits=digits)
            for key, item in value.items()
        }
    return value


def build_top1_prompt(
    *,
    scene_graph: dict[str, Any],
    metadata: dict[str, str],
    gaze_annotation: dict[str, Any],
    candidate_payloads: list[dict[str, Any]],
) -> str:
    candidates = []
    for candidate in candidate_payloads:
        candidates.append(
            {
                "object_id": candidate["object_id"],
                "label": candidate.get("label", ""),
                "semantic_label_id": candidate.get("semantic_label_id"),
                "centroid": candidate.get("centroid", []),
                "bbox_min": candidate.get("bbox_min", []),
                "bbox_max": candidate.get("bbox_max", []),
                "human_relative_metrics": candidate.get("scene_graph_metrics", {}),
                "gaze_metrics": candidate.get("gaze_metrics", {}),
            }
        )
    payload = {
        "sample_id": scene_graph.get("sample_id", ""),
        "want_text": metadata.get("want_text", ""),
        "scene_name": scene_graph.get("scene_name", ""),
        "human_final_frame": scene_graph.get("human_final_frame", {}),
        "gaze_annotation": gaze_annotation,
        "candidate_objects": candidates,
    }
    return (
        f"{TOP1_TEACHER_INSTRUCTIONS}\n"
        f"{PROMPT_INPUT_MARKER}\n"
        f"{json.dumps(rounded_payload(payload), ensure_ascii=False, separators=(',', ':'))}"
    )


def build_soft_logits_prompt(
    *,
    scene_graph: dict[str, Any],
    metadata: dict[str, str],
    gaze_annotation: dict[str, Any],
    candidate_payloads: list[dict[str, Any]],
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    if len(candidate_payloads) > 26:
        raise ValueError(
            f"soft_logits supports at most 26 candidates, got {len(candidate_payloads)}"
        )

    sample_id = str(scene_graph.get("sample_id", ""))
    shuffled = list(candidate_payloads)
    stable_seed = int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    random.Random(stable_seed).shuffle(shuffled)

    choice_candidates: list[tuple[str, dict[str, Any]]] = []
    prompt_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(shuffled):
        choice = chr(ord("A") + index)
        choice_candidates.append((choice, candidate))
        prompt_candidates.append(
            {
                "choice": choice,
                "object_id": candidate["object_id"],
                "label": candidate.get("label", ""),
                "semantic_label_id": candidate.get("semantic_label_id"),
                "centroid": candidate.get("centroid", []),
                "bbox_min": candidate.get("bbox_min", []),
                "bbox_max": candidate.get("bbox_max", []),
                "human_relative_metrics": candidate.get("scene_graph_metrics", {}),
                "gaze_metrics": candidate.get("gaze_metrics", {}),
            }
        )

    payload = {
        "sample_id": sample_id,
        "want_text": metadata.get("want_text", ""),
        "scene_name": scene_graph.get("scene_name", ""),
        "human_final_frame": scene_graph.get("human_final_frame", {}),
        "gaze_annotation": gaze_annotation,
        "candidate_objects": prompt_candidates,
        "valid_choices": [choice for choice, _ in choice_candidates],
    }
    prompt = (
        f"{SOFT_LOGIT_TEACHER_INSTRUCTIONS}\n"
        f"{PROMPT_INPUT_MARKER}\n"
        f"{json.dumps(rounded_payload(payload), ensure_ascii=False, separators=(',', ':'))}"
    )
    return prompt, choice_candidates


def extract_prompt_input_payload(prompt: str) -> dict[str, Any]:
    if PROMPT_INPUT_MARKER not in prompt:
        raise ValueError(f"Prompt is missing {PROMPT_INPUT_MARKER}")
    text = prompt.split(PROMPT_INPUT_MARKER, 1)[1].strip()
    return extract_json_object(text)


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(str(text).strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return dict(payload)


def validate_integer_score(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer in [1, 5]")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str) and re.fullmatch(r"[1-5]", value.strip()):
        score = int(value.strip())
    else:
        raise ValueError(f"{field_name} must be an integer in [1, 5]")
    if not 1 <= score <= 5:
        raise ValueError(f"{field_name} must be an integer in [1, 5]")
    return score


def validate_reasoning_trace(trace: str, object_id: str) -> str:
    text = str(trace or "").strip()
    if not text:
        raise ValueError(f"reasoning_trace for {object_id!r} must be non-empty")
    if len(text.split()) > MAX_REASONING_WORDS:
        raise ValueError(
            f"reasoning_trace for {object_id!r} is too long; max {MAX_REASONING_WORDS} words"
        )
    return text


def parse_llm_response(
    *,
    sample_id: str,
    response_text: str,
    candidate_payloads: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    payload = extract_json_object(response_text)
    warnings: list[str] = []
    response_sample_id = str(payload.get("sample_id", "")).strip()
    if response_sample_id and response_sample_id != sample_id:
        warnings.append(f"response_sample_id_mismatch:{response_sample_id}")

    candidate_ids = [str(item["object_id"]) for item in candidate_payloads]
    candidate_lookup = {str(item["object_id"]): item for item in candidate_payloads}
    raw_scores = payload.get("semantic_scores")
    if not isinstance(raw_scores, list):
        raise ValueError("semantic_scores must be a list")

    parsed_lookup: dict[str, dict[str, Any]] = {}
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict):
            continue
        object_id = str(raw_score.get("object_id", "")).strip()
        if object_id not in candidate_lookup or object_id in parsed_lookup:
            continue
        match = validate_integer_score(raw_score.get("match"), f"{object_id}.match")
        ease = validate_integer_score(raw_score.get("ease"), f"{object_id}.ease")
        trace = validate_reasoning_trace(raw_score.get("reasoning_trace"), object_id)
        parsed_lookup[object_id] = {
            "object_id": object_id,
            "match": int(match),
            "ease": int(ease),
            "semantic_total": float(match + ease),
            "reasoning_trace": trace,
            "label": candidate_lookup[object_id].get("label", ""),
            "gaze_prompt_score": float(candidate_lookup[object_id]["gaze_metrics"]["prompt_score"]),
            "bbox_hit_ratio": float(candidate_lookup[object_id]["gaze_metrics"]["bbox_hit_ratio"]),
            "mean_bbox_distance": float(candidate_lookup[object_id]["gaze_metrics"]["mean_bbox_distance"]),
            "mean_centroid_distance": float(
                candidate_lookup[object_id]["gaze_metrics"]["mean_centroid_distance"]
            ),
        }

    for candidate_id in candidate_ids:
        if candidate_id in parsed_lookup:
            continue
        warnings.append(f"missing_score_repaired:{candidate_id}")
        prompt_score = float(candidate_lookup[candidate_id]["gaze_metrics"]["prompt_score"])
        repaired_match = 2 if prompt_score > 0.3 else 1
        repaired_ease = 2 if prompt_score > 0.2 else 1
        parsed_lookup[candidate_id] = {
            "object_id": candidate_id,
            "match": repaired_match,
            "ease": repaired_ease,
            "semantic_total": float(repaired_match + repaired_ease),
            "reasoning_trace": "Repaired from prompt-side gaze prior because the LLM omitted this object.",
            "label": candidate_lookup[candidate_id].get("label", ""),
            "gaze_prompt_score": float(candidate_lookup[candidate_id]["gaze_metrics"]["prompt_score"]),
            "bbox_hit_ratio": float(candidate_lookup[candidate_id]["gaze_metrics"]["bbox_hit_ratio"]),
            "mean_bbox_distance": float(candidate_lookup[candidate_id]["gaze_metrics"]["mean_bbox_distance"]),
            "mean_centroid_distance": float(
                candidate_lookup[candidate_id]["gaze_metrics"]["mean_centroid_distance"]
            ),
        }

    parsed_scores = [parsed_lookup[candidate_id] for candidate_id in candidate_ids]
    parsed_scores.sort(
        key=lambda item: (
            -float(item["semantic_total"]),
            -float(item["gaze_prompt_score"]),
            float(item["mean_bbox_distance"]),
            item["object_id"],
        )
    )
    total = sum(float(item["semantic_total"]) for item in parsed_scores)
    if total <= 1.0e-8:
        for item in parsed_scores:
            item["posterior"] = 0.0
        parsed_scores[0]["posterior"] = 1.0
    else:
        for item in parsed_scores:
            item["posterior"] = float(item["semantic_total"] / total)
    for index, item in enumerate(parsed_scores, start=1):
        item["rank"] = index

    selected_object = str(payload.get("selected_object", "")).strip()
    top_object = parsed_scores[0]["object_id"]
    if selected_object not in candidate_lookup:
        warnings.append(f"selected_object_repaired:{selected_object or 'missing'}")
        selected_object = top_object
    elif selected_object != top_object:
        warnings.append(f"selected_object_top1_mismatch:{selected_object}->{top_object}")
        selected_object = top_object

    return selected_object, parsed_scores, warnings


def parse_top1_response(
    *,
    sample_id: str,
    response_text: str,
    candidate_payloads: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    payload = extract_json_object(response_text)
    warnings: list[str] = []
    response_sample_id = str(payload.get("sample_id", "")).strip()
    if response_sample_id and response_sample_id != sample_id:
        warnings.append(f"response_sample_id_mismatch:{response_sample_id}")

    candidate_lookup = {
        str(candidate["object_id"]): candidate
        for candidate in candidate_payloads
    }
    selected_object = str(payload.get("selected_object", "")).strip()
    if selected_object not in candidate_lookup:
        raise ValueError(
            f"selected_object must be one of the provided candidates, got {selected_object!r}"
        )

    parsed_rows: list[dict[str, Any]] = []
    ordered_candidates = [candidate_lookup[selected_object]] + [
        candidate
        for candidate in candidate_payloads
        if str(candidate["object_id"]) != selected_object
    ]
    for index, candidate in enumerate(ordered_candidates):
        gaze_metrics = candidate["gaze_metrics"]
        parsed_rows.append(
            {
                "object_id": str(candidate["object_id"]),
                "label": candidate.get("label", ""),
                "match": "",
                "ease": "",
                "semantic_total": "",
                "posterior": "",
                "reasoning_trace": "",
                "rank": 1 if index == 0 else "",
                "gaze_prompt_score": float(gaze_metrics["prompt_score"]),
                "bbox_hit_ratio": float(gaze_metrics["bbox_hit_ratio"]),
                "mean_bbox_distance": float(gaze_metrics["mean_bbox_distance"]),
                "mean_centroid_distance": float(gaze_metrics["mean_centroid_distance"]),
            }
        )
    return selected_object, parsed_rows, warnings


def parse_soft_logits(
    *,
    choice_candidates: list[tuple[str, dict[str, Any]]],
    choice_logits: dict[str, float],
    temperature: float,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    safe_temperature = float(temperature)
    if not math.isfinite(safe_temperature) or safe_temperature <= 0.0:
        raise ValueError("--softmax-temperature must be finite and > 0")

    expected_choices = [choice for choice, _ in choice_candidates]
    if set(choice_logits) != set(expected_choices):
        missing = sorted(set(expected_choices) - set(choice_logits))
        extra = sorted(set(choice_logits) - set(expected_choices))
        raise ValueError(f"choice logit mismatch: missing={missing}, extra={extra}")

    raw_logits = np.asarray(
        [float(choice_logits[choice]) for choice in expected_choices],
        dtype=np.float64,
    )
    if not np.isfinite(raw_logits).all():
        raise ValueError("Qwen returned a non-finite candidate logit")
    scaled = raw_logits / safe_temperature
    scaled -= float(scaled.max())
    probabilities = np.exp(scaled)
    probabilities /= np.clip(probabilities.sum(), 1.0e-12, None)

    parsed_rows: list[dict[str, Any]] = []
    for (choice, candidate), teacher_logit, posterior in zip(
        choice_candidates,
        raw_logits.tolist(),
        probabilities.tolist(),
    ):
        gaze_metrics = candidate["gaze_metrics"]
        scene_metrics = candidate.get("scene_graph_metrics", {})
        bbox_min = np.asarray(candidate.get("bbox_min", [0.0, 0.0, 0.0]), dtype=np.float64)
        bbox_max = np.asarray(candidate.get("bbox_max", [0.0, 0.0, 0.0]), dtype=np.float64)
        bbox_size = np.maximum(bbox_max - bbox_min, 0.0)
        parsed_rows.append(
            {
                "object_id": str(candidate["object_id"]),
                "label": candidate.get("label", ""),
                "category": candidate.get("label", ""),
                "semantic_label_id": candidate.get("semantic_label_id", ""),
                "choice_token": choice,
                "match": "",
                "ease": "",
                "semantic_total": float(teacher_logit),
                "teacher_logit": float(teacher_logit),
                "posterior": float(posterior),
                "reasoning_trace": "",
                "gaze_prompt_score": float(gaze_metrics["prompt_score"]),
                "bbox_hit_ratio": float(gaze_metrics["bbox_hit_ratio"]),
                "last_bbox_hit": float(gaze_metrics["last_bbox_hit"]),
                "mean_bbox_distance": float(gaze_metrics["mean_bbox_distance"]),
                "last_bbox_distance": float(gaze_metrics["last_bbox_distance"]),
                "mean_centroid_distance": float(gaze_metrics["mean_centroid_distance"]),
                "last_centroid_distance": float(gaze_metrics["last_centroid_distance"]),
                "distance": float(scene_metrics.get("distance") or 0.0),
                "head_centroid_distance_3d": float(
                    scene_metrics.get("head_centroid_distance_3d") or 0.0
                ),
                "root_alignment": float(scene_metrics.get("root_alignment") or 0.0),
                "torso_alignment": float(scene_metrics.get("torso_alignment") or 0.0),
                "face_normal_alignment": float(
                    scene_metrics.get("face_normal_alignment") or 0.0
                ),
                "root_torso_alignment": float(
                    scene_metrics.get("root_torso_alignment") or 0.0
                ),
                "bbox_size_x": float(bbox_size[0]),
                "bbox_size_y": float(bbox_size[1]),
                "bbox_size_z": float(bbox_size[2]),
                "bbox_volume": float(np.prod(bbox_size)),
            }
        )

    parsed_rows.sort(
        key=lambda item: (
            -float(item["posterior"]),
            -float(item["gaze_prompt_score"]),
            item["object_id"],
        )
    )
    for rank, item in enumerate(parsed_rows, start=1):
        item["rank"] = rank
    return parsed_rows[0]["object_id"], parsed_rows, []


def noisy_gt_object_ids(scene_graph: dict[str, Any]) -> list[str]:
    """Read post-corruption target lineage for evaluation only (never added to prompts)."""
    objects = list(scene_graph.get("objects", []))
    source_paths = [
        Path(str(obj.get("source_ply_path", "")))
        for obj in objects
        if str(obj.get("source_ply_path", "")).strip()
    ]
    if not source_paths:
        return []
    manifest_path = source_paths[0].parent.parent / "corruption_manifest.json"
    if not manifest_path.exists():
        manifest_path = source_paths[0].parent / "corruption_manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    descendants = payload.get("target_descendant_instance_ids")
    if isinstance(descendants, list) and descendants:
        return [str(value) for value in descendants if str(value).strip()]
    for key in ("target_instance_id_after", "gt_target_instance_id"):
        value = str(payload.get(key, "")).strip()
        if value:
            return [value]
    return []


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def maybe_write_prompt(
    *,
    args: argparse.Namespace,
    split: str,
    sample_id: str,
    sample_uid: str,
    prompt: str,
) -> None:
    if not args.save_prompts:
        return
    debug_dir = args.output_dir / split / "prompts"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{sample_uid.replace('|', '__').replace(':', '_')}.txt"
    path.write_text(prompt, encoding="utf-8")


def run_selector(
    args: argparse.Namespace,
    *,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    gimo_rows = load_csv_rows(args.gimo_dataset_csv)
    metadata_by_sample = load_metadata(args.metadata_csv)
    splits = ["train", "test"] if args.split == "all" else [args.split]
    if llm_client is None:
        llm_client = build_qwen_client(args)

    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []

    for split in splits:
        split_rows = [row for row in gimo_rows if split_name_from_row(row) == split]
        scene_graph_lookup = load_scene_graphs(args.scene_graph_root, split)
        gaze_tensor = load_gaze_tensor(args.slice_root, split)
        if gaze_tensor.shape[0] != len(split_rows):
            raise ValueError(
                f"{split} gaze tensor length {gaze_tensor.shape[0]} does not match dataset rows {len(split_rows)}"
            )

        processed = 0
        completed = 0
        failed = 0
        limit = len(split_rows) if args.limit < 0 else min(len(split_rows), args.limit)
        for split_index, row in enumerate(split_rows[:limit]):
            sample_id = build_sample_id(
                row["scene"],
                row["sequence_path"],
                row["start_frame"],
                row["end_frame"],
            )
            sample_uid = f"{split}:{split_index:04d}:{sample_id}"
            if args.sample_id and sample_id != args.sample_id:
                continue
            processed += 1
            print(f"[{split}] selecting sample_index={split_index} sample_id={sample_id}", flush=True)
            try:
                scene_graph = scene_graph_lookup.get(sample_uid)
                if scene_graph is None:
                    raise KeyError(
                        f"missing scene graph for {sample_uid}"
                    )
                metadata = metadata_by_sample.get(sample_id, {})
                gaze_sequence = gaze_tensor[split_index].mean(dim=1).numpy()
                gaze_annotation, candidate_payloads, _ = build_gaze_candidate_payloads(
                    scene_graph=scene_graph,
                    gaze_sequence=gaze_sequence,
                    args=args,
                )
                if not candidate_payloads:
                    raise ValueError("no valid object candidates after gaze payload construction")

                choice_candidates: list[tuple[str, dict[str, Any]]] = []
                if args.selection_mode == "top1":
                    prompt = build_top1_prompt(
                        scene_graph=scene_graph,
                        metadata=metadata,
                        gaze_annotation=gaze_annotation,
                        candidate_payloads=candidate_payloads,
                    )
                elif args.selection_mode == "soft_logits":
                    prompt, choice_candidates = build_soft_logits_prompt(
                        scene_graph=scene_graph,
                        metadata=metadata,
                        gaze_annotation=gaze_annotation,
                        candidate_payloads=candidate_payloads,
                    )
                else:
                    prompt = build_prompt(
                        scene_graph=scene_graph,
                        metadata=metadata,
                        gaze_annotation=gaze_annotation,
                        candidate_payloads=candidate_payloads,
                    )
                maybe_write_prompt(
                    args=args,
                    split=split,
                    sample_id=sample_id,
                    sample_uid=sample_uid,
                    prompt=prompt,
                )
                if args.selection_mode == "soft_logits":
                    choice_tokens = [choice for choice, _ in choice_candidates]
                    raw_choice_logits = llm_client.choice_logits(prompt, choice_tokens)
                    selected_object, parsed_scores, warnings = parse_soft_logits(
                        choice_candidates=choice_candidates,
                        choice_logits=raw_choice_logits,
                        temperature=args.softmax_temperature,
                    )
                    response_text = json.dumps(
                        {
                            "choice_logits": raw_choice_logits,
                            "softmax_temperature": args.softmax_temperature,
                        },
                        ensure_ascii=False,
                    )
                else:
                    response_text = llm_client.complete(prompt)
                if args.selection_mode == "top1":
                    selected_object, parsed_scores, warnings = parse_top1_response(
                        sample_id=sample_id,
                        response_text=response_text,
                        candidate_payloads=candidate_payloads,
                    )
                elif args.selection_mode == "full_scores":
                    selected_object, parsed_scores, warnings = parse_llm_response(
                        sample_id=sample_id,
                        response_text=response_text,
                        candidate_payloads=candidate_payloads,
                    )

                final_row = parsed_scores[0]
                accepted_gt_ids = noisy_gt_object_ids(scene_graph)
                result_rows.append(
                    {
                        "split": split,
                        "split_index": split_index,
                        "sample_id": sample_id,
                        "sample_uid": sample_uid,
                        "selected_object_id": selected_object,
                        "response_text": response_text,
                        "warnings": warnings,
                        "metadata": metadata,
                        "gaze_annotation": gaze_annotation,
                        "candidate_scores": parsed_scores,
                    }
                )
                summary_rows.append(
                    {
                        "status": "ok",
                        "split": split,
                        "split_index": split_index,
                        "sample_id": sample_id,
                        "sample_uid": sample_uid,
                        "scene_name": scene_graph.get("scene_name", ""),
                        "sequence_name": scene_graph.get("sequence_name", ""),
                        "start_frame": scene_graph.get("start_frame", ""),
                        "end_frame": scene_graph.get("end_frame", ""),
                        "action_norm": metadata.get("action_norm", ""),
                        "object_norm": metadata.get("object_norm", ""),
                        "want_text": metadata.get("want_text", ""),
                        "num_candidates": len(candidate_payloads),
                        "selected_object_id": selected_object,
                        "selected_label": final_row.get("label", ""),
                        "selected_match": final_row["match"],
                        "selected_ease": final_row["ease"],
                        "selected_total": final_row["semantic_total"],
                        "selected_posterior": final_row["posterior"],
                        "accepted_gt_object_ids": json.dumps(
                            accepted_gt_ids, ensure_ascii=False
                        ),
                        "warnings": json.dumps(warnings, ensure_ascii=False),
                    }
                )
                for item in parsed_scores:
                    candidate_rows.append(
                        {
                            "split": split,
                            "split_index": split_index,
                            "sample_id": sample_id,
                            "sample_uid": sample_uid,
                            "want": metadata.get("want_text", ""),
                            "object_id": item["object_id"],
                            "name": item.get("label", ""),
                            "category": item.get("category", item.get("label", "")),
                            "semantic_label_id": item.get("semantic_label_id", ""),
                            "choice_token": item.get("choice_token", ""),
                            "match": item["match"],
                            "ease": item["ease"],
                            "semantic_total": item["semantic_total"],
                            "teacher_logit": item.get(
                                "teacher_logit", item["semantic_total"]
                            ),
                            "masked_fusion_total": item["semantic_total"],
                            "posterior": item["posterior"],
                            "reasoning_trace": item["reasoning_trace"],
                            "llm_rank": item["rank"],
                            "gaze_prompt_score": item["gaze_prompt_score"],
                            "bbox_hit_ratio": item["bbox_hit_ratio"],
                            "last_bbox_hit": item.get("last_bbox_hit", ""),
                            "mean_bbox_distance": item["mean_bbox_distance"],
                            "last_bbox_distance": item.get("last_bbox_distance", ""),
                            "mean_centroid_distance": item["mean_centroid_distance"],
                            "last_centroid_distance": item.get(
                                "last_centroid_distance", ""
                            ),
                            "distance": item.get("distance", ""),
                            "head_centroid_distance_3d": item.get(
                                "head_centroid_distance_3d", ""
                            ),
                            "root_alignment": item.get("root_alignment", ""),
                            "torso_alignment": item.get("torso_alignment", ""),
                            "face_normal_alignment": item.get(
                                "face_normal_alignment", ""
                            ),
                            "root_torso_alignment": item.get(
                                "root_torso_alignment", ""
                            ),
                            "bbox_size_x": item.get("bbox_size_x", ""),
                            "bbox_size_y": item.get("bbox_size_y", ""),
                            "bbox_size_z": item.get("bbox_size_z", ""),
                            "bbox_volume": item.get("bbox_volume", ""),
                            "is_final": int(item["object_id"] == selected_object),
                            "is_gt": int(item["object_id"] in accepted_gt_ids),
                            "accepted_gt_object_ids": json.dumps(
                                accepted_gt_ids, ensure_ascii=False
                            ),
                        }
                    )
                completed += 1
                if args.selection_mode == "top1":
                    print(
                        f"[{split}] selected={selected_object} sample_id={sample_id}",
                        flush=True,
                    )
                elif args.selection_mode == "soft_logits":
                    print(
                        f"[{split}] selected={selected_object} "
                        f"posterior={float(final_row['posterior']):.4f} "
                        f"sample_id={sample_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{split}] selected={selected_object} total={final_row['semantic_total']:.2f} "
                        f"sample_id={sample_id}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                error_rows.append(
                    {
                        "split": split,
                        "split_index": split_index,
                        "sample_id": sample_id,
                        "sample_uid": sample_uid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                summary_rows.append(
                    {
                        "status": "failed",
                        "split": split,
                        "split_index": split_index,
                        "sample_id": sample_id,
                        "sample_uid": sample_uid,
                        "scene_name": row.get("scene", ""),
                        "sequence_name": row.get("sequence_path", ""),
                        "start_frame": row.get("start_frame", ""),
                        "end_frame": row.get("end_frame", ""),
                        "action_norm": "",
                        "object_norm": "",
                        "want_text": "",
                        "num_candidates": 0,
                        "selected_object_id": "",
                        "selected_label": "",
                        "selected_match": "",
                        "selected_ease": "",
                        "selected_total": "",
                        "selected_posterior": "",
                        "accepted_gt_object_ids": "",
                        "warnings": str(exc),
                    }
                )
                print(f"[{split}] failed sample_id={sample_id}: {type(exc).__name__}: {exc}", flush=True)

        split_summaries.append(
            {
                "split": split,
                "processed": processed,
                "completed": completed,
                "failed": failed,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "teacher_selection_results.jsonl"
    with result_path.open("w", encoding="utf-8") as file:
        for row in result_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_fields = [
        "status",
        "split",
        "split_index",
        "sample_id",
        "sample_uid",
        "scene_name",
        "sequence_name",
        "start_frame",
        "end_frame",
        "action_norm",
        "object_norm",
        "want_text",
        "num_candidates",
        "selected_object_id",
        "selected_label",
        "selected_match",
        "selected_ease",
        "selected_total",
        "selected_posterior",
        "accepted_gt_object_ids",
        "warnings",
    ]
    candidate_fields = [
        "split",
        "split_index",
        "sample_id",
        "sample_uid",
        "want",
        "object_id",
        "name",
        "category",
        "semantic_label_id",
        "choice_token",
        "match",
        "ease",
        "semantic_total",
        "teacher_logit",
        "masked_fusion_total",
        "posterior",
        "reasoning_trace",
        "llm_rank",
        "gaze_prompt_score",
        "bbox_hit_ratio",
        "last_bbox_hit",
        "mean_bbox_distance",
        "last_bbox_distance",
        "mean_centroid_distance",
        "last_centroid_distance",
        "distance",
        "head_centroid_distance_3d",
        "root_alignment",
        "torso_alignment",
        "face_normal_alignment",
        "root_torso_alignment",
        "bbox_size_x",
        "bbox_size_y",
        "bbox_size_z",
        "bbox_volume",
        "is_final",
        "is_gt",
        "accepted_gt_object_ids",
    ]
    error_fields = [
        "split",
        "split_index",
        "sample_id",
        "sample_uid",
        "error_type",
        "error",
    ]
    write_csv(args.output_dir / "teacher_selection_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "teacher_candidate_scores.csv", candidate_rows, candidate_fields)
    write_csv(args.output_dir / "teacher_selection_failures.csv", error_rows, error_fields)

    summary = {
        "output_dir": str(args.output_dir),
        "result_path": str(result_path),
        "summary_csv": str(args.output_dir / "teacher_selection_summary.csv"),
        "candidate_csv": str(args.output_dir / "teacher_candidate_scores.csv"),
        "failure_csv": str(args.output_dir / "teacher_selection_failures.csv"),
        "backend": args.backend,
        "model_name": args.model_name if args.backend == "qwen" else "mock",
        "selection_mode": args.selection_mode,
        "softmax_temperature": args.softmax_temperature,
        "split": args.split,
        "limit": args.limit,
        "sample_id": args.sample_id,
        "last_k_frames": args.last_k_frames,
        "temporal_decay": args.temporal_decay,
        "bbox_margin": args.bbox_margin,
        "point_distance_scale": args.point_distance_scale,
        "centroid_distance_scale": args.centroid_distance_scale,
        "bbox_hit_weight": args.bbox_hit_weight,
        "point_score_weight": args.point_score_weight,
        "centroid_score_weight": args.centroid_score_weight,
        "splits": split_summaries,
    }
    write_json(args.output_dir / "run_summary.json", summary)
    return summary


def build_points(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = REPO_ROOT / "dataset"
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))
    from create_final_object_points_from_selector import (
        build_scene_instance_index,
        build_selector_lookup,
        build_split,
        load_csv_rows as load_selector_csv_rows,
    )

    output_name = args.object_output_name
    selector_summary_csv = args.output_dir / "teacher_selection_summary.csv"
    if not output_name.endswith(".pth"):
        raise ValueError(f"--object-output-name must end with .pth, got: {output_name}")

    gimo_rows = load_selector_csv_rows(
        args.gimo_dataset_csv,
        required=("scene", "sequence_path", "start_frame", "end_frame", "training", "transformation"),
    )
    selector_rows = load_selector_csv_rows(
        selector_summary_csv,
        required=("sample_id", "selected_object_id"),
    )
    selector_lookup, duplicate_conflicts = build_selector_lookup(selector_rows)
    scene_index = build_scene_instance_index(args.openins3d_root)

    for split_name in ("train", "test"):
        build_split(
            split_name=split_name,
            gimo_rows=gimo_rows,
            selector_lookup=selector_lookup,
            scene_index=scene_index,
            output_root=args.output_root,
            output_name=output_name,
            gimo_root=args.gimo_root,
            gimo_dataset_csv=args.gimo_dataset_csv,
            selector_summary_csv=selector_summary_csv,
            openins3d_root=args.openins3d_root,
            num_object_points=args.num_object_points,
            duplicate_conflicts=duplicate_conflicts,
        )

    build_summary = {
        "object_output_name": output_name,
        "output_root": str(args.output_root),
        "selector_summary_csv": str(selector_summary_csv),
        "openins3d_root": str(args.openins3d_root),
        "num_object_points": args.num_object_points,
        "selector_run_summary": summary,
    }
    write_json(args.output_dir / "point_build_summary.json", build_summary)
    return build_summary


def main() -> None:
    args = parse_args()
    summary = run_selector(args)
    if args.build_points:
        build_points(args, summary)
        print(
            f"built points -> {args.output_root / 'train' / args.object_output_name} "
            f"and {args.output_root / 'test' / args.object_output_name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
