from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIMO_ROOT = Path("data/GIMO")
DEFAULT_OUTPUT_ROOT = DEFAULT_GIMO_ROOT / "SLICES_888s"
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "object_selector" / "gaze_runs" / "gaze_final_nearest_object_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-sequence object priors by selecting the object nearest to the final "
            "gaze point in each sequence, without modifying existing training code."
        )
    )
    parser.add_argument("--gimo-root", type=Path, default=DEFAULT_GIMO_ROOT)
    parser.add_argument("--gimo-dataset-csv", type=Path, default=DEFAULT_GIMO_ROOT / "dataset.csv")
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-object-points", type=int, default=1024)
    parser.add_argument("--bbox-margin", type=float, default=0.0)
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Output .pth filename written under each split directory. Defaults to "
            "final_object_gaze_final_nearest_points_<num-object-points>.pth"
        ),
    )
    return parser.parse_args()


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


def stable_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.md5(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


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
            sample_id = str(graph.get("sample_id", "")).strip()
            if sample_id:
                lookup[sample_id] = graph
    return lookup


def load_gaze_tensor(slice_root: Path, split: str) -> torch.Tensor:
    path = slice_root / split / "gazes.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing gaze tensor: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)!r}")
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected gaze tensor [N, T, G, 3], got {tuple(tensor.shape)}")
    return tensor.detach().cpu().float()


def load_transform_context(gimo_root: Path, row: dict[str, str]) -> tuple[float, np.ndarray]:
    scene_name = norm_value(row["scene"])
    sequence_name = norm_value(row["sequence_path"])
    transform_name = norm_value(row["transformation"]) or "transform_info.json"

    transform_info_path = gimo_root / scene_name / sequence_name / transform_name
    if not transform_info_path.exists():
        raise FileNotFoundError(f"Missing transform info: {transform_info_path}")
    transform_info = json.loads(transform_info_path.read_text(encoding="utf-8"))

    scale = float(transform_info["scale"])
    transform_norm_path = gimo_root / scene_name / "scene_obj" / "transform_norm.txt"
    transform_norm = np.loadtxt(transform_norm_path, dtype=np.float32).reshape(4, 4)
    transform_norm[:3, 3] /= scale
    return scale, transform_norm


def apply_scene_normalization(points: np.ndarray, scale: float, transform_norm: np.ndarray) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float32) * (1.0 / scale)
    return (transform_norm[:3, :3] @ scaled.T + transform_norm[:3, 3:]).T.astype(np.float32)


def load_raw_points(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise ValueError(f"No vertices attribute in {path}")
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty point cloud with shape (N, 3), got {points.shape} from {path}")
    return points


def sample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("Cannot sample from an empty point cloud")
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < count
    indices = rng.choice(points.shape[0], size=count, replace=replace)
    return points[indices].astype(np.float32, copy=False)


def bbox_contains(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float) -> bool:
    lower = bbox_min - float(margin)
    upper = bbox_max + float(margin)
    return bool(np.all(point >= lower) and np.all(point <= upper))


def bbox_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    clamped = np.minimum(np.maximum(point, bbox_min), bbox_max)
    return float(np.linalg.norm(point - clamped))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty csv: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def build_candidate_row(
    *,
    split: str,
    split_index: int,
    sample_id: str,
    object_row: dict[str, Any],
    final_gaze_point: np.ndarray,
    is_inside_bbox: bool,
    candidate_bbox_distance: float,
    candidate_centroid_distance: float,
) -> dict[str, Any]:
    metrics = object_row.get("metrics", {})
    return {
        "split": split,
        "split_index": int(split_index),
        "sample_id": sample_id,
        "object_id": object_row.get("object_id", ""),
        "label": object_row.get("label", ""),
        "semantic_label_id": object_row.get("semantic_label_id", ""),
        "is_inside_bbox": int(is_inside_bbox),
        "bbox_distance": float(candidate_bbox_distance),
        "centroid_distance": float(candidate_centroid_distance),
        "rank": 0,
        "is_selected": 0,
        "final_gaze_point": json.dumps([float(value) for value in final_gaze_point.tolist()]),
        "root_distance": metrics.get("distance"),
        "root_alignment": metrics.get("root_alignment"),
        "face_normal_alignment": metrics.get("face_normal_alignment"),
    }


def process_split(
    *,
    split: str,
    rows: list[dict[str, str]],
    gaze_tensor: torch.Tensor,
    scene_graph_lookup: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if gaze_tensor.shape[0] != len(rows):
        raise ValueError(
            f"{split} gaze tensor length {gaze_tensor.shape[0]} does not match dataset rows {len(rows)}"
        )

    output_name = args.output_name or f"final_object_gaze_final_nearest_points_{args.num_object_points}.pth"
    mask_name = output_name.replace(".pth", "_mask.pth")
    split_dir = args.output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    limit = len(rows) if args.limit < 0 else min(len(rows), args.limit)
    point_tensor = torch.zeros((limit, args.num_object_points, 3), dtype=torch.float32)
    mask_tensor = torch.zeros((limit,), dtype=torch.bool)

    raw_points_cache: dict[Path, np.ndarray] = {}
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    missing_scene_graph: list[dict[str, Any]] = []
    missing_objects: list[dict[str, Any]] = []
    sample_index_mismatches: list[dict[str, Any]] = []

    processed_rows = rows[:limit]
    for split_index, row in enumerate(processed_rows):
        sample_id = build_sample_id(
            row["scene"],
            row["sequence_path"],
            row["start_frame"],
            row["end_frame"],
        )
        graph = scene_graph_lookup.get(sample_id)
        if graph is None:
            missing_scene_graph.append(
                {
                    "split": split,
                    "split_index": split_index,
                    "sample_id": sample_id,
                    "reason": "missing_scene_graph",
                }
            )
            continue

        scene_graph_index = int(graph.get("sample_index", split_index))
        if scene_graph_index != split_index:
            sample_index_mismatches.append(
                {
                    "split": split,
                    "split_index": split_index,
                    "scene_graph_sample_index": scene_graph_index,
                    "sample_id": sample_id,
                }
            )

        gaze_frames = gaze_tensor[split_index].mean(dim=1).numpy()
        final_gaze_point = np.asarray(gaze_frames[-1], dtype=np.float32)

        scene_name = norm_value(row["scene"])
        scale, transform_norm = load_transform_context(args.gimo_root, row)

        best_payload: dict[str, Any] | None = None
        object_candidates = list(graph.get("objects", []))

        for object_row in object_candidates:
            object_id = str(object_row.get("object_id", "")).strip()
            source_ply_path = str(object_row.get("source_ply_path", "")).strip()
            if not object_id or not source_ply_path:
                continue

            centroid = np.asarray(object_row.get("centroid", []), dtype=np.float32)
            bbox_min = np.asarray(object_row.get("bbox_min", []), dtype=np.float32)
            bbox_max = np.asarray(object_row.get("bbox_max", []), dtype=np.float32)
            if centroid.shape != (3,) or bbox_min.shape != (3,) or bbox_max.shape != (3,):
                continue

            is_inside = bbox_contains(final_gaze_point, bbox_min, bbox_max, args.bbox_margin)
            candidate_bbox_distance = bbox_distance(final_gaze_point, bbox_min, bbox_max)
            candidate_centroid_distance = float(np.linalg.norm(final_gaze_point - centroid))

            candidate_row = build_candidate_row(
                split=split,
                split_index=split_index,
                sample_id=sample_id,
                object_row=object_row,
                final_gaze_point=final_gaze_point,
                is_inside_bbox=is_inside,
                candidate_bbox_distance=candidate_bbox_distance,
                candidate_centroid_distance=candidate_centroid_distance,
            )
            candidate_rows.append(candidate_row)

            candidate_payload = {
                "candidate_row": candidate_row,
                "source_ply_path": source_ply_path,
                "object_id": object_id,
                "label": object_row.get("label", ""),
                "is_inside_bbox": int(is_inside),
                "bbox_distance": float(candidate_bbox_distance),
                "centroid_distance": float(candidate_centroid_distance),
            }
            if best_payload is None:
                best_payload = candidate_payload
                continue

            best_key = (
                -int(best_payload["is_inside_bbox"]),
                float(best_payload["bbox_distance"]),
                float(best_payload["centroid_distance"]),
                str(best_payload["object_id"]),
            )
            candidate_key = (
                -int(candidate_payload["is_inside_bbox"]),
                float(candidate_payload["bbox_distance"]),
                float(candidate_payload["centroid_distance"]),
                str(candidate_payload["object_id"]),
            )
            if candidate_key < best_key:
                best_payload = candidate_payload

        sample_summary = {
            "split": split,
            "split_index": split_index,
            "sample_id": sample_id,
            "scene_name": scene_name,
            "sequence_name": norm_value(row["sequence_path"]),
            "start_frame": norm_value(row["start_frame"]),
            "end_frame": norm_value(row["end_frame"]),
            "num_candidates": len(object_candidates),
            "final_gaze_point": json.dumps([float(value) for value in final_gaze_point.tolist()]),
            "selected_object_id": "",
            "selected_label": "",
            "selected_is_inside_bbox": "",
            "selected_bbox_distance": "",
            "selected_centroid_distance": "",
            "status": "no_candidate_selected",
        }

        if best_payload is None:
            missing_objects.append(
                {
                    "split": split,
                    "split_index": split_index,
                    "sample_id": sample_id,
                    "reason": "no_valid_object_candidates",
                }
            )
            sample_rows.append(sample_summary)
            continue

        best_ply_path = Path(best_payload["source_ply_path"])
        if best_ply_path not in raw_points_cache:
            raw_points_cache[best_ply_path] = load_raw_points(best_ply_path)
        aligned_points = apply_scene_normalization(raw_points_cache[best_ply_path], scale, transform_norm)
        stored_points = sample_points(
            aligned_points,
            args.num_object_points,
            stable_seed(split, sample_id, best_payload["object_id"], "store"),
        )
        point_tensor[split_index] = torch.from_numpy(stored_points)
        mask_tensor[split_index] = True

        sample_summary["selected_object_id"] = best_payload["object_id"]
        sample_summary["selected_label"] = best_payload["label"]
        sample_summary["selected_is_inside_bbox"] = int(best_payload["is_inside_bbox"])
        sample_summary["selected_bbox_distance"] = float(best_payload["bbox_distance"])
        sample_summary["selected_centroid_distance"] = float(best_payload["centroid_distance"])
        sample_summary["status"] = "ok"
        sample_rows.append(sample_summary)

    selected_by_key = {
        (row["split"], row["split_index"], row["sample_id"]): row["selected_object_id"]
        for row in sample_rows
        if row["status"] == "ok"
    }
    grouped_rank: dict[tuple[str, int, str], int] = {}
    candidate_rows.sort(
        key=lambda row: (
            row["split"],
            int(row["split_index"]),
            row["sample_id"],
            -int(row["is_inside_bbox"]),
            float(row["bbox_distance"]),
            float(row["centroid_distance"]),
            row["object_id"],
        )
    )
    for row in candidate_rows:
        key = (row["split"], int(row["split_index"]), row["sample_id"])
        grouped_rank[key] = grouped_rank.get(key, 0) + 1
        row["rank"] = grouped_rank[key]
        if selected_by_key.get(key, "") == row["object_id"]:
            row["is_selected"] = 1

    point_path = split_dir / output_name
    mask_path = split_dir / mask_name
    torch.save(point_tensor, point_path)
    torch.save(mask_tensor, mask_path)

    analysis_split_dir = args.analysis_dir / split
    analysis_split_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        analysis_split_dir / "candidate_scores.csv",
        candidate_rows
        or [
            {
                "split": split,
                "split_index": 0,
                "sample_id": "",
                "object_id": "",
                "label": "",
                "semantic_label_id": "",
                "is_inside_bbox": "",
                "bbox_distance": "",
                "centroid_distance": "",
                "rank": "",
                "is_selected": "",
                "final_gaze_point": "",
                "root_distance": "",
                "root_alignment": "",
                "face_normal_alignment": "",
            }
        ],
    )
    write_csv(
        analysis_split_dir / "sample_summary.csv",
        sample_rows
        or [
            {
                "split": split,
                "split_index": 0,
                "sample_id": "",
                "scene_name": "",
                "sequence_name": "",
                "start_frame": "",
                "end_frame": "",
                "num_candidates": "",
                "final_gaze_point": "",
                "selected_object_id": "",
                "selected_label": "",
                "selected_is_inside_bbox": "",
                "selected_bbox_distance": "",
                "selected_centroid_distance": "",
                "status": "",
            }
        ],
    )

    summary = {
        "split": split,
        "processed_samples": limit,
        "saved_tensor_shape": list(point_tensor.shape),
        "saved_mask_true_count": int(mask_tensor.sum().item()),
        "output_points_path": str(point_path),
        "output_mask_path": str(mask_path),
        "candidate_scores_csv": str(analysis_split_dir / "candidate_scores.csv"),
        "sample_summary_csv": str(analysis_split_dir / "sample_summary.csv"),
        "missing_scene_graph_count": len(missing_scene_graph),
        "missing_object_count": len(missing_objects),
        "sample_index_mismatch_count": len(sample_index_mismatches),
        "bbox_margin": float(args.bbox_margin),
        "num_object_points": int(args.num_object_points),
        "selection_rule": (
            "Pick the object nearest to the final gaze point using this priority: "
            "inside bbox first, then smaller bbox distance, then smaller centroid distance."
        ),
        "missing_scene_graph_samples": missing_scene_graph,
        "missing_object_samples": missing_objects,
        "sample_index_mismatches": sample_index_mismatches,
    }
    write_json(analysis_split_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    dataset_rows = load_csv_rows(args.gimo_dataset_csv)
    split_order = ["train", "test"] if args.split == "all" else [args.split]

    all_summaries: list[dict[str, Any]] = []
    for split in split_order:
        split_rows = [row for row in dataset_rows if split_name_from_row(row) == split]
        scene_graph_lookup = load_scene_graphs(args.scene_graph_root, split)
        gaze_tensor = load_gaze_tensor(args.slice_root, split)
        summary = process_split(
            split=split,
            rows=split_rows,
            gaze_tensor=gaze_tensor,
            scene_graph_lookup=scene_graph_lookup,
            args=args,
        )
        all_summaries.append(summary)

    merged = {
        "splits": all_summaries,
        "analysis_dir": str(args.analysis_dir),
        "scene_graph_root": str(args.scene_graph_root),
        "gimo_root": str(args.gimo_root),
        "gimo_dataset_csv": str(args.gimo_dataset_csv),
        "slice_root": str(args.slice_root),
        "output_root": str(args.output_root),
    }
    write_json(args.analysis_dir / "summary.json", merged)


if __name__ == "__main__":
    main()
