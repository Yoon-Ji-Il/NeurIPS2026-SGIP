import argparse
import json
from pathlib import Path

import numpy as np
import torch

from create_final_object_points_from_selector import (
    apply_scene_normalization,
    build_scene_instance_index,
    load_csv_rows,
    load_raw_points,
    load_transform_context,
    norm_value,
    normalize_instance_id,
    sample_key,
    sample_object_points,
    stable_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build SIF3D object point tensors by posterior-weighting top-k teacher "
            "candidate object point clouds into one [P, 3] tensor per sample."
        )
    )
    parser.add_argument(
        "--teacher-candidate-scores-csv",
        default=(
            "./object_selector/teacher_targets/"
            "qwen3_14b_sequence_selector_gpu1_5/motion_c_fusion_score_final/"
            "teacher_candidate_scores.csv"
        ),
        help="Teacher candidate score CSV containing sample_id, object_id, and posterior columns.",
    )
    parser.add_argument(
        "--gimo-dataset-csv",
        default="data/GIMO/dataset.csv",
        help="Canonical GIMO dataset.csv used to define train/test sample order.",
    )
    parser.add_argument(
        "--gimo-root",
        default="data/GIMO",
        help="GIMO root containing scene transform files.",
    )
    parser.add_argument(
        "--openins3d-root",
        default="data/OpenIns3D/output/results_demo",
        help="Root containing scene label_pointcloud_export.json files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/GIMO/SLICES_888s",
        help="SLICES_888s root. The script writes into train/ and test/.",
    )
    parser.add_argument("--num-object-points", default=1024, type=int)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument(
        "--weight-column",
        default="posterior",
        help="Candidate score column used as the non-negative fusion weight.",
    )
    parser.add_argument(
        "--score-column",
        default="masked_fusion_total",
        help="Fallback/sort score column used when posterior weights are zero.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output .pth filename under each split directory. Defaults to "
            "final_object_teacher_top3_points_<num-object-points>.pth"
        ),
    )
    return parser.parse_args()


def parse_sample_id(sample_id):
    parts = norm_value(sample_id).split("|")
    if len(parts) != 4:
        raise ValueError(f"Expected sample_id as scene|sequence|start|end, got: {sample_id!r}")
    return tuple(norm_value(part) for part in parts)


def safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def build_candidate_lookup(candidate_rows):
    lookup = {}
    duplicate_objects = []

    for row in candidate_rows:
        key = parse_sample_id(row["sample_id"])
        object_id = normalize_instance_id(row.get("object_id", ""))
        if not object_id:
            continue

        normalized = dict(row)
        normalized["object_id"] = object_id
        sample_candidates = lookup.setdefault(key, {})
        if object_id in sample_candidates:
            duplicate_objects.append({"sample_key": list(key), "object_id": object_id})
            continue
        sample_candidates[object_id] = normalized

    return {key: list(value.values()) for key, value in lookup.items()}, duplicate_objects


def select_topk_with_weights(rows, top_k, weight_column, score_column, eps=1e-12):
    scored = []
    for row in rows:
        weight_value = max(0.0, safe_float(row.get(weight_column), 0.0))
        score_value = safe_float(row.get(score_column), -1e9)
        scored.append((weight_value, score_value, row))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = scored[:top_k]
    raw_weights = np.asarray([item[0] for item in selected], dtype=np.float64)

    if raw_weights.sum() <= eps:
        fallback_scores = np.asarray([item[1] for item in selected], dtype=np.float64)
        finite_mask = np.isfinite(fallback_scores)
        if finite_mask.any():
            fallback_scores = np.where(finite_mask, fallback_scores, fallback_scores[finite_mask].min())
            fallback_scores = fallback_scores - fallback_scores.max()
            raw_weights = np.exp(np.clip(fallback_scores, -50.0, 50.0))
        else:
            raw_weights = np.ones((len(selected),), dtype=np.float64)

    if raw_weights.sum() <= eps:
        weights = np.ones((len(selected),), dtype=np.float64) / max(1, len(selected))
    else:
        weights = raw_weights / raw_weights.sum()

    return [
        {
            "row": row,
            "object_id": row["object_id"],
            "object_name": row.get("name", ""),
            "raw_weight": float(raw_weight),
            "normalized_weight": float(weight),
            "score": float(score),
        }
        for (raw_weight, score, row), weight in zip(selected, weights)
    ]


def build_weighted_points_for_sample(
    key,
    row,
    selected,
    scene_index,
    gimo_root,
    num_object_points,
    raw_points_cache,
):
    scene_name = key[0]
    scene_instances = scene_index.get(scene_name)
    if scene_instances is None:
        raise KeyError(f"Missing scene json for {scene_name}")

    fused_points = np.zeros((num_object_points, 3), dtype=np.float32)
    component_manifest = []

    for item in selected:
        weight = item["normalized_weight"]
        object_id = item["object_id"]
        instance_info = scene_instances.get(object_id)
        if instance_info is None:
            if weight <= 0.0:
                component_manifest.append({**item, "status": "skipped_zero_weight_missing_instance"})
                continue
            raise KeyError(f"Missing selected instance {object_id} in scene json for {scene_name}")

        ply_path = Path(instance_info["ply_path"])
        if ply_path not in raw_points_cache:
            raw_points_cache[ply_path] = load_raw_points(ply_path)

        scale, transform_norm = load_transform_context(gimo_root, row)
        aligned_points = apply_scene_normalization(raw_points_cache[ply_path], scale, transform_norm)
        sampled_points, used_replacement = sample_object_points(
            aligned_points,
            num_object_points,
            stable_seed(key, object_id),
        )
        fused_points += (sampled_points * np.float32(weight)).astype(np.float32)

        component_manifest.append(
            {
                **item,
                "status": "ok",
                "object_ply_path": str(ply_path),
                "object_name_from_json": instance_info.get("semantic_label_name", ""),
                "object_num_points_from_json": instance_info.get("num_points", 0),
                "raw_point_count": int(aligned_points.shape[0]),
                "stored_point_count": int(num_object_points),
                "used_replacement": bool(used_replacement),
            }
        )

    return fused_points, component_manifest


def build_split(
    split_name,
    gimo_rows,
    candidate_lookup,
    scene_index,
    output_root,
    output_name,
    gimo_root,
    gimo_dataset_csv,
    teacher_candidate_scores_csv,
    openins3d_root,
    num_object_points,
    top_k,
    weight_column,
    score_column,
    duplicate_objects,
):
    split_value = "1" if split_name == "train" else "0"
    split_rows = [row for row in gimo_rows if norm_value(row["training"]) == split_value]
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    tensor = torch.zeros((len(split_rows), num_object_points, 3), dtype=torch.float32)
    mask = torch.zeros((len(split_rows),), dtype=torch.bool)
    manifest = []
    missing = []
    raw_points_cache = {}

    for index, row in enumerate(split_rows):
        key = sample_key(row)
        selected = []
        manifest_entry = {
            "index": index,
            "split": split_name,
            "scene_name": key[0],
            "sequence_name": key[1],
            "start_frame": key[2],
            "end_frame": key[3],
            "training": norm_value(row["training"]),
            "transformation": norm_value(row["transformation"]),
            "sample_key": list(key),
        }

        try:
            candidates = candidate_lookup.get(key)
            if not candidates:
                raise KeyError("missing_candidate_scores")

            selected = select_topk_with_weights(candidates, top_k, weight_column, score_column)
            if not selected:
                raise KeyError("empty_topk_selection")

            fused_points, components = build_weighted_points_for_sample(
                key=key,
                row=row,
                selected=selected,
                scene_index=scene_index,
                gimo_root=gimo_root,
                num_object_points=num_object_points,
                raw_points_cache=raw_points_cache,
            )
            tensor[index] = torch.from_numpy(fused_points)
            mask[index] = True
            manifest_entry["status"] = "ok"
            manifest_entry["topk_components"] = components
            manifest_entry["topk_object_ids"] = [item["object_id"] for item in selected]
            manifest_entry["topk_weights"] = [item["normalized_weight"] for item in selected]
        except Exception as exc:
            manifest_entry["status"] = f"top3_points_build_failed: {exc}"
            manifest_entry["topk_components"] = selected
            missing.append(manifest_entry)

        manifest.append(manifest_entry)

    output_path = split_dir / output_name
    mask_path = split_dir / output_name.replace(".pth", "_mask.pth")
    meta_path = split_dir / output_name.replace(".pth", "_meta.json")
    missing_path = split_dir / output_name.replace(".pth", "_missing.json")
    manifest_path = split_dir / output_name.replace(".pth", "_manifest.json")

    torch.save(tensor, output_path)
    torch.save(mask, mask_path)

    meta = {
        "split": split_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "valid_count": int(mask.sum().item()),
        "missing_count": len(missing),
        "num_object_points": num_object_points,
        "top_k": top_k,
        "weight_column": weight_column,
        "score_column": score_column,
        "fusion_mode": "pointwise_weighted_average_after_independent_object_sampling",
        "output_path": str(output_path),
        "mask_path": str(mask_path),
        "gimo_dataset_csv": str(gimo_dataset_csv),
        "teacher_candidate_scores_csv": str(teacher_candidate_scores_csv),
        "gimo_root": str(gimo_root),
        "openins3d_root": str(openins3d_root),
        "duplicate_candidate_object_count": len(duplicate_objects),
        "duplicate_candidate_objects": duplicate_objects,
        "todo": (
            "Point-wise averaging is a quick compatibility test. A future model-side "
            "variant should encode each object independently and weight object embeddings."
        ),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with missing_path.open("w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[{split_name}] wrote {output_path}")
    print(f"[{split_name}] valid={meta['valid_count']} missing={meta['missing_count']} shape={meta['shape']}")


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be positive, got: {args.top_k}")

    output_name = args.output_name or f"final_object_teacher_top3_points_{args.num_object_points}.pth"
    if not output_name.endswith(".pth"):
        raise ValueError(f"--output-name must end with .pth, got: {output_name}")
    if "_top3" not in output_name:
        raise ValueError("Use an output name containing '_top3' to avoid mixing hard top-1 and top-3 tensors.")

    gimo_dataset_csv = Path(args.gimo_dataset_csv)
    teacher_candidate_scores_csv = Path(args.teacher_candidate_scores_csv)
    gimo_root = Path(args.gimo_root)
    openins3d_root = Path(args.openins3d_root)
    output_root = Path(args.output_root)

    gimo_rows = load_csv_rows(
        gimo_dataset_csv,
        required=("scene", "sequence_path", "start_frame", "end_frame", "training", "transformation"),
    )
    candidate_rows = load_csv_rows(
        teacher_candidate_scores_csv,
        required=("sample_id", "object_id", args.weight_column, args.score_column),
    )

    candidate_lookup, duplicate_objects = build_candidate_lookup(candidate_rows)
    scene_index = build_scene_instance_index(openins3d_root)

    for split_name in ("train", "test"):
        build_split(
            split_name=split_name,
            gimo_rows=gimo_rows,
            candidate_lookup=candidate_lookup,
            scene_index=scene_index,
            output_root=output_root,
            output_name=output_name,
            gimo_root=gimo_root,
            gimo_dataset_csv=gimo_dataset_csv,
            teacher_candidate_scores_csv=teacher_candidate_scores_csv,
            openins3d_root=openins3d_root,
            num_object_points=args.num_object_points,
            top_k=args.top_k,
            weight_column=args.weight_column,
            score_column=args.score_column,
            duplicate_objects=duplicate_objects,
        )


if __name__ == "__main__":
    main()
