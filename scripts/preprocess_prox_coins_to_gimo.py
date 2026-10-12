#!/usr/bin/env python
"""Convert PROX/COINS interaction semantics into SIF3D/GIMO-style tensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer
from plyfile import PlyData, PlyElement


INPUT_LEN = 6
OUTPUT_LEN = 10
TOTAL_LEN = INPUT_LEN + OUTPUT_LEN
FLOOR_WALL = {"floor", "wall"}


@dataclass(frozen=True)
class InstanceInfo:
    object_idx: int
    vertex_instance_id: int
    semantic_label_name: str
    declared_num_points: int


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha1("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def sample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((count, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < count
    idx = rng.choice(points.shape[0], size=count, replace=replace)
    return points[idx].astype(np.float32, copy=False)


def read_ply_vertex_count(scene_path: Path) -> int:
    with scene_path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"Could not read vertex count from {scene_path}")


def load_scene_vertices(scene_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(scene_path))
    vertices = ply["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    color_names = {"red", "green", "blue"}
    if color_names.issubset(vertices.dtype.names or ()):
        rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = None
    return xyz, rgb


def write_point_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        vertex = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    else:
        vertex = np.empty(
            xyz.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def parse_instance_txt(path: Path) -> dict[int, InstanceInfo]:
    pattern = re.compile(r"^(\d+):\s+(.+)_([0-9]+)\s+of\s+(\d+)\s+vertices$")
    instances: dict[int, InstanceInfo] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match is None:
            raise ValueError(f"Could not parse instance line in {path}: {line!r}")
        object_idx = int(match.group(1))
        semantic_label_name = match.group(2)
        declared_num_points = int(match.group(4))
        instances[object_idx] = InstanceInfo(
            object_idx=object_idx,
            vertex_instance_id=object_idx + 1,
            semantic_label_name=semantic_label_name,
            declared_num_points=declared_num_points,
        )
    return instances


def load_scene_inventory(
    scene: str,
    scene_candidates: list[Path],
    segmentation_dir: Path,
    scene_objects_root: Path,
    export_instances: bool,
) -> dict[str, Any]:
    segmentation = json.loads((segmentation_dir / f"{scene}_segment.json").read_text())
    vertex_instance = np.asarray(segmentation["vertex_instance"], dtype=np.int64)
    vertex_category = np.asarray(segmentation["vertex_category"], dtype=np.int64)
    matching_scene_paths = [
        path
        for path in scene_candidates
        if path.exists() and read_ply_vertex_count(path) == vertex_instance.shape[0]
    ]
    if not matching_scene_paths:
        candidate_counts = {
            str(path): read_ply_vertex_count(path) for path in scene_candidates if path.exists()
        }
        raise ValueError(
            f"{scene}: no PLY vertex count matches segmentation length {vertex_instance.shape[0]}. "
            f"Candidates: {candidate_counts}"
        )
    scene_path = matching_scene_paths[0]
    xyz, rgb = load_scene_vertices(scene_path)
    if xyz.shape[0] != vertex_instance.shape[0]:
        raise ValueError(
            f"{scene}: scene vertices ({xyz.shape[0]}) and segmentation ({vertex_instance.shape[0]}) differ"
        )

    instance_infos = parse_instance_txt(segmentation_dir / f"{scene}.txt")
    scene_root = scene_objects_root / scene
    instances_root = scene_root / "instances"

    catalog_instances: list[dict[str, Any]] = []
    points_by_object_idx: dict[int, np.ndarray] = {}
    colors_by_object_idx: dict[int, np.ndarray | None] = {}

    for object_idx, info in sorted(instance_infos.items()):
        mask = vertex_instance == info.vertex_instance_id
        obj_xyz = xyz[mask]
        obj_rgb = rgb[mask] if rgb is not None else None
        points_by_object_idx[object_idx] = obj_xyz
        colors_by_object_idx[object_idx] = obj_rgb

        ply_rel = Path("instances") / f"instance_{object_idx:04d}.ply"
        if export_instances:
            write_point_ply(scene_root / ply_rel, obj_xyz, obj_rgb)

        if obj_xyz.shape[0]:
            bbox_min = obj_xyz.min(axis=0).astype(float).tolist()
            bbox_max = obj_xyz.max(axis=0).astype(float).tolist()
            centroid = obj_xyz.mean(axis=0).astype(float).tolist()
        else:
            bbox_min = bbox_max = centroid = [0.0, 0.0, 0.0]

        category_values = vertex_category[mask]
        semantic_label_id = int(np.bincount(category_values).argmax()) if category_values.size else None
        catalog_instances.append(
            {
                "instance_id": f"instance_{object_idx:04d}",
                "scene_name": scene,
                "object_idx": object_idx,
                "vertex_instance_id": info.vertex_instance_id,
                "semantic_label_id": semantic_label_id,
                "semantic_label_name": info.semantic_label_name,
                "num_points": int(obj_xyz.shape[0]),
                "declared_num_points": info.declared_num_points,
                "centroid": centroid,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "ply_path": str(ply_rel),
            }
        )

    catalog = {
        "scene_name": scene,
        "coordinate_frame": "PROX_scene",
        "num_scene_points": int(xyz.shape[0]),
        "instance_id_rule": "interaction_obj_idx k maps to instance_%04d and segmentation vertex_instance == k + 1",
        "instances": catalog_instances,
    }
    scene_root.mkdir(parents=True, exist_ok=True)
    (scene_root / "label_pointcloud_export.json").write_text(json.dumps(catalog, indent=2))

    return {
        "scene": scene,
        "scene_ply_path": str(scene_path),
        "xyz": xyz,
        "rgb": rgb,
        "catalog": catalog,
        "points_by_object_idx": points_by_object_idx,
        "colors_by_object_idx": colors_by_object_idx,
    }


def frame_pose(record: dict[str, Any]) -> np.ndarray:
    param = record["smplx_param"]
    global_orient = np.asarray(param["global_orient"], dtype=np.float32).reshape(-1)[:3]
    transl = np.asarray(param["transl"], dtype=np.float32).reshape(-1)[:3]
    pose_embedding = np.asarray(param["pose_embedding"], dtype=np.float32).reshape(-1)[:32]
    return np.concatenate([global_orient, transl, pose_embedding], axis=0).astype(np.float32)


def action_key(record: dict[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    return tuple(record["interaction_labels"]), tuple(int(v) for v in record["interaction_obj_idx"])


def object_name_from_label(label: str) -> str:
    if "-" not in label:
        return label.strip().lower()
    return label.rsplit("-", 1)[1].strip().lower()


def choose_target_object(record: dict[str, Any], scene_catalog: dict[int, InstanceInfo]) -> dict[str, Any]:
    candidates = []
    for label, object_idx in zip(record["interaction_labels"], record["interaction_obj_idx"]):
        object_idx = int(object_idx)
        label_object_name = object_name_from_label(label)
        catalog_name = scene_catalog.get(object_idx).semantic_label_name if object_idx in scene_catalog else None
        semantic_name = (catalog_name or label_object_name).lower()
        is_floor_wall = label_object_name in FLOOR_WALL or semantic_name in FLOOR_WALL
        row = {
            "interaction_label": label,
            "object_idx": object_idx,
            "label_object_name": label_object_name,
            "semantic_label_name": catalog_name,
            "is_floor_wall": is_floor_wall,
        }
        candidates.append(row)

    non_floor_wall = [row for row in candidates if not row["is_floor_wall"]]
    if not non_floor_wall:
        return {"target": None, "candidates": candidates}
    return {"target": non_floor_wall[-1], "candidates": candidates}


def segment_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sequence[record["sequence"]].append(record)

    segments: list[list[dict[str, Any]]] = []
    for sequence in sorted(by_sequence):
        rows = sorted(by_sequence[sequence], key=lambda row: int(row["frame_idx"]))
        start = 0
        for idx in range(1, len(rows) + 1):
            if idx == len(rows) or action_key(rows[idx]) != action_key(rows[start]):
                segments.append(rows[start:idx])
                start = idx
    return segments


def make_zero_smplx_args(batch_size: int, dtype: torch.dtype, device: torch.device, body_model: Any) -> dict[str, torch.Tensor]:
    return {
        "betas": torch.zeros(batch_size, getattr(body_model, "num_betas", 10), dtype=dtype, device=device),
        "left_hand_pose": torch.zeros(batch_size, getattr(body_model, "num_pca_comps", 12), dtype=dtype, device=device),
        "right_hand_pose": torch.zeros(batch_size, getattr(body_model, "num_pca_comps", 12), dtype=dtype, device=device),
        "expression": torch.zeros(batch_size, getattr(body_model, "num_expression_coeffs", 10), dtype=dtype, device=device),
        "jaw_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
        "leye_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
        "reye_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
    }


@torch.no_grad()
def poses_to_joints(
    poses: torch.Tensor,
    vposer: Any,
    body_model: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    flat = poses.reshape(-1, poses.shape[-1]).to(device)
    outputs: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = flat[start : start + batch_size]
        body_pose = vposer.decode(chunk[:, 6:], output_type="aa").view(-1, 63)
        body_model.batch_size = chunk.shape[0]
        smplx_out = body_model(
            return_verts=False,
            body_pose=body_pose,
            global_orient=chunk[:, :3],
            transl=chunk[:, 3:6],
            pose_embedding=chunk[:, 6:],
            **make_zero_smplx_args(chunk.shape[0], chunk.dtype, device, body_model),
        )
        outputs.append(smplx_out.joints.detach().cpu())
    return torch.cat(outputs, dim=0).reshape(poses.shape[0], poses.shape[1], 127, 3).float()


def build_split(
    split: str,
    records_path: Path,
    output_split_root: Path,
    inventories: dict[str, dict[str, Any]],
    segmentation_dir: Path,
    sample_points: int,
    object_points: int,
    drop_no_target: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = pickle.loads(records_path.read_bytes())
    raw_segments = segment_records(records)
    kept_segments = [segment for segment in raw_segments if len(segment) >= TOTAL_LEN]

    output_split_root.mkdir(parents=True, exist_ok=True)
    poses_all: list[np.ndarray] = []
    scene_points_all: list[np.ndarray] = []
    object_points_all: list[np.ndarray] = []
    object_mask_all: list[bool] = []
    manifest: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    skipped_no_target = 0

    scene_instance_info = {
        scene: parse_instance_txt(segmentation_dir / f"{scene}.txt") for scene in inventories
    }

    for segment in kept_segments:
        scene = segment[0]["scene_name"]
        sequence = segment[0]["sequence"]
        sampled_idx = np.rint(np.linspace(0, len(segment) - 1, TOTAL_LEN)).astype(np.int64)
        sampled = [segment[int(idx)] for idx in sampled_idx]
        last_record = sampled[-1]
        target_info = choose_target_object(last_record, scene_instance_info[scene])
        target = target_info["target"]
        has_target = target is not None
        if drop_no_target and not has_target:
            skipped_no_target += 1
            continue

        sample_index = len(poses_all)
        sample_id = f"{split}_{sample_index:05d}_{sequence}_f{segment[0]['frame_idx']}_f{segment[-1]['frame_idx']}"
        poses = np.stack([frame_pose(record) for record in sampled], axis=0)
        poses_all.append(poses)

        scene_xyz = inventories[scene]["xyz"]
        scene_points_all.append(sample_points_fn(scene_xyz, sample_points, stable_seed("scene", scene)))

        object_idx = int(target["object_idx"]) if has_target else None
        if object_idx is None:
            object_points_all.append(np.zeros((object_points, 3), dtype=np.float32))
            object_mask_all.append(False)
        else:
            obj_xyz = inventories[scene]["points_by_object_idx"].get(object_idx, np.zeros((0, 3), dtype=np.float32))
            has_points = obj_xyz.shape[0] > 0
            object_points_all.append(sample_points_fn(obj_xyz, object_points, stable_seed("object", scene, object_idx)))
            object_mask_all.append(has_points)
            if not has_points:
                has_target = False

        manifest.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "split": split,
                "scene_name": scene,
                "sequence": sequence,
                "segment_start_frame_idx": int(segment[0]["frame_idx"]),
                "segment_end_frame_idx": int(segment[-1]["frame_idx"]),
                "segment_num_frames": len(segment),
                "sampled_frame_indices": [int(record["frame_idx"]) for record in sampled],
                "interaction_labels": list(segment[0]["interaction_labels"]),
                "interaction_obj_idx": [int(v) for v in segment[0]["interaction_obj_idx"]],
                "final_frame_interactions": target_info["candidates"],
                "target_object": target,
                "target_object_mask": bool(object_mask_all[-1]),
                "target_rule": "last sampled frame, last non-floor/wall interaction",
            }
        )
        dataset_rows.append(
            {
                "sequence_path": sample_id,
                "start_frame": int(segment[0]["frame_idx"]),
                "end_frame": int(segment[-1]["frame_idx"]),
                "scene": scene,
                "transformation": "identity",
                "training": 1 if split == "train" else 0,
            }
        )

    if not poses_all:
        raise ValueError(f"{split}: no samples produced")

    poses_tensor = torch.from_numpy(np.stack(poses_all, axis=0)).float()
    torch.save(poses_tensor[:, :INPUT_LEN], output_split_root / "poses_input.pth")
    torch.save(poses_tensor[:, INPUT_LEN:], output_split_root / "poses_label.pth")
    torch.save(torch.from_numpy(np.stack(scene_points_all, axis=0)).float(), output_split_root / f"scene_points_{sample_points}.pth")
    torch.save(torch.from_numpy(np.stack(object_points_all, axis=0)).float(), output_split_root / f"final_object_gt_points_{object_points}.pth")
    torch.save(torch.tensor(object_mask_all, dtype=torch.bool), output_split_root / f"final_object_gt_points_{object_points}_mask.pth")
    (output_split_root / "segment_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_split_root / f"final_object_gt_points_{object_points}_manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "split": split,
        "records": len(records),
        "raw_segments": len(raw_segments),
        "segments_len_at_least_16": len(kept_segments),
        "samples": len(poses_all),
        "object_valid_samples": int(sum(object_mask_all)),
        "object_missing_or_floor_wall_samples": int(len(object_mask_all) - sum(object_mask_all)),
        "skipped_no_target": skipped_no_target,
    }
    (output_split_root / "summary_no_joints_yet.json").write_text(json.dumps(summary, indent=2))
    return dataset_rows, summary


def sample_points_fn(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    return sample_points(points, count, seed)


def add_joints_to_split(
    output_split_root: Path,
    vposer: Any,
    body_model: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    poses_input = torch.load(output_split_root / "poses_input.pth", map_location="cpu", weights_only=False).float()
    poses_label = torch.load(output_split_root / "poses_label.pth", map_location="cpu", weights_only=False).float()
    poses_full = torch.cat([poses_input, poses_label], dim=1)
    joints = poses_to_joints(poses_full, vposer, body_model, batch_size=batch_size, device=device)
    torch.save(joints[:, :INPUT_LEN], output_split_root / "joints_input.pth")
    torch.save(joints[:, INPUT_LEN:], output_split_root / "joints_label.pth")
    return {
        "poses_input": list(poses_input.shape),
        "poses_label": list(poses_label.shape),
        "joints_input": list(joints[:, :INPUT_LEN].shape),
        "joints_label": list(joints[:, INPUT_LEN:].shape),
    }


def write_dataset_csv(output_root: Path, rows: list[dict[str, Any]]) -> None:
    with (output_root / "dataset.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence_path", "start_frame", "end_frame", "scene", "transformation", "training"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-root", type=Path, default=Path("data/PROX"))
    parser.add_argument("--coins-root", type=Path, default=Path("data/PROX/COINS"))
    parser.add_argument("--scenes-semantics-root", type=Path, default=Path("data/PROX/scenes_semantics"))
    parser.add_argument("--output-root", type=Path, default=Path("data/PROX/PROX_COINS_GIMO"))
    parser.add_argument("--smplx-path", type=Path, default=Path("data/body_models"))
    parser.add_argument("--vposer-path", type=Path, default=Path("./vposer_v1_0"))
    parser.add_argument("--sample-points", type=int, default=32768)
    parser.add_argument("--object-points", type=int, default=1024)
    parser.add_argument("--smplx-batch-size", type=int, default=256)
    parser.add_argument("--drop-no-target", action="store_true")
    parser.add_argument("--no-export-instance-ply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    slices_root = output_root / "SLICES_888s"
    scene_objects_root = output_root / "scene_objects"
    scene_dir = args.prox_root / "scenes"
    scene_semantics_dir = args.scenes_semantics_root
    segmentation_dir = args.coins_root / "scene_segmentation"
    semantics_dir = args.coins_root / "interaction_semantics"

    scene_names = sorted(path.name.removesuffix("_segment.json") for path in segmentation_dir.glob("*_segment.json"))
    inventories = {
        scene: load_scene_inventory(
            scene,
            [
                scene_dir / f"{scene}.ply",
                scene_semantics_dir / f"{scene}.ply",
                scene_semantics_dir / f"{scene}_withlabels.ply",
                scene_semantics_dir / f"{scene}_withlabels_OpenAWall.ply",
                scene_semantics_dir / f"{scene}_withlabels_old.ply",
            ],
            segmentation_dir,
            scene_objects_root,
            export_instances=not args.no_export_instance_ply,
        )
        for scene in scene_names
    }

    dataset_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in ("train", "test"):
        rows, summary = build_split(
            split=split,
            records_path=semantics_dir / f"{split}.pkl",
            output_split_root=slices_root / split,
            inventories=inventories,
            segmentation_dir=segmentation_dir,
            sample_points=args.sample_points,
            object_points=args.object_points,
            drop_no_target=args.drop_no_target,
        )
        dataset_rows.extend(rows)
        split_summaries[split] = summary

    write_dataset_csv(output_root, dataset_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vposer, _ = load_vposer(str(args.vposer_path), vp_model="snapshot")
    vposer = vposer.to(device).eval()
    body_model = smplx.create(
        str(args.smplx_path),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        num_pca_comps=12,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=args.smplx_batch_size,
    ).to(device).eval()

    shape_summary = {}
    for split in ("train", "test"):
        shape_summary[split] = add_joints_to_split(
            slices_root / split,
            vposer=vposer,
            body_model=body_model,
            batch_size=args.smplx_batch_size,
            device=device,
        )

    final_summary = {
        "output_root": str(output_root),
        "coordinate_frame": "PROX_scene",
        "input_seq_len": INPUT_LEN,
        "output_seq_len": OUTPUT_LEN,
        "sample_points": args.sample_points,
        "object_points": args.object_points,
        "target_rule": "last sampled frame, last non-floor/wall interaction; mask false if no valid object",
        "splits": split_summaries,
        "tensor_shapes": shape_summary,
        "dataset_csv_rows": len(dataset_rows),
    }
    (output_root / "preprocess_summary.json").write_text(json.dumps(final_summary, indent=2))
    for split, summary in split_summaries.items():
        summary_path = slices_root / split / "summary.json"
        summary = dict(summary)
        summary["tensor_shapes"] = shape_summary[split]
        summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
