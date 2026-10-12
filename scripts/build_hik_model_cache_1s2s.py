#!/usr/bin/env python3
"""Build split-aligned HIK H25F50 motion, joint, and 1k-scene caches."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_hik_1s2s import HIK1s2sConfig  # noqa: E402
from dataset.hik_raw_1s2s import (  # noqa: E402
    BOX_TRIANGLES,
    RawHIKIndex,
    _smplx_zeros,
    hik_orientations_to_model,
    hik_points_to_model,
    merge_environment_and_people,
    other_people_last_frame_joint_points,
)
from object_selector.hik_action_geometry import (  # noqa: E402
    HIKObjectCatalog,
    iter_jsonl,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/HIK/SAST/data"
        ),
    )
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/HIK/"
            "hik_preprocessed/H25F50/SLICES_888s"
        ),
    )
    parser.add_argument("--scene-points", type=int, default=1000)
    parser.add_argument("--scene-radius", type=float, default=3.5)
    parser.add_argument("--other-person-radius", type=float, default=2.5)
    parser.add_argument("--max-other-people", type=int, default=15)
    parser.add_argument("--max-people-points", type=int, default=435)
    parser.add_argument(
        "--vposer-path",
        type=Path,
        default=REPO_ROOT / "vposer_v1_0",
    )
    parser.add_argument(
        "--smplx-path",
        type=Path,
        default=Path("data/body_models"),
    )
    return parser.parse_args()


def selected_splits(value: str) -> tuple[str, ...]:
    return ("train", "test") if value == "all" else (value,)


def make_body_model(
    smplx_path: Path,
    device: torch.device,
    batch_size: int,
):
    return smplx.create(
        str(smplx_path),
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
        batch_size=batch_size,
    ).to(device).eval()


class EnvironmentSampler:
    """Area-sample cached HIK boxes without reopening every NPY per event."""

    def __init__(self, catalog: HIKObjectCatalog):
        self.objects = {
            kitchen: [obj for obj in rows if obj.shape == "box"]
            for kitchen, rows in catalog.by_kitchen.items()
        }

    @staticmethod
    def _draw(
        triangles: np.ndarray,
        areas: np.ndarray,
        count: int,
        rng: np.random.Generator,
        origin: np.ndarray,
    ) -> np.ndarray:
        chosen_indices = rng.choice(
            len(triangles),
            size=int(count),
            replace=True,
            p=areas / areas.sum(),
        )
        chosen = triangles[chosen_indices]
        u = rng.random(count, dtype=np.float32)
        v = rng.random(count, dtype=np.float32)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        points = (
            chosen[:, 0]
            + u[:, None] * (chosen[:, 1] - chosen[:, 0])
            + v[:, None] * (chosen[:, 2] - chosen[:, 0])
        )
        return hik_points_to_model(points) - origin[None]

    def sample(
        self,
        kitchen: str,
        frame: int,
        origin: np.ndarray,
        count: int,
        radius: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        triangles = np.concatenate(
            [
                np.asarray(obj.geometry_at(frame), dtype=np.float32)[BOX_TRIANGLES]
                for obj in self.objects[kitchen]
            ],
            axis=0,
        )
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) * 0.5
        valid = np.isfinite(areas) & (areas > 1e-10)
        triangles = triangles[valid]
        areas = areas[valid]
        accepted = []
        candidate_count = max(int(count) * 8, 4096)
        for _ in range(8):
            candidates = self._draw(
                triangles,
                areas,
                candidate_count,
                rng,
                origin,
            )
            nearby = candidates[
                np.linalg.norm(candidates, axis=-1) <= float(radius)
            ]
            if len(nearby):
                accepted.append(nearby)
            if sum(len(item) for item in accepted) >= int(count):
                break
        if not accepted:
            raise RuntimeError(
                f"No environment points for kitchen={kitchen}, frame={frame}"
            )
        accepted_points = np.concatenate(accepted, axis=0)
        keep = rng.choice(
            len(accepted_points),
            size=int(count),
            replace=len(accepted_points) < int(count),
        )
        return accepted_points[keep].astype(np.float32, copy=False)


def load_target_payload(
    index: RawHIKIndex,
    graph: dict,
) -> dict:
    frames = tuple(graph["input_frames"]) + tuple(graph["future_frames"])
    ref = index.find_complete_track(graph["kitchen"], graph["pid"], frames)
    if ref is None:
        raise KeyError(f"No complete track for {graph['sample_id']}")
    rows = index.load_rows(ref, frames)
    origin = hik_points_to_model(rows["poses3d"][24, 0]).reshape(3)
    transforms = np.asarray(rows["transforms"], dtype=np.float32)
    return {
        "rows": rows,
        "origin": origin,
        "body_pose": np.asarray(rows["smpl"], dtype=np.float32).reshape(75, 63),
        "global_orient": hik_orientations_to_model(transforms[:, 3:6]),
        "translation": (
            hik_points_to_model(transforms[:, :3]) - origin[None]
        ).astype(np.float32),
        "betas": np.repeat(
            np.asarray(rows["betas"], dtype=np.float32)[None],
            75,
            axis=0,
        ),
        "track": ref.path.name,
    }


@torch.no_grad()
def encode_batch(
    payloads: list[dict],
    vposer,
    body_model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(payloads)
    total = batch * 75
    body_pose = torch.as_tensor(
        np.stack([item["body_pose"] for item in payloads]).reshape(total, 63),
        dtype=torch.float32,
        device=device,
    )
    global_orient = torch.as_tensor(
        np.stack([item["global_orient"] for item in payloads]).reshape(total, 3),
        dtype=torch.float32,
        device=device,
    )
    translation = torch.as_tensor(
        np.stack([item["translation"] for item in payloads]).reshape(total, 3),
        dtype=torch.float32,
        device=device,
    )
    betas = torch.as_tensor(
        np.stack([item["betas"] for item in payloads]).reshape(total, 10),
        dtype=torch.float32,
        device=device,
    )
    encoded = vposer.encode(body_pose).mean
    body_model.batch_size = total
    output = body_model(
        return_verts=False,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=translation,
        betas=betas,
        **_smplx_zeros(
            body_model,
            total,
            body_pose.dtype,
            device,
        ),
    )
    motion = torch.cat([global_orient, translation, encoded], dim=-1)
    return (
        motion.reshape(batch, 75, 38).detach().cpu(),
        output.joints[:, :23].reshape(batch, 75, 23, 3).detach().cpu(),
    )


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_split(
    split: str,
    args: argparse.Namespace,
    index: RawHIKIndex,
    environment_sampler: EnvironmentSampler,
    vposer,
    body_model,
    device: torch.device,
) -> dict:
    graphs = list(iter_jsonl(args.scene_graph_root / split / "samples.jsonl"))
    if args.limit >= 0:
        graphs = graphs[: args.limit]
    count = len(graphs)
    poses = torch.empty((count, 75, 38), dtype=torch.float32)
    joints = torch.empty((count, 75, 23, 3), dtype=torch.float32)
    scenes = torch.empty((count, args.scene_points, 3), dtype=torch.float32)
    manifest = []
    started = time.time()

    for batch_start in range(0, count, args.batch_size):
        batch_graphs = graphs[batch_start : batch_start + args.batch_size]
        payloads = [load_target_payload(index, graph) for graph in batch_graphs]
        batch_motion, batch_joints = encode_batch(
            payloads,
            vposer,
            body_model,
            device,
        )
        poses[batch_start : batch_start + len(batch_graphs)] = batch_motion
        joints[batch_start : batch_start + len(batch_graphs)] = batch_joints

        for offset, (graph, payload) in enumerate(zip(batch_graphs, payloads)):
            output_index = batch_start + offset
            rng = np.random.default_rng(
                stable_seed("hik_scene", graph["sample_id"], args.seed)
            )
            people, people_meta = other_people_last_frame_joint_points(
                index=index,
                kitchen=graph["kitchen"],
                target_pid=graph["pid"],
                last_observed_frame=graph["last_observed_frame"],
                scene_origin=payload["origin"],
                max_root_distance=args.other_person_radius,
                max_other_people=args.max_other_people,
            )
            environment = environment_sampler.sample(
                graph["kitchen"],
                graph["last_observed_frame"],
                payload["origin"],
                args.scene_points,
                args.scene_radius,
                rng,
            )
            scene, scene_meta = merge_environment_and_people(
                environment,
                people,
                args.scene_points,
                args.max_people_points,
                rng,
            )
            scenes[output_index] = torch.from_numpy(scene)
            manifest.append(
                {
                    "index": output_index,
                    "sample_id": graph["sample_id"],
                    "split": split,
                    "kitchen": graph["kitchen"],
                    "pid": graph["pid"],
                    "event_action": graph["event_action"],
                    "future_anchor": graph["future_anchor"],
                    "track": payload["track"],
                    "scene_origin_model_y_up": payload["origin"].tolist(),
                    "other_people": people_meta,
                    "scene": scene_meta,
                }
            )
        done = batch_start + len(batch_graphs)
        elapsed = time.time() - started
        rate = done / max(elapsed, 1e-6)
        remaining = (count - done) / max(rate, 1e-6)
        print(
            f"[{split}] {done}/{count} "
            f"rate={rate:.2f} samples/s eta={remaining / 60:.1f}m",
            flush=True,
        )

    split_dir = args.output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "poses_input.pth": poses[:, :25].contiguous(),
        "poses_label.pth": poses[:, 25:].contiguous(),
        "joints_input.pth": joints[:, :25].contiguous(),
        "joints_label.pth": joints[:, 25:].contiguous(),
        f"scene_points_{args.scene_points}.pth": scenes.contiguous(),
    }
    for name, value in outputs.items():
        atomic_torch_save(value, split_dir / name)
    write_json(split_dir / "hik_h25f50_cache_manifest.json", manifest)
    summary = {
        "split": split,
        "samples": count,
        "poses_input_shape": list(outputs["poses_input.pth"].shape),
        "poses_label_shape": list(outputs["poses_label.pth"].shape),
        "joints_input_shape": list(outputs["joints_input.pth"].shape),
        "joints_label_shape": list(outputs["joints_label.pth"].shape),
        "scene_shape": list(
            outputs[f"scene_points_{args.scene_points}.pth"].shape
        ),
        "seconds": time.time() - started,
        "people_histogram": {
            str(number): sum(
                row["other_people"]["num_people"] == number for row in manifest
            )
            for number in range(args.max_other_people + 1)
        },
    }
    write_json(split_dir / "hik_h25f50_cache_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VPoser/SMPL-X cache generation")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.scene_points != 1000:
        raise ValueError("Native HIK scene cache must contain 1000 points")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    config = HIK1s2sConfig().parse_args([])
    index = RawHIKIndex(args.raw_root)
    catalog = HIKObjectCatalog(args.raw_root)
    environment_sampler = EnvironmentSampler(catalog)
    vposer, _ = load_vposer(str(args.vposer_path), vp_model="snapshot")
    vposer = vposer.to(device).eval()
    body_model = make_body_model(
        args.smplx_path,
        device,
        args.batch_size * 75,
    )

    summaries = {}
    for split in selected_splits(args.split):
        summaries[split] = build_split(
            split,
            args,
            index,
            environment_sampler,
            vposer,
            body_model,
            device,
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "hik_h25f50_model_cache_summary.json",
        {
            "raw_root": str(args.raw_root.resolve()),
            "scene_graph_root": str(args.scene_graph_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "gpu": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "splits": summaries,
        },
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
