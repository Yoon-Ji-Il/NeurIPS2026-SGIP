#!/usr/bin/env python3
"""Run one unchanged HIK H25F50 sample through the separate SIF3D model."""

from __future__ import annotations

import argparse
import json
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
    HIK1s2sTemporalSpec,
    RawHIKIndex,
    flatten_test_events,
    merge_environment_and_people,
    other_people_last_frame_joint_points,
    primary_motion_and_joints,
    sample_hik_environment,
)
from model.sif3d_hik_1s2s import (  # noqa: E402
    SIF3D,
    load_gimo_checkpoint_for_hik,
)
from utils.hik_metrics import compute_hik_horizon_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kitchen", default="A", choices=list("ABCD"))
    parser.add_argument("--action", default="walking")
    parser.add_argument(
        "--action-index",
        type=int,
        default=2,
        help=(
            "Defaults to a verified A/walking multi-person event with four "
            "nearby non-target people."
        ),
    )
    parser.add_argument("--scene-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs_hik"
            / "scene_graphs_h25f50_action_geometry_v1"
        ),
    )
    parser.add_argument(
        "--object-cache-root",
        type=Path,
        default=Path(
            "data/HIK/"
            "hik_preprocessed/H25F50/SLICES_888s"
        ),
    )
    parser.add_argument(
        "--object-points-name",
        default="",
        help="Defaults to HIK1s2sConfig.object_points_name.",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "./runs_dh/"
            "teacher_motion_c_final_seed7/best_loss_trans.pth"
        ),
        help=(
            "Trained GIMO checkpoint used as a warm start. Pass an empty "
            "string to run with random HIK-model weights."
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "./"
            "runs_hik/hik_1s2s_real_sample_smoke.json"
        ),
    )
    return parser.parse_args()


def choose_event(index, kitchen, action, action_index):
    events = [
        event
        for event in flatten_test_events(index.load_events())
        if event["kitchen"] == kitchen and event["action"] == action
    ]
    if action_index < 0 or action_index >= len(events):
        raise IndexError(
            f"{kitchen}/{action} has {len(events)} events; "
            f"action-index={action_index} is invalid"
        )
    return events[action_index]


def make_body_model(config, device):
    return smplx.create(
        config.smplx_path,
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
        batch_size=75,
    ).to(device)


def event_sample_id(event):
    return (
        f"{event['kitchen']}|{event['pid']}|{event['action']}|"
        f"{event['action_index']}|{event['future_anchor']}"
    )


def load_cached_object(args, config, event):
    split = "train" if event["kitchen"] in {"A", "B", "C"} else "test"
    sample_id = event_sample_id(event)
    graph_path = args.scene_graph_root / split / "samples.jsonl"
    graph_index = None
    with graph_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            graph = json.loads(line)
            if graph["sample_id"] == sample_id:
                graph_index = index
                break
    if graph_index is None:
        raise KeyError(f"{sample_id} is not present in {graph_path}")

    object_name = args.object_points_name or config.object_points_name
    object_path = args.object_cache_root / split / object_name
    mask_path = object_path.with_name(
        object_path.name.replace(".pth", "_mask.pth")
    )
    manifest_path = object_path.with_name(
        object_path.name.replace(".pth", "_manifest.json")
    )
    object_cache = torch.load(
        object_path,
        map_location="cpu",
        weights_only=False,
    ).float()
    object_mask_cache = torch.load(
        mask_path,
        map_location="cpu",
        weights_only=False,
    ).bool()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        len(object_cache)
        == len(object_mask_cache)
        == len(manifest)
    ):
        raise ValueError("HIK object tensor/mask/manifest lengths disagree")
    manifest_row = manifest[graph_index]
    if manifest_row["sample_id"] != sample_id:
        raise ValueError(
            f"Object cache order mismatch at {graph_index}: "
            f"{manifest_row['sample_id']} != {sample_id}"
        )
    return (
        object_cache[graph_index].unsqueeze(0),
        object_mask_cache[graph_index].reshape(1),
        {
            "sample_id": sample_id,
            "split": split,
            "graph_index": graph_index,
            "object_path": str(object_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "selected_object_id": manifest_row.get("selected_object_id"),
            "selected_object_name": manifest_row.get("selected_object_name"),
            "selector_status": manifest_row.get("selector_status"),
            "selector_confidence": manifest_row.get("selector_confidence"),
            "object_cache_status": manifest_row.get("status"),
            "object_points": int(object_cache.shape[1]),
            "object_mask": bool(object_mask_cache[graph_index]),
        },
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The PointNet++ extension used by this repository requires CUDA"
        )
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    config = HIK1s2sConfig().parse_args(
        [
            "--batch_size",
            "1",
            "--num_workers",
            "0",
            "--sample_points",
            str(args.scene_points),
        ]
    )
    spec = HIK1s2sTemporalSpec()
    spec.validate()
    index = RawHIKIndex(config.hik_raw_root)
    event = choose_event(
        index,
        args.kitchen,
        args.action,
        args.action_index,
    )
    frames = spec.frames(event["future_anchor"])

    vposer, _ = load_vposer(config.vposer_path, vp_model="snapshot")
    vposer = vposer.to(device).eval()
    body_model = make_body_model(config, device).eval()

    started = time.time()
    motion, joints, target_track, scene_origin = primary_motion_and_joints(
        index=index,
        kitchen=event["kitchen"],
        pid=event["pid"],
        frames=frames,
        vposer=vposer,
        body_model=body_model,
        device=device,
        input_frames=spec.input_frames,
    )
    last_observed_frame = frames[spec.input_frames - 1]
    people_points, people_metadata = other_people_last_frame_joint_points(
        index=index,
        kitchen=event["kitchen"],
        target_pid=event["pid"],
        last_observed_frame=last_observed_frame,
        scene_origin=scene_origin,
        max_root_distance=config.hik_other_person_radius,
        max_other_people=config.hik_max_other_people,
    )
    environment = sample_hik_environment(
        data_root=config.hik_raw_root,
        kitchen=event["kitchen"],
        # Dynamic objects are frozen at the latest observed frame.
        frame=last_observed_frame,
        num_points=args.scene_points,
        rng=rng,
        scene_origin=scene_origin,
        max_distance=config.hik_scene_radius,
    )
    scene, scene_metadata = merge_environment_and_people(
        environment_points=environment,
        people_points=people_points,
        total_points=args.scene_points,
        max_people_points=config.hik_max_people_scene_points,
        rng=rng,
    )
    preprocess_seconds = time.time() - started

    model = SIF3D(config, vposer, body_model).to(device)
    if args.checkpoint:
        checkpoint_report = load_gimo_checkpoint_for_hik(
            model,
            args.checkpoint,
            map_location="cpu",
        )
    else:
        checkpoint_report = {
            "checkpoint": None,
            "requires_hik_finetuning": True,
        }
    model = model.to(device).eval()
    object_points, object_mask, object_metadata = load_cached_object(
        args,
        config,
        event,
    )
    object_points = object_points.to(device)
    object_mask = object_mask.to(device)
    input_motion = motion[: spec.input_frames].unsqueeze(0).to(device)
    input_joints = joints[: spec.input_frames].unsqueeze(0).to(device)
    scene_tensor = torch.as_tensor(
        scene,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    torch.cuda.synchronize(device)
    started = time.time()
    with torch.no_grad():
        predicted_motion, predicted_joints = model(
            input_motion,
            input_joints,
            scene_tensor,
            object_points,
            object_mask,
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.time() - started

    target_joints = joints.unsqueeze(0).to(device)
    metrics = compute_hik_horizon_metrics(
        predicted_joints,
        target_joints,
        input_seq_len=spec.input_frames,
        output_seq_len=spec.future_frames,
        fps=spec.target_fps,
    )

    expected_motion_shape = (1, 75, 38)
    expected_joint_shape = (1, 75, 23, 3)
    if tuple(predicted_motion.shape) != expected_motion_shape:
        raise AssertionError(
            f"motion shape {tuple(predicted_motion.shape)} != "
            f"{expected_motion_shape}"
        )
    if tuple(predicted_joints.shape) != expected_joint_shape:
        raise AssertionError(
            f"joint shape {tuple(predicted_joints.shape)} != "
            f"{expected_joint_shape}"
        )
    if model.motion_decoder.gc1.in_features != 75:
        raise AssertionError(
            "The isolated GCN decoder is not 75 frames wide"
        )
    if tuple(scene.shape) != (1000, 3):
        raise AssertionError(
            f"Expected native HIK scene shape (1000, 3), got {scene.shape}"
        )
    if tuple(object_points.shape) != (1, 1024, 3):
        raise AssertionError(
            "Expected selected-object shape (1, 1024, 3), "
            f"got {tuple(object_points.shape)}"
        )
    if people_metadata["points"] != people_metadata["num_people"] * 29:
        raise AssertionError(
            "Every included non-target person must contribute exactly "
            "29 last-frame joint points"
        )
    if not torch.isfinite(predicted_motion).all():
        raise AssertionError("Non-finite predicted motion")
    if not torch.isfinite(predicted_joints).all():
        raise AssertionError("Non-finite predicted joints")

    report = {
        "status": "PASS",
        "note": (
            "GIMO checkpoint warm-start only; temporal parameters were resized "
            "from 16 to 75 frames and still require HIK fine-tuning."
        ),
        "event": event,
        "target_track": target_track.path.name,
        "source_fps": spec.source_fps,
        "target_fps": spec.target_fps,
        "offsets": list(spec.offsets),
        "frames": list(frames),
        "input_frames": list(frames[: spec.input_frames]),
        "future_frames": list(frames[spec.input_frames :]),
        "motion_shape": list(predicted_motion.shape),
        "joint_shape": list(predicted_joints.shape),
        "gcn_temporal_features": model.motion_decoder.gc1.in_features,
        "checkpoint_initialization": checkpoint_report,
        "people": people_metadata,
        "scene": scene_metadata,
        "object": object_metadata,
        "scene_origin_model_y_up": scene_origin.tolist(),
        "metrics": metrics,
        "preprocess_seconds": preprocess_seconds,
        "forward_seconds": forward_seconds,
        "gpu": torch.cuda.get_device_name(device),
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
