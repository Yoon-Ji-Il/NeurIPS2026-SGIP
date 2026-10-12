#!/usr/bin/env python3
"""Train VPoser-free SIF3D on native HIK 25-observed/50-future joints."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_hik_direct29_1s2s import HIKDirect29Config
from dataset.hik_dataset_direct29_1s2s import EgoEvalDataset
from model.sif3d_hik_direct29_1s2s import (
    SIF3DDirect29,
    load_gimo_checkpoint_for_direct29,
)
from utils.hik_direct29_losses import direct29_losses
from utils.hik_metrics import compute_hik_horizon_metrics


def make_parser():
    parser = HIKDirect29Config()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument(
        "--warm_start_gimo",
        type=str,
        default=(
            "./runs_dh/"
            "teacher_motion_c_final_seed7/best_loss_trans.pth"
        ),
    )
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient_audit_batches",
        type=int,
        default=0,
        help=(
            "Before training, separately backpropagate root/pose/yaw for this "
            "many batches and save per-module gradient norms."
        ),
    )
    return parser


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def atomic_write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_checkpoint(model, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)


def move_batch(batch, device):
    (
        objects,
        object_mask,
        joints_input,
        joints_label,
        orient_input,
        orient_label,
        scenes,
        sample_ids,
        kitchens,
    ) = batch
    return {
        "objects": objects.to(device, non_blocking=True),
        "object_mask": object_mask.to(device, non_blocking=True),
        "joints_input": joints_input.to(device, non_blocking=True),
        "joints_label": joints_label.to(device, non_blocking=True),
        "orient_input": orient_input.to(device, non_blocking=True),
        "orient_label": orient_label.to(device, non_blocking=True),
        "scenes": scenes.to(device, non_blocking=True).contiguous(),
        "sample_ids": sample_ids,
        "kitchens": kitchens,
    }


def forward_and_losses(model, batch, config):
    predicted_joints, auxiliary = model(
        batch["joints_input"],
        batch["orient_input"],
        batch["scenes"],
        batch["objects"],
        batch["object_mask"],
    )
    target_joints = torch.cat(
        [batch["joints_input"], batch["joints_label"]],
        dim=1,
    )
    target_orientation = torch.cat(
        [batch["orient_input"], batch["orient_label"]],
        dim=1,
    )
    losses = direct29_losses(
        predicted_joints,
        target_joints,
        auxiliary["orientation"],
        target_orientation,
        input_frames=config.input_seq_len,
        root_weight=config.direct29_root_weight,
        pose_weight=config.direct29_pose_weight,
        yaw_weight=config.direct29_yaw_weight,
        mode=config.direct29_loss_mode,
        destination_weight=config.direct29_destination_weight,
    )
    return predicted_joints, target_joints, losses


def add_weighted(accum, values, weight):
    for key, value in values.items():
        accum[key] += float(value) * int(weight)


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    loss_accum = defaultdict(float)
    trajectory_accum = defaultdict(float)
    mpjpe_accum = defaultdict(float)
    samples = 0
    started = time.time()
    for raw_batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(raw_batch, device)
        predicted_joints, target_joints, losses = forward_and_losses(
            model,
            batch,
            config,
        )
        batch_size = int(batch["objects"].shape[0])
        add_weighted(loss_accum, losses, batch_size)
        horizons = compute_hik_horizon_metrics(
            predicted_joints,
            target_joints,
            input_seq_len=config.input_seq_len,
            output_seq_len=config.output_seq_len,
            fps=config.fps,
        )
        add_weighted(trajectory_accum, horizons["trajectory"], batch_size)
        add_weighted(mpjpe_accum, horizons["mpjpe"], batch_size)
        samples += batch_size
    return {
        "samples": samples,
        "joint_count": 29,
        "representation": "native_hik_poses3d_direct29",
        "loss": {key: value / samples for key, value in loss_accum.items()},
        "trajectory": {
            key: value / samples for key, value in trajectory_accum.items()
        },
        "mpjpe": {key: value / samples for key, value in mpjpe_accum.items()},
        "seconds": time.time() - started,
    }


def parameter_group(name):
    if name.startswith("trajectory_fc."):
        return "trajectory_head"
    if name.startswith("pose_fc."):
        return "pose_head"
    if name.startswith("motion_decoder."):
        return "gcn_decoder"
    return "shared_backbone"


def audit_component_gradients(model, loader, device, config, max_batches):
    model.eval()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    totals = {
        component: defaultdict(float)
        for component in ("root_objective", "pose_objective", "yaw")
    }
    batches = 0
    for raw_batch in loader:
        if batches >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        _, _, losses = forward_and_losses(model, batch, config)
        for component in ("root_objective", "pose_objective", "yaw"):
            gradients = torch.autograd.grad(
                losses[component],
                [parameter for _, parameter in named_parameters],
                retain_graph=True,
                allow_unused=True,
            )
            squared = defaultdict(float)
            for (name, _), gradient in zip(named_parameters, gradients):
                if gradient is not None:
                    squared[parameter_group(name)] += float(
                        gradient.detach().float().pow(2).sum().cpu()
                    )
            for group, value in squared.items():
                totals[component][group] += value**0.5
        batches += 1
    return {
        "batches": batches,
        "unweighted_component_gradient_l2": {
            component: {
                group: value / batches
                for group, value in sorted(groups.items())
            }
            for component, groups in totals.items()
        },
        "configured_weights": {
            "root": config.direct29_root_weight,
            "pose": config.direct29_pose_weight,
            "yaw": config.direct29_yaw_weight,
        },
        "loss_mode": config.direct29_loss_mode,
        "destination_weight": config.direct29_destination_weight,
    }


def load_direct_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Direct29 checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def evaluate_checkpoints(model, output, loader, device, config):
    results = {}
    for name in (
        "best_trajectory_mean.pth",
        "best_mpjpe_mean.pth",
        "best_total_loss.pth",
        "last.pth",
    ):
        path = output / name
        if not path.is_file():
            continue
        load_direct_checkpoint(model, path)
        results[name] = evaluate(model, loader, device, config)
    atomic_write_json(output / "final_test_metrics_by_checkpoint.json", results)
    return results


def main():
    config = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.early_stop_patience != 0:
        raise ValueError(
            "Direct29 uses a fixed epoch budget; set --early_stop_patience 0"
        )
    if config.use_pose_loss or config.use_bone_loss or config.use_dct_loss:
        raise ValueError(
            "Direct29 has an isolated non-overlapping root/pose/yaw loss. "
            "Do not pass legacy pose, bone, or DCT loss flags."
        )
    seed_everything(config.seed, config.deterministic)
    device = torch.device(f"cuda:{config.gpu}")
    torch.cuda.set_device(device)
    output = Path(config.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    model = SIF3DDirect29(config).to(device)
    initialization = None
    if config.load_model_dir:
        load_direct_checkpoint(model, config.load_model_dir)
        initialization = {"direct29_checkpoint": config.load_model_dir}
    elif config.warm_start_gimo:
        initialization = load_gimo_checkpoint_for_direct29(
            model,
            config.warm_start_gimo,
            map_location="cpu",
        )

    train_dataset = EgoEvalDataset(config, train=True)
    test_dataset = EgoEvalDataset(config, train=False)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        drop_last=False,
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    run_config = {
        **vars(config),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "initialization": initialization,
        "vposer_used": False,
        "smplx_used": False,
        "loss_terms": [
            (
                "future_root_mean_plus_destination"
                if config.direct29_loss_mode == "legacy_overlap"
                else "future_root_position_once"
            ),
            (
                "future_pose_mean_plus_destination"
                if config.direct29_loss_mode == "legacy_overlap"
                else "future_root_relative_pose_once"
            ),
            "future_wrapped_yaw_once",
        ],
    }
    atomic_write_json(output / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    history = []
    if config.eval_before_train or config.eval_only:
        baseline = evaluate(model, test_loader, device, config)
        baseline["epoch"] = -1
        history.append(baseline)
        atomic_write_json(output / "baseline_eval.json", baseline)
        print("baseline", json.dumps(baseline, indent=2), flush=True)
    if config.gradient_audit_batches > 0 and not config.eval_only:
        audit = audit_component_gradients(
            model,
            train_loader,
            device,
            config,
            config.gradient_audit_batches,
        )
        atomic_write_json(output / "gradient_audit_initial.json", audit)
        print("gradient_audit", json.dumps(audit, indent=2), flush=True)
    if config.eval_only:
        return

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.gamma,
    )
    best = {
        "trajectory": float("inf"),
        "mpjpe": float("inf"),
        "total": float("inf"),
    }
    for epoch in range(config.epoch):
        model.train()
        train_accum = defaultdict(float)
        samples = 0
        started = time.time()
        progress = tqdm(train_loader, desc=f"train {epoch:03d}")
        for step, raw_batch in enumerate(progress):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, _, losses = forward_and_losses(model, batch, config)
            losses["total"].backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
            optimizer.step()
            batch_size = int(batch["objects"].shape[0])
            add_weighted(train_accum, losses, batch_size)
            samples += batch_size
            if step % max(config.log_every, 1) == 0:
                progress.set_postfix(
                    total=f"{float(losses['total']):.4f}",
                    root=f"{float(losses['root']):.4f}",
                    pose=f"{float(losses['pose']):.4f}",
                )
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": {
                key: value / samples for key, value in train_accum.items()
            },
            "train_seconds": time.time() - started,
        }
        if epoch % config.val_fre == 0 or epoch == config.epoch - 1:
            validation = evaluate(model, test_loader, device, config)
            record["test"] = validation
            values = {
                "trajectory": validation["trajectory"]["mean"],
                "mpjpe": validation["mpjpe"]["mean"],
                "total": validation["loss"]["total"],
            }
            checkpoint_names = {
                "trajectory": "best_trajectory_mean.pth",
                "mpjpe": "best_mpjpe_mean.pth",
                "total": "best_total_loss.pth",
            }
            for metric, value in values.items():
                if value < best[metric]:
                    best[metric] = value
                    save_checkpoint(model, output / checkpoint_names[metric])
        history.append(record)
        atomic_write_json(output / "history.json", history)
        save_checkpoint(model, output / "last.pth")
        print(json.dumps(record, indent=2), flush=True)

    final = evaluate_checkpoints(model, output, test_loader, device, config)
    summary = {"best_values_during_training": best, "checkpoints": final}
    atomic_write_json(output / "final_summary.json", summary)
    print("final", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
