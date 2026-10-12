#!/usr/bin/env python3
"""Train or distill SIF3D direct24 on HOI-M3 Entity HIK-scale."""

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

from config.config_hoim3_direct24_1s2s import HOIM3Direct24Config
from dataset.hoim3_dataset_direct24_1s2s import EgoEvalDataset
from model.sif3d_hoim3_direct24_1s2s import (
    SIF3DDirect24,
    load_gimo_checkpoint_for_direct24,
)
from utils.hik_metrics import compute_hik_horizon_metrics
from utils.hoim3_direct24_losses import (
    direct24_kd_losses,
    direct24_losses,
)


def make_parser():
    parser = HOIM3Direct24Config()
    parser.add_argument("--gpu", type=int, default=0)
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
    return parser


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)


def load_direct24_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Direct24 checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def move_batch(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    output["scenes"] = output["scenes"].contiguous()
    return output


def forward_and_losses(
    model: SIF3DDirect24,
    batch: dict,
    config,
    teacher: SIF3DDirect24 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    predicted, student_features = model(
        batch["joints_input"],
        batch["orient_input"],
        batch["scenes"],
        batch["objects"],
        batch["object_mask"],
        return_features=True,
    )
    target_joints = torch.cat(
        [batch["joints_input"], batch["joints_label"]],
        dim=1,
    )
    target_orientation = torch.cat(
        [batch["orient_input"], batch["orient_label"]],
        dim=1,
    )
    losses = direct24_losses(
        predicted,
        target_joints,
        student_features["orientation"],
        target_orientation,
        input_frames=config.input_seq_len,
        root_weight=config.direct24_root_weight,
        pose_weight=config.direct24_pose_weight,
        yaw_weight=config.direct24_yaw_weight,
        use_pose_loss=config.use_pose_loss,
    )
    total = losses["supervised_total"]
    if config.use_kd:
        if teacher is None:
            raise RuntimeError("KD is enabled but no teacher model was provided")
        with torch.no_grad():
            _, teacher_features = teacher(
                batch["joints_input"],
                batch["orient_input"],
                batch["scenes"],
                batch["teacher_objects"],
                batch["teacher_object_mask"],
                return_features=True,
            )
        kd = direct24_kd_losses(student_features, teacher_features, config)
        losses.update(kd)
        total = total + kd["kd_total"]
    losses["total"] = total
    return predicted, target_joints, losses


def add_weighted(
    accumulator: dict[str, float],
    values: dict,
    weight: int,
) -> None:
    for key, value in values.items():
        accumulator[key] += float(value) * int(weight)


@torch.no_grad()
def evaluate(
    model: SIF3DDirect24,
    loader: DataLoader,
    device: torch.device,
    config,
    teacher: SIF3DDirect24 | None = None,
) -> dict:
    model.eval()
    if teacher is not None:
        teacher.eval()
    loss_accum = defaultdict(float)
    trajectory_accum = defaultdict(float)
    mpjpe_accum = defaultdict(float)
    samples = 0
    started = time.time()
    for raw_batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(raw_batch, device)
        predicted, target, losses = forward_and_losses(
            model,
            batch,
            config,
            teacher,
        )
        batch_size = int(batch["objects"].shape[0])
        add_weighted(loss_accum, losses, batch_size)
        metrics = compute_hik_horizon_metrics(
            predicted,
            target,
            input_seq_len=config.input_seq_len,
            output_seq_len=config.output_seq_len,
            fps=config.fps,
        )
        add_weighted(
            trajectory_accum,
            metrics["trajectory"],
            batch_size,
        )
        add_weighted(mpjpe_accum, metrics["mpjpe"], batch_size)
        samples += batch_size
    return {
        "samples": samples,
        "protocol": config.hoim3_protocol,
        "joint_count": 24,
        "representation": "native_hoim3_direct24_no_smplx",
        "units": "metres",
        "loss": {
            key: value / samples for key, value in loss_accum.items()
        },
        "trajectory": {
            key: value / samples
            for key, value in trajectory_accum.items()
        },
        "mpjpe": {
            key: value / samples for key, value in mpjpe_accum.items()
        },
        "seconds": time.time() - started,
    }


def evaluate_checkpoints(
    model: SIF3DDirect24,
    output: Path,
    loader: DataLoader,
    device: torch.device,
    config,
    teacher: SIF3DDirect24 | None,
) -> dict:
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
        load_direct24_checkpoint(model, path)
        results[name] = evaluate(
            model,
            loader,
            device,
            config,
            teacher,
        )
    atomic_json(output / "final_test_metrics_by_checkpoint.json", results)
    return results


def main() -> None:
    config = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not config.use_pose_loss:
        raise ValueError(
            "Pass --use_pose_loss. In direct24 this means native root-relative "
            "joint supervision; VPoser and SMPL-X are not used."
        )
    if config.use_bone_loss or config.use_dct_loss:
        raise ValueError("Bone/DCT losses are outside the fixed direct24 experiment")
    if config.use_kd and not config.teacher_load_model_dir:
        raise ValueError("--teacher_load_model_dir is required with --use_kd")

    seed_everything(config.seed, config.deterministic)
    device = torch.device(f"cuda:{config.gpu}")
    torch.cuda.set_device(device)
    output = Path(config.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    model = SIF3DDirect24(config).to(device)
    initialization = None
    if config.load_model_dir:
        load_direct24_checkpoint(model, config.load_model_dir)
        initialization = {"direct24_checkpoint": config.load_model_dir}
    elif config.warm_start_gimo:
        initialization = load_gimo_checkpoint_for_direct24(
            model,
            config.warm_start_gimo,
            map_location="cpu",
        )

    teacher = None
    if config.use_kd:
        teacher = SIF3DDirect24(config).to(device)
        load_direct24_checkpoint(teacher, config.teacher_load_model_dir)
        teacher.requires_grad_(False)
        teacher.eval()

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
        "native_joint_count": 24,
        "scene_points": 1000,
        "object_points": 1024,
        "loss_design": {
            "root": "future root position exactly once",
            "pose": "future root-relative 23 non-root joints exactly once",
            "yaw": "wrapped y-axis orientation auxiliary",
            "legacy_use_pose_loss_mapping": (
                "--use_pose_loss enables native direct24 root-relative MPJPE"
            ),
        },
        "kd_design": {
            "enabled": bool(config.use_kd),
            "teacher_checkpoint": config.teacher_load_model_dir,
            "teacher_object_points": config.teacher_object_points_name,
            "student_object_points": config.object_points_name,
        },
    }
    atomic_json(output / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    history = []
    if config.eval_before_train or config.eval_only:
        baseline = evaluate(
            model,
            test_loader,
            device,
            config,
            teacher,
        )
        baseline["epoch"] = -1
        history.append(baseline)
        atomic_json(output / "baseline_eval.json", baseline)
        print("baseline", json.dumps(baseline, indent=2), flush=True)
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
        if teacher is not None:
            teacher.eval()
        train_accum = defaultdict(float)
        samples = 0
        started = time.time()
        progress = tqdm(train_loader, desc=f"train {epoch:03d}")
        for step, raw_batch in enumerate(progress):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, _, losses = forward_and_losses(
                model,
                batch,
                config,
                teacher,
            )
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
                postfix = {
                    "total": f"{float(losses['total']):.4f}",
                    "root": f"{float(losses['root']):.4f}",
                    "pose": f"{float(losses['pose']):.4f}",
                }
                if config.use_kd:
                    postfix["kd"] = f"{float(losses['kd_total']):.4f}"
                progress.set_postfix(**postfix)
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": {
                key: value / samples
                for key, value in train_accum.items()
            },
            "train_seconds": time.time() - started,
        }
        if epoch % config.val_fre == 0 or epoch == config.epoch - 1:
            validation = evaluate(
                model,
                test_loader,
                device,
                config,
                teacher,
            )
            record["test"] = validation
            values = {
                "trajectory": validation["trajectory"]["mean"],
                "mpjpe": validation["mpjpe"]["mean"],
                "total": validation["loss"]["total"],
            }
            names = {
                "trajectory": "best_trajectory_mean.pth",
                "mpjpe": "best_mpjpe_mean.pth",
                "total": "best_total_loss.pth",
            }
            for metric, value in values.items():
                if value < best[metric]:
                    best[metric] = value
                    save_checkpoint(model, output / names[metric])
        history.append(record)
        atomic_json(output / "history.json", history)
        save_checkpoint(model, output / "last.pth")
        print(json.dumps(record, indent=2), flush=True)

    final = evaluate_checkpoints(
        model,
        output,
        test_loader,
        device,
        config,
        teacher,
    )
    summary = {
        "best_values_during_training": best,
        "checkpoints": final,
    }
    atomic_json(output / "final_summary.json", summary)
    print("final", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
