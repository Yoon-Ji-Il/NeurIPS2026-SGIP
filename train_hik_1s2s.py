"""Train the isolated SIF3D HIK model on native 25-observed/50-future data."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import smplx
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_hik_1s2s import HIK1s2sConfig
from dataset.hik_dataset_dual_1s2s import EgoEvalDataset
from human_body_prior.tools.model_loader import load_vposer
from model.sif3d_hik_1s2s import SIF3D, load_gimo_checkpoint_for_hik
from utils.hik_metrics import compute_hik_horizon_metrics


def make_parser() -> HIK1s2sConfig:
    parser = HIK1s2sConfig()
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


def make_body_model(config, device: torch.device):
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
        batch_size=config.batch_size * 75,
    ).to(device)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def supervised_losses(
    predicted_motion: torch.Tensor,
    predicted_joints: torch.Tensor,
    poses_input: torch.Tensor,
    poses_label: torch.Tensor,
    joints_input: torch.Tensor,
    joints_label: torch.Tensor,
    input_frames: int,
    des_trans_weight: float,
    mpjpe_weight: float,
) -> dict[str, torch.Tensor]:
    target_motion = torch.cat([poses_input, poses_label], dim=1)
    target_joints = torch.cat([joints_input, joints_label], dim=1)
    future = slice(input_frames, None)
    motion = F.smooth_l1_loss(
        predicted_motion[:, future],
        target_motion[:, future],
    )
    trajectory_per_frame = torch.linalg.vector_norm(
        predicted_joints[:, :, 0] - target_joints[:, :, 0],
        dim=-1,
    )
    predicted_relative = predicted_joints - predicted_joints[:, :, [0]]
    target_relative = target_joints - target_joints[:, :, [0]]
    mpjpe_per_frame = torch.linalg.vector_norm(
        predicted_relative - target_relative,
        dim=-1,
    ).mean(dim=-1)
    trajectory = trajectory_per_frame[:, future].mean()
    destination_trajectory = trajectory_per_frame[:, -1].mean()
    mpjpe = mpjpe_per_frame[:, future].mean()
    destination_mpjpe = mpjpe_per_frame[:, -1].mean()
    total = (
        motion
        + trajectory
        + float(des_trans_weight) * destination_trajectory
        + float(mpjpe_weight) * (mpjpe + destination_mpjpe)
    )
    return {
        "total": total,
        "motion": motion,
        "trajectory": trajectory,
        "destination_trajectory": destination_trajectory,
        "mpjpe": mpjpe,
        "destination_mpjpe": destination_mpjpe,
    }


def add_weighted(
    accum: dict[str, float],
    values: dict[str, float],
    weight: int,
) -> None:
    for key, value in values.items():
        accum[key] += float(value) * int(weight)


@torch.no_grad()
def evaluate(model, loader, device, config) -> dict:
    model.eval()
    loss_accum = defaultdict(float)
    trajectory_accum = defaultdict(float)
    mpjpe_accum = defaultdict(float)
    samples = 0
    started = time.time()
    for batch in tqdm(loader, desc="eval", leave=False):
        (
            objects,
            object_mask,
            gt_prior,
            gt_prior_mask,
            poses_input,
            poses_label,
            joints_input,
            joints_label,
            scenes,
            _,
            _,
        ) = batch
        objects = objects.to(device)
        object_mask = object_mask.to(device)
        poses_input = poses_input.to(device)
        poses_label = poses_label.to(device)
        joints_input = joints_input.to(device)
        joints_label = joints_label.to(device)
        scenes = scenes.to(device).contiguous()
        predicted_motion, predicted_joints = model(
            poses_input,
            joints_input,
            scenes,
            objects,
            object_mask,
            gt_prior.to(device),
            gt_prior_mask.to(device),
        )
        losses = supervised_losses(
            predicted_motion,
            predicted_joints,
            poses_input,
            poses_label,
            joints_input,
            joints_label,
            config.input_seq_len,
            config.des_trans_weight,
            config.mpjpe_weight,
        )
        batch_size = int(objects.shape[0])
        add_weighted(
            loss_accum,
            {key: value.detach().cpu() for key, value in losses.items()},
            batch_size,
        )
        target_joints = torch.cat([joints_input, joints_label], dim=1)
        horizons = compute_hik_horizon_metrics(
            predicted_joints,
            target_joints,
            input_seq_len=config.input_seq_len,
            output_seq_len=config.output_seq_len,
            fps=config.fps,
        )
        add_weighted(
            trajectory_accum,
            horizons["trajectory"],
            batch_size,
        )
        add_weighted(mpjpe_accum, horizons["mpjpe"], batch_size)
        samples += batch_size
    return {
        "samples": samples,
        "loss": {key: value / samples for key, value in loss_accum.items()},
        "trajectory": {
            key: value / samples for key, value in trajectory_accum.items()
        },
        "mpjpe": {key: value / samples for key, value in mpjpe_accum.items()},
        "seconds": time.time() - started,
    }


def save_checkpoint(model, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    config = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(config.seed, config.deterministic)
    device = torch.device(f"cuda:{config.gpu}")
    torch.cuda.set_device(device)
    output = Path(config.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    vposer, _ = load_vposer(config.vposer_path, vp_model="snapshot")
    vposer = vposer.to(device)
    body_model = make_body_model(config, device)
    model = SIF3D(config, vposer, body_model).to(device)
    initialization = None
    if config.load_model_dir:
        state = torch.load(
            config.load_model_dir,
            map_location="cpu",
            weights_only=False,
        )
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("smplx_model.")
        }
        incompatible = model.load_state_dict(state, strict=False)
        initialization = {
            "hik_checkpoint": str(config.load_model_dir),
            "missing": len(incompatible.missing_keys),
            "unexpected": len(incompatible.unexpected_keys),
        }
    elif config.warm_start_gimo:
        initialization = load_gimo_checkpoint_for_hik(
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
    }
    write_json(output / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    history = []
    if config.eval_before_train or config.eval_only:
        baseline = evaluate(model, test_loader, device, config)
        baseline["epoch"] = -1
        history.append(baseline)
        write_json(output / "baseline_eval.json", baseline)
        print("baseline", json.dumps(baseline, indent=2), flush=True)
    if config.eval_only:
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.gamma,
    )
    best = float("inf")
    best_mpjpe = float("inf")
    best_total_loss = float("inf")
    bad_epochs = 0
    for epoch in range(config.epoch):
        model.train()
        train_accum = defaultdict(float)
        samples = 0
        started = time.time()
        progress = tqdm(train_loader, desc=f"train {epoch:03d}")
        for step, batch in enumerate(progress):
            (
                objects,
                object_mask,
                gt_prior,
                gt_prior_mask,
                poses_input,
                poses_label,
                joints_input,
                joints_label,
                scenes,
                _,
                _,
            ) = batch
            objects = objects.to(device, non_blocking=True)
            object_mask = object_mask.to(device, non_blocking=True)
            poses_input = poses_input.to(device, non_blocking=True)
            poses_label = poses_label.to(device, non_blocking=True)
            joints_input = joints_input.to(device, non_blocking=True)
            joints_label = joints_label.to(device, non_blocking=True)
            scenes = scenes.to(device, non_blocking=True).contiguous()
            optimizer.zero_grad(set_to_none=True)
            predicted_motion, predicted_joints = model(
                poses_input,
                joints_input,
                scenes,
                objects,
                object_mask,
                gt_prior.to(device, non_blocking=True),
                gt_prior_mask.to(device, non_blocking=True),
            )
            losses = supervised_losses(
                predicted_motion,
                predicted_joints,
                poses_input,
                poses_label,
                joints_input,
                joints_label,
                config.input_seq_len,
                config.des_trans_weight,
                config.mpjpe_weight,
            )
            losses["total"].backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
            optimizer.step()
            batch_size = int(objects.shape[0])
            add_weighted(
                train_accum,
                {
                    key: value.detach().cpu()
                    for key, value in losses.items()
                },
                batch_size,
            )
            samples += batch_size
            if step % max(config.log_every, 1) == 0:
                progress.set_postfix(
                    loss=f"{float(losses['total']):.4f}",
                    traj=f"{float(losses['trajectory']):.4f}",
                    mpjpe=f"{float(losses['mpjpe']):.4f}",
                )
        scheduler.step()
        train_result = {
            key: value / samples for key, value in train_accum.items()
        }
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_result,
            "train_seconds": time.time() - started,
        }
        if epoch % config.val_fre == 0 or epoch == config.epoch - 1:
            validation = evaluate(model, test_loader, device, config)
            record["test"] = validation
            monitored = validation["trajectory"]["mean"]
            monitored_mpjpe = validation["mpjpe"]["mean"]
            monitored_total_loss = validation["loss"]["total"]
            if monitored < best - config.early_stop_min_delta:
                best = monitored
                bad_epochs = 0
                save_checkpoint(model, output / "best_trajectory_mean.pth")
            else:
                bad_epochs += 1
            if monitored_mpjpe < best_mpjpe:
                best_mpjpe = monitored_mpjpe
                save_checkpoint(model, output / "best_mpjpe_mean.pth")
            if monitored_total_loss < best_total_loss:
                best_total_loss = monitored_total_loss
                save_checkpoint(model, output / "best_total_loss.pth")
        history.append(record)
        write_json(output / "history.json", history)
        save_checkpoint(model, output / "last.pth")
        print(json.dumps(record, indent=2), flush=True)
        if (
            config.early_stop_patience > 0
            and epoch >= config.early_stop_warmup
            and bad_epochs >= config.early_stop_patience
        ):
            print(
                f"early stop at epoch={epoch}, best_trajectory_mean={best}",
                flush=True,
            )
            break

    best_path = output / "best_trajectory_mean.pth"
    if best_path.is_file():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("smplx_model.")
        }
        model.load_state_dict(state, strict=False)
    final = evaluate(model, test_loader, device, config)
    final["best_trajectory_mean"] = best
    final["best_mpjpe_mean"] = best_mpjpe
    final["best_total_loss"] = best_total_loss
    write_json(output / "final_test_metrics.json", final)
    print("final", json.dumps(final, indent=2), flush=True)

    for checkpoint_name, result_name, selected_value in (
        (
            "best_mpjpe_mean.pth",
            "final_test_metrics_best_mpjpe.json",
            best_mpjpe,
        ),
        (
            "best_total_loss.pth",
            "final_test_metrics_best_total_loss.json",
            best_total_loss,
        ),
    ):
        checkpoint_path = output / checkpoint_name
        if not checkpoint_path.is_file():
            continue
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("smplx_model.")
        }
        model.load_state_dict(state, strict=False)
        selected = evaluate(model, test_loader, device, config)
        selected["selected_checkpoint"] = checkpoint_name
        selected["selected_value"] = selected_value
        write_json(output / result_name, selected)
        print(
            result_name,
            json.dumps(selected, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
