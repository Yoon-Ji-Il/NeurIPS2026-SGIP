"""HIK H25F50 trajectory and pose metrics at fixed time horizons."""

from __future__ import annotations

import math
from typing import Iterable

import torch


DEFAULT_HORIZONS_SECONDS = (0.5, 1.0, 1.5, 2.0)


def _future_only(
    joints: torch.Tensor,
    input_seq_len: int,
    output_seq_len: int,
) -> torch.Tensor:
    if joints.ndim != 4 or joints.shape[-1] != 3:
        raise ValueError(
            "Expected joints with shape [B, T, J, 3], "
            f"got {tuple(joints.shape)}"
        )
    if joints.shape[1] == input_seq_len + output_seq_len:
        return joints[:, input_seq_len:]
    if joints.shape[1] == output_seq_len:
        return joints
    raise ValueError(
        f"Expected T={input_seq_len + output_seq_len} (full) or "
        f"T={output_seq_len} (future only), got T={joints.shape[1]}"
    )


def _interpolate_future_at_seconds(
    future: torch.Tensor,
    seconds: float,
    fps: int,
) -> tuple[torch.Tensor, dict]:
    """Interpolate a future sequence at an exact time after observation.

    Future frame zero occurs one frame interval after the last observation.
    At 25 Hz, 0.5 seconds is therefore future-frame position 11.5 (zero
    indexed), halfway between the 12th and 13th future frames.
    """

    position = float(seconds) * int(fps) - 1.0
    if position < 0.0 or position > future.shape[1] - 1:
        raise ValueError(
            f"Horizon {seconds}s is outside a {future.shape[1] / fps:.3f}s "
            "future sequence"
        )
    low = int(math.floor(position))
    high = int(math.ceil(position))
    alpha = float(position - low)
    value = future[:, low]
    if high != low:
        value = value * (1.0 - alpha) + future[:, high] * alpha
    info = {
        "future_position_zero_based": position,
        "lower_frame_one_based": low + 1,
        "upper_frame_one_based": high + 1,
        "upper_weight": alpha,
    }
    return value, info


def compute_hik_horizon_metrics(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    *,
    input_seq_len: int = 25,
    output_seq_len: int = 50,
    fps: int = 25,
    horizons_seconds: Iterable[float] = DEFAULT_HORIZONS_SECONDS,
) -> dict:
    """Compute root trajectory error and root-relative MPJPE in metres.

    ``trajectory`` is the Euclidean error of joint zero. ``mpjpe`` first
    subtracts joint zero independently from prediction and target, matching
    the existing SIF3D ``calc_loss_gcn`` definition.
    """

    if predicted_joints.shape != target_joints.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{tuple(predicted_joints.shape)} vs "
            f"{tuple(target_joints.shape)}"
        )
    predicted = _future_only(
        predicted_joints,
        input_seq_len,
        output_seq_len,
    )
    target = _future_only(
        target_joints,
        input_seq_len,
        output_seq_len,
    )

    trajectory_per_frame = torch.linalg.vector_norm(
        predicted[:, :, 0] - target[:, :, 0],
        dim=-1,
    )
    predicted_relative = predicted - predicted[:, :, [0]]
    target_relative = target - target[:, :, [0]]
    mpjpe_per_frame = torch.linalg.vector_norm(
        predicted_relative - target_relative,
        dim=-1,
    ).mean(dim=-1)

    trajectory = {"mean": float(trajectory_per_frame.mean().detach().cpu())}
    mpjpe = {"mean": float(mpjpe_per_frame.mean().detach().cpu())}
    interpolation = {}
    for seconds in horizons_seconds:
        key = f"{float(seconds):.1f}s"
        predicted_at, interpolation_info = _interpolate_future_at_seconds(
            predicted,
            float(seconds),
            fps,
        )
        target_at, _ = _interpolate_future_at_seconds(
            target,
            float(seconds),
            fps,
        )
        trajectory_at = torch.linalg.vector_norm(
            predicted_at[:, 0] - target_at[:, 0],
            dim=-1,
        ).mean()
        predicted_relative_at = predicted_at - predicted_at[:, [0]]
        target_relative_at = target_at - target_at[:, [0]]
        mpjpe_at = torch.linalg.vector_norm(
            predicted_relative_at - target_relative_at,
            dim=-1,
        ).mean()
        trajectory[key] = float(trajectory_at.detach().cpu())
        mpjpe[key] = float(mpjpe_at.detach().cpu())
        interpolation[key] = interpolation_info

    return {
        "units": "metres",
        "fps": int(fps),
        "input_frames": int(input_seq_len),
        "future_frames": int(output_seq_len),
        "trajectory": trajectory,
        "mpjpe": mpjpe,
        "interpolation": interpolation,
    }
