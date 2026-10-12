"""Non-overlapping losses for direct HIK root and root-relative pose."""

from __future__ import annotations

import torch


def direct29_losses(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    predicted_orientation: torch.Tensor,
    target_orientation: torch.Tensor,
    *,
    input_frames: int = 25,
    root_weight: float = 1.0,
    pose_weight: float = 1.0,
    yaw_weight: float = 0.1,
    mode: str = "disjoint",
    destination_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Supervise each semantic quantity exactly once on future frames.

    Root position and root-relative articulation are disjoint by construction.
    Wrapped yaw is the only auxiliary term and exists solely because SIF3D uses
    predicted yaw to normalize the scene before pose decoding.
    """

    if predicted_joints.shape != target_joints.shape:
        raise ValueError(
            f"Joint shape mismatch: {predicted_joints.shape} vs "
            f"{target_joints.shape}"
        )
    if predicted_joints.ndim != 4 or predicted_joints.shape[2:] != (29, 3):
        raise ValueError(
            f"Expected joints [B,T,29,3], got {predicted_joints.shape}"
        )
    if predicted_orientation.shape != target_orientation.shape:
        raise ValueError(
            "Orientation shape mismatch: "
            f"{predicted_orientation.shape} vs {target_orientation.shape}"
        )
    future = slice(int(input_frames), None)
    predicted_future = predicted_joints[:, future]
    target_future = target_joints[:, future]

    if mode not in {"disjoint", "legacy_overlap"}:
        raise ValueError(f"Unknown direct29 loss mode: {mode}")
    root_per_frame = torch.linalg.vector_norm(
        predicted_future[:, :, 0] - target_future[:, :, 0],
        dim=-1,
    )
    predicted_relative = (
        predicted_future - predicted_future[:, :, [0]]
    )
    target_relative = target_future - target_future[:, :, [0]]
    pose_per_joint = torch.linalg.vector_norm(
        predicted_relative[:, :, 1:] - target_relative[:, :, 1:],
        dim=-1,
    )
    root = root_per_frame.mean()
    pose = pose_per_joint.mean()
    destination_root = root_per_frame[:, -1].mean()
    destination_pose = pose_per_joint[:, -1].mean()
    yaw_delta = (
        predicted_orientation[:, future, 1]
        - target_orientation[:, future, 1]
    )
    yaw = (1.0 - torch.cos(yaw_delta)).mean()
    if mode == "legacy_overlap":
        root_objective = root + float(destination_weight) * destination_root
        pose_objective = pose + float(destination_weight) * destination_pose
    else:
        root_objective = root
        pose_objective = pose
    total = (
        float(root_weight) * root_objective
        + float(pose_weight) * pose_objective
        + float(yaw_weight) * yaw
    )
    return {
        "total": total,
        "root": root,
        "pose": pose,
        "yaw": yaw,
        "destination_root_diagnostic": destination_root,
        "destination_pose_diagnostic": destination_pose,
        "root_objective": root_objective,
        "pose_objective": pose_objective,
    }
