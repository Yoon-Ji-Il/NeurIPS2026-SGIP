"""Non-overlapping direct-24 supervision and feature KD losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def direct24_losses(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    predicted_orientation: torch.Tensor,
    target_orientation: torch.Tensor,
    *,
    input_frames: int = 30,
    root_weight: float = 1.0,
    pose_weight: float = 1.0,
    yaw_weight: float = 0.1,
    use_pose_loss: bool = True,
) -> dict[str, torch.Tensor]:
    if predicted_joints.shape != target_joints.shape:
        raise ValueError(
            f"Joint shape mismatch: {predicted_joints.shape} vs "
            f"{target_joints.shape}"
        )
    if predicted_joints.ndim != 4 or predicted_joints.shape[2:] != (24, 3):
        raise ValueError(
            f"Expected joints [B,T,24,3], got {predicted_joints.shape}"
        )
    if predicted_orientation.shape != target_orientation.shape:
        raise ValueError("Orientation shape mismatch")
    future = slice(int(input_frames), None)
    predicted_future = predicted_joints[:, future]
    target_future = target_joints[:, future]
    root_per_frame = torch.linalg.vector_norm(
        predicted_future[:, :, 0] - target_future[:, :, 0],
        dim=-1,
    )
    predicted_relative = predicted_future - predicted_future[:, :, [0]]
    target_relative = target_future - target_future[:, :, [0]]
    pose_per_joint = torch.linalg.vector_norm(
        predicted_relative[:, :, 1:] - target_relative[:, :, 1:],
        dim=-1,
    )
    root = root_per_frame.mean()
    pose = pose_per_joint.mean()
    yaw_delta = (
        predicted_orientation[:, future, 1]
        - target_orientation[:, future, 1]
    )
    yaw = (1.0 - torch.cos(yaw_delta)).mean()
    pose_term = pose if use_pose_loss else pose.detach() * 0.0
    total = (
        float(root_weight) * root
        + float(pose_weight) * pose_term
        + float(yaw_weight) * yaw
    )
    return {
        "supervised_total": total,
        "root": root,
        "pose": pose,
        "yaw": yaw,
        "destination_root_diagnostic": root_per_frame[:, -1].mean(),
        "destination_pose_diagnostic": pose_per_joint[:, -1].mean(),
    }


def _feature_loss(student: torch.Tensor, teacher: torch.Tensor, kind: str):
    if kind == "smooth_l1":
        return F.smooth_l1_loss(student, teacher)
    if kind == "cosine":
        return (
            1.0
            - F.cosine_similarity(
                student.flatten(0, -2),
                teacher.flatten(0, -2),
                dim=-1,
            )
        ).mean()
    raise ValueError(f"Unknown KD feature kind: {kind}")


def direct24_kd_losses(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    config,
) -> dict[str, torch.Tensor]:
    zero = student["out_tia"].sum() * 0.0
    losses = {
        "kd_feat_out_tia": zero,
        "kd_feat_out_sca": zero,
        "kd_feat_object": zero,
        "kd_feat_motion_enc": zero,
        "kd_out_traj": zero,
        "kd_out_traj_dest": zero,
        "kd_out_pose": zero,
    }
    kind = config.kd_feat_kind
    if config.kd_feat_out_tia_weight > 0:
        losses["kd_feat_out_tia"] = _feature_loss(
            student["out_tia"], teacher["out_tia"], kind
        )
    if config.kd_feat_out_sca_weight > 0:
        losses["kd_feat_out_sca"] = _feature_loss(
            student["out_sca"], teacher["out_sca"], kind
        )
    if config.kd_feat_object_weight > 0:
        losses["kd_feat_object"] = _feature_loss(
            student["object_emb"], teacher["object_emb"], kind
        )
    if config.kd_feat_motion_enc_weight > 0:
        losses["kd_feat_motion_enc"] = _feature_loss(
            student["motion_enc"], teacher["motion_enc"], kind
        )
    student_traj = torch.cat(
        [student["orientation"], student["root"]],
        dim=-1,
    )
    teacher_traj = torch.cat(
        [teacher["orientation"], teacher["root"]],
        dim=-1,
    )
    if config.kd_out_traj_weight > 0:
        losses["kd_out_traj"] = F.smooth_l1_loss(
            student_traj, teacher_traj
        )
    if config.kd_out_traj_dest_weight > 0:
        losses["kd_out_traj_dest"] = F.smooth_l1_loss(
            student_traj[:, -1], teacher_traj[:, -1]
        )
    if config.kd_out_pose_weight > 0:
        losses["kd_out_pose"] = F.smooth_l1_loss(
            student["pre_gcn_joints"],
            teacher["pre_gcn_joints"],
        )
    total = (
        config.kd_feat_out_tia_weight * losses["kd_feat_out_tia"]
        + config.kd_feat_out_sca_weight * losses["kd_feat_out_sca"]
        + config.kd_feat_object_weight * losses["kd_feat_object"]
        + config.kd_feat_motion_enc_weight * losses["kd_feat_motion_enc"]
        + config.kd_out_traj_weight * losses["kd_out_traj"]
        + config.kd_out_traj_dest_weight * losses["kd_out_traj_dest"]
        + config.kd_out_pose_weight * losses["kd_out_pose"]
    )
    losses["kd_total"] = total
    return losses
