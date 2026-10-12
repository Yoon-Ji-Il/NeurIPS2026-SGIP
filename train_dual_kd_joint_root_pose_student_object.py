import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from train_dual_kd_common import run_kd_training_student_conditioned


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    warmup_epochs = getattr(config, "kd_warmup_epochs", 3)
    ramp_epochs = getattr(config, "kd_ramp_epochs", 5)
    max_scale = getattr(config, "kd_max_scale", 0.1)
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_scale
    ramp = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return max_scale * ramp


def time_weighted_smooth_l1(student_value, teacher_value, final_boost):
    loss = F.smooth_l1_loss(student_value, teacher_value, reduction="none").mean(dim=-1)
    if loss.shape[1] == 1:
        return loss.mean()
    weights = torch.linspace(
        1.0,
        final_boost,
        steps=loss.shape[1],
        device=loss.device,
        dtype=loss.dtype,
    )
    return (loss * weights.view(1, -1)).sum() / (loss.shape[0] * weights.sum())


def cosine_sequence_loss(student_value, teacher_value, eps):
    return (1.0 - F.cosine_similarity(student_value, teacher_value, dim=-1, eps=eps)).mean()


def normalized_mse_loss(student_value, teacher_value, eps):
    student_value = F.normalize(student_value, dim=-1, eps=eps)
    teacher_value = F.normalize(teacher_value, dim=-1, eps=eps)
    return F.mse_loss(student_value, teacher_value)


def pose_distill_loss(config, student_pose, teacher_pose):
    loss_type = getattr(config, "kd_pose_loss_type", "smooth_l1")
    eps = getattr(config, "kd_pose_loss_eps", 1.0e-6)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_pose, teacher_pose)
    if loss_type == "cosine":
        return cosine_sequence_loss(student_pose, teacher_pose, eps)
    if loss_type == "normalized_mse":
        return normalized_mse_loss(student_pose, teacher_pose, eps)
    raise ValueError(f"Unsupported kd_pose_loss_type: {loss_type}")


def relative_joint_features(joints):
    joints = joints - joints[:, :, [0]]
    joints = joints[:, :, 1:]
    return joints.reshape(joints.shape[0], joints.shape[1], -1)


def joint_root_pose_student_object_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)

    student_joints = student_out["joints"][:, future_slice]
    teacher_joints = teacher_out["joints"][:, future_slice]

    student_root = student_joints[:, :, 0]
    teacher_root = teacher_joints[:, :, 0]
    root_traj_kd = time_weighted_smooth_l1(student_root, teacher_root, final_boost=1.5)
    dest_root_kd = F.smooth_l1_loss(student_root[:, -1], teacher_root[:, -1])

    student_rel_joints = relative_joint_features(student_joints)
    teacher_rel_joints = relative_joint_features(teacher_joints)
    rel_joint_kd = time_weighted_smooth_l1(student_rel_joints, teacher_rel_joints, final_boost=1.5)
    dest_rel_joint_kd = F.smooth_l1_loss(student_rel_joints[:, -1], teacher_rel_joints[:, -1])

    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kd = pose_distill_loss(config, student_pose, teacher_pose)
    dest_pose_kd = pose_distill_loss(config, student_pose[:, -1:], teacher_pose[:, -1:])

    student_ori = student_out["ori"][:, future_slice]
    teacher_ori = teacher_out["ori"][:, future_slice]
    dest_ori_kd = F.smooth_l1_loss(student_ori[:, -1], teacher_ori[:, -1])

    total_kd = (
        config.kd_traj_weight * root_traj_kd
        + config.kd_dest_weight * dest_root_kd
        + config.kd_joints_weight * rel_joint_kd
        + config.kd_dest_joints_weight * dest_rel_joint_kd
        + config.kd_pose_weight * pose_kd
        + config.kd_dest_pose_weight * dest_pose_kd
        + config.kd_dest_ori_weight * dest_ori_kd
    )
    return kd_scale(config) * total_kd


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training_student_conditioned(
        config,
        "joint_root_pose_student_object_kd",
        joint_root_pose_student_object_kd_loss,
    )
