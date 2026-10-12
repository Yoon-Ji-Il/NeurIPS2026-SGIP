import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from train_dual_kd_common import run_kd_training


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    warmup_epochs = getattr(config, "kd_warmup_epochs", 12)
    ramp_epochs = getattr(config, "kd_ramp_epochs", 8)
    max_scale = getattr(config, "kd_max_scale", 1.0)
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_scale
    ramp = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return max_scale * ramp


def cosine_sequence_loss(student_value, teacher_value, eps):
    return (1.0 - F.cosine_similarity(student_value, teacher_value, dim=-1, eps=eps)).mean()


def normalized_mse_loss(student_value, teacher_value, eps):
    student_value = F.normalize(student_value, dim=-1, eps=eps)
    teacher_value = F.normalize(teacher_value, dim=-1, eps=eps)
    return F.mse_loss(student_value, teacher_value)


def pose_distill_loss(config, student_pose, teacher_pose):
    loss_type = getattr(config, "kd_pose_loss_type", "cosine")
    eps = getattr(config, "kd_pose_loss_eps", 1.0e-6)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_pose, teacher_pose)
    if loss_type == "cosine":
        return cosine_sequence_loss(student_pose, teacher_pose, eps)
    if loss_type == "normalized_mse":
        return normalized_mse_loss(student_pose, teacher_pose, eps)
    raise ValueError(f"Unsupported kd_pose_loss_type: {loss_type}")


def root_relative_joint_features(joints):
    joints = joints - joints[:, :, [0]]
    joints = joints[:, :, 1:]
    return joints.reshape(joints.shape[0], joints.shape[1], -1)


def joints_pose_cosine_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)
    eps = getattr(config, "kd_pose_loss_eps", 1.0e-6)

    student_joints = root_relative_joint_features(student_out["joints"][:, future_slice])
    teacher_joints = root_relative_joint_features(teacher_out["joints"][:, future_slice])
    joints_kd = cosine_sequence_loss(student_joints, teacher_joints, eps)
    dest_joints_kd = cosine_sequence_loss(student_joints[:, -1:], teacher_joints[:, -1:], eps)

    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kd = pose_distill_loss(config, student_pose, teacher_pose)
    dest_pose_kd = pose_distill_loss(config, student_pose[:, -1:], teacher_pose[:, -1:])

    total_kd = (
        config.kd_joints_weight * joints_kd
        + config.kd_dest_joints_weight * dest_joints_kd
        + config.kd_pose_weight * pose_kd
        + config.kd_dest_pose_weight * dest_pose_kd
    )
    return kd_scale(config) * total_kd


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training(
        config,
        "joints_pose_cosine_teacher_object_kd",
        joints_pose_cosine_kd_loss,
    )
