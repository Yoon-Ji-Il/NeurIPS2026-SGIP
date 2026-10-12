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


def kl_from_teacher(student_logits, teacher_logits, temperature):
    tau = max(float(temperature), 1.0e-6)
    student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return kl.mean() * (tau ** 2)


def joints_pose_kl_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)
    temperature = getattr(config, "kd_kl_temperature", 2.0)

    student_joints = student_out["joints"][:, future_slice].reshape(student_out["joints"].shape[0], -1, 23 * 3)
    teacher_joints = teacher_out["joints"][:, future_slice].reshape(teacher_out["joints"].shape[0], -1, 23 * 3)
    joints_kl = kl_from_teacher(student_joints, teacher_joints, temperature)
    dest_joints_kl = kl_from_teacher(student_joints[:, -1:], teacher_joints[:, -1:], temperature)

    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kl = kl_from_teacher(student_pose, teacher_pose, temperature)
    dest_pose_kl = kl_from_teacher(student_pose[:, -1:], teacher_pose[:, -1:], temperature)

    total_kd = (
        config.kd_joints_weight * joints_kl
        + config.kd_dest_joints_weight * dest_joints_kl
        + config.kd_pose_weight * pose_kl
        + config.kd_dest_pose_weight * dest_pose_kl
    )
    return kd_scale(config) * total_kd


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training(
        config,
        "kl_joints_pose_teacher_object_kd",
        joints_pose_kl_kd_loss,
    )
