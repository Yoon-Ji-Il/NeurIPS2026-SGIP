import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from train_dual_kd_common import run_kd_training_student_conditioned


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    warmup_epochs = getattr(config, "kd_warmup_epochs", 0)
    ramp_epochs = getattr(config, "kd_ramp_epochs", 5)
    max_scale = getattr(config, "kd_max_scale", 1.0)
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_scale
    ramp = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return max_scale * ramp


def late_focused_ori_loss(student_ori, teacher_ori):
    ori_loss = F.smooth_l1_loss(student_ori, teacher_ori, reduction="none")
    time_steps = student_ori.shape[1]
    time_weights = torch.linspace(
        0.7,
        1.3,
        steps=time_steps,
        device=student_ori.device,
        dtype=student_ori.dtype,
    )
    if time_steps >= 3:
        time_weights[-3:] = time_weights[-3:] * torch.tensor(
            [1.2, 1.4, 1.8],
            device=student_ori.device,
            dtype=student_ori.dtype,
        )
    return (ori_loss * time_weights.view(1, time_steps, 1)).mean()


def pose_loss(config, student_pose, teacher_pose):
    loss_type = getattr(config, "kd_pose_loss_type", "smooth_l1")
    eps = getattr(config, "kd_pose_loss_eps", 1.0e-6)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_pose, teacher_pose)
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(student_pose, teacher_pose, dim=-1, eps=eps)).mean()
    if loss_type == "normalized_mse":
        student_pose = F.normalize(student_pose, dim=-1, eps=eps)
        teacher_pose = F.normalize(teacher_pose, dim=-1, eps=eps)
        return F.mse_loss(student_pose, teacher_pose)
    raise ValueError(f"Unsupported kd_pose_loss_type: {loss_type}")


def pose_ori_student_object_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)

    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kd = pose_loss(config, student_pose, teacher_pose)

    student_ori = student_out["ori"][:, future_slice]
    teacher_ori = teacher_out["ori"][:, future_slice]
    ori_kd = late_focused_ori_loss(student_ori, teacher_ori)
    dest_ori_kd = F.smooth_l1_loss(student_ori[:, -1], teacher_ori[:, -1])

    total_kd = (
        config.kd_pose_weight * pose_kd
        + config.kd_ori_weight * ori_kd
        + config.kd_dest_ori_weight * dest_ori_kd
    )
    return kd_scale(config) * total_kd


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training_student_conditioned(
        config,
        "pose_ori_student_object_configurable_kd",
        pose_ori_student_object_kd_loss,
    )
