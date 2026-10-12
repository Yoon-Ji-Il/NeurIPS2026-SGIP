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


def late_focused_loss(student_value, teacher_value, tail_boost):
    value_loss = F.smooth_l1_loss(student_value, teacher_value, reduction="none")
    time_steps = student_value.shape[1]
    time_weights = torch.linspace(
        0.7,
        1.3,
        steps=time_steps,
        device=student_value.device,
        dtype=student_value.dtype,
    )
    if time_steps >= len(tail_boost):
        time_weights[-len(tail_boost):] = time_weights[-len(tail_boost):] * torch.tensor(
            tail_boost,
            device=student_value.device,
            dtype=student_value.dtype,
        )
    return (value_loss * time_weights.view(1, time_steps, 1)).mean()


def normalized_mse_loss(student_value, teacher_value, eps):
    student_value = F.normalize(student_value, dim=-1, eps=eps)
    teacher_value = F.normalize(teacher_value, dim=-1, eps=eps)
    return F.mse_loss(student_value, teacher_value)


def cosine_sequence_loss(student_value, teacher_value, eps):
    return (1.0 - F.cosine_similarity(student_value, teacher_value, dim=-1, eps=eps)).mean()


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


def trans_pose_ori_warmup_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)

    student_trans = student_out["trans"][:, future_slice]
    teacher_trans = teacher_out["trans"][:, future_slice]
    traj_kd = late_focused_loss(student_trans, teacher_trans, [1.1, 1.3, 1.6])
    dest_trans_kd = F.smooth_l1_loss(student_trans[:, -1], teacher_trans[:, -1])

    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kd = pose_distill_loss(config, student_pose, teacher_pose)

    student_ori = student_out["ori"][:, future_slice]
    teacher_ori = teacher_out["ori"][:, future_slice]
    ori_kd = late_focused_loss(student_ori, teacher_ori, [1.2, 1.4, 1.8])
    dest_ori_kd = F.smooth_l1_loss(student_ori[:, -1], teacher_ori[:, -1])

    total_kd = (
        config.kd_traj_weight * traj_kd
        + config.kd_dest_weight * dest_trans_kd
        + config.kd_pose_weight * pose_kd
        + config.kd_ori_weight * ori_kd
        + config.kd_dest_ori_weight * dest_ori_kd
    )
    return kd_scale(config) * total_kd


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training(
        config,
        "trans_pose_teacher_object_ori_warmup_kd",
        trans_pose_ori_warmup_kd_loss,
    )
