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


def pose_ori_warmup_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)

    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kd = F.smooth_l1_loss(student_pose, teacher_pose)

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
    run_kd_training(
        config,
        "pose_teacher_object_ori_warmup_kd",
        pose_ori_warmup_kd_loss,
    )
