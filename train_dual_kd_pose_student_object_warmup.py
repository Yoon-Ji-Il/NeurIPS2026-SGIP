import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from train_dual_kd_common import run_kd_training_student_conditioned


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


def pose_warmup_kd_loss(config, student_out, teacher_out):
    future_slice = slice(config.input_seq_len, None)
    student_pose = student_out["pose"][:, future_slice]
    teacher_pose = teacher_out["pose"][:, future_slice]
    pose_kd = F.smooth_l1_loss(student_pose, teacher_pose)
    return kd_scale(config) * config.kd_pose_weight * pose_kd


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training_student_conditioned(
        config,
        "pose_student_object_warmup_kd",
        pose_warmup_kd_loss,
    )
