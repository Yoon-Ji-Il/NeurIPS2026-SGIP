from config.config_dual_kd import MotionFromGazeDualKDConfig
from train_dual_kd_common import run_kd_training_student_conditioned
from train_dual_kd_trans_pose_teacher_object_ori_warmup import trans_pose_ori_warmup_kd_loss


if __name__ == "__main__":
    config = MotionFromGazeDualKDConfig().parse_args()
    run_kd_training_student_conditioned(
        config,
        "trans_pose_student_object_ori_warmup_kd",
        trans_pose_ori_warmup_kd_loss,
    )
