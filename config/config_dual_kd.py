from config.config_dual import MotionFromGazeDualConfig


class MotionFromGazeDualKDConfig(MotionFromGazeDualConfig):
    def __init__(self):
        super().__init__()

        kd_configs = self.add_argument_group("teacher_kd")
        # These options are already registered by the base configuration.
        self.set_defaults(
            teacher_load_model_dir="runs_dh/teacher_motion_c_final_seed7/best_loss_trans.pth",
            teacher_object_points_name="final_object_qwen3_14b_motion_c_final_points_1024.pth",
        )
        kd_configs.add_argument("--kd_traj_weight", type=float, default=1.0)
        kd_configs.add_argument("--kd_dest_weight", type=float, default=2.0)
        kd_configs.add_argument("--kd_pose_weight", type=float, default=0.5)
        kd_configs.add_argument("--kd_ori_weight", type=float, default=0.08)
        kd_configs.add_argument("--kd_dest_ori_weight", type=float, default=0.12)
        kd_configs.add_argument("--kd_joints_weight", type=float, default=0.25)
        kd_configs.add_argument("--kd_dest_joints_weight", type=float, default=0.5)
        kd_configs.add_argument("--kd_dest_pose_weight", type=float, default=0.25)
        kd_configs.add_argument("--kd_kl_temperature", type=float, default=2.0)
        kd_configs.add_argument(
            "--kd_pose_loss_type",
            type=str,
            choices=("smooth_l1", "cosine", "normalized_mse"),
            default="smooth_l1",
        )
        kd_configs.add_argument("--kd_pose_loss_eps", type=float, default=1.0e-6)
        kd_configs.add_argument("--kd_warmup_epochs", type=int, default=12)
        kd_configs.add_argument("--kd_ramp_epochs", type=int, default=8)
        kd_configs.add_argument("--kd_max_scale", type=float, default=1.0)
        kd_configs.add_argument("--kd_epoch_offset", type=int, default=0)
        kd_configs.add_argument("--kd_only", action="store_true")
