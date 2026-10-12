"""Isolated configuration for HOI-M3 HIK-scale direct-24 forecasting."""

from __future__ import annotations

from .config_dual import MotionFromGazeDualConfig


class HOIM3Direct24Config(MotionFromGazeDualConfig):
    INPUT_FRAMES = 30
    FUTURE_FRAMES = 60
    TARGET_FPS = 30
    N_JOINTS = 24

    def __init__(self):
        super().__init__()
        self.set_defaults(
            batch_size=16,
            num_workers=4,
            input_seq_len=self.INPUT_FRAMES,
            output_seq_len=self.FUTURE_FRAMES,
            fps=self.TARGET_FPS,
            sample_points=1000,
            dataroot=(
                "data/HOIm3/"
                "ours_hikscale_direct24_h30f60"
            ),
            object_points_name=(
                "final_object_student_hoim3_all_objects_no_cone_"
                "points_1024.pth"
            ),
            save_path="runs_hoim3/hoim3_hikscale_direct24_seed7",
            epoch=30,
            lr=1.0e-4,
            gamma=0.98,
            early_stop_patience=0,
        )
        self.input_configs.add_argument(
            "--hoim3_protocol",
            default="hoim3-entity-interaction-hikscale-v1",
        )
        self.train_configs.add_argument(
            "--direct24_root_weight",
            type=float,
            default=1.0,
        )
        self.train_configs.add_argument(
            "--direct24_pose_weight",
            type=float,
            default=1.0,
        )
        self.train_configs.add_argument(
            "--direct24_yaw_weight",
            type=float,
            default=0.1,
        )
        self.train_configs.add_argument(
            "--direct24_loss_mode",
            choices=("disjoint",),
            default="disjoint",
        )

    def parse_args(self, args=None, namespace=None):
        config = super().parse_args(args=args, namespace=namespace)
        expected = (
            self.INPUT_FRAMES,
            self.FUTURE_FRAMES,
            self.TARGET_FPS,
            1000,
        )
        actual = (
            config.input_seq_len,
            config.output_seq_len,
            config.fps,
            config.sample_points,
        )
        if actual != expected:
            raise ValueError(
                "HOI-M3 HIK-scale direct24 requires "
                f"(input,future,fps,scene_points)={expected}, got {actual}"
            )
        for name in (
            "direct24_root_weight",
            "direct24_pose_weight",
            "direct24_yaw_weight",
        ):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if config.early_stop_patience != 0:
            raise ValueError("Use a fixed epoch budget; early stopping must be off")
        return config


MotionFromGazeDualConfigHOIM3Direct24 = HOIM3Direct24Config
