"""Separate model configuration for HIK's native H25F50 experiment."""

from .config_dual import MotionFromGazeDualConfig


class HIK1s2sConfig(MotionFromGazeDualConfig):
    """Use HIK unchanged: 25 observed and 50 future frames at 25 Hz.

    The parent configuration is intentionally left unchanged so existing GIMO
    experiments continue to use 6 observed + 10 future frames.
    """

    INPUT_FRAMES = 25
    FUTURE_FRAMES = 50
    TARGET_FPS = 25

    def __init__(self):
        super().__init__()
        self.set_defaults(
            batch_size=1,
            input_seq_len=self.INPUT_FRAMES,
            output_seq_len=self.FUTURE_FRAMES,
            fps=self.TARGET_FPS,
            sample_points=1000,
            dataroot=(
                "data/HIK/"
                "hik_preprocessed/H25F50"
            ),
            object_points_name=(
                "final_object_qwen3_14b_action_geometry_h25f50_"
                "t25_gt29_points_1024.pth"
            ),
            save_path="runs_hik/hik_h25f50_1s2s",
            vposer_path="./vposer_v1_0",
            smplx_path="data/body_models",
        )
        self.input_configs.add_argument("--hik_source_fps", type=int, default=25)
        self.input_configs.add_argument(
            "--hik_raw_root",
            type=str,
            default=(
                "data/HIK/"
                "SAST/data"
            ),
        )
        self.input_configs.add_argument(
            "--hik_max_other_people",
            type=int,
            default=15,
            help=(
                "Maximum number of non-target people. HIK supports at most "
                "16 simultaneous people including the target."
            ),
        )
        self.input_configs.add_argument(
            "--hik_max_people_scene_points",
            type=int,
            default=435,
            help=(
                "Point budget reserved for last-observed 29-joint person "
                "point sets: 15 people * 29 joints."
            ),
        )
        self.input_configs.add_argument(
            "--hik_scene_radius",
            type=float,
            default=3.5,
            help="HIK environment crop radius around the target root in metres.",
        )
        self.input_configs.add_argument(
            "--hik_other_person_radius",
            type=float,
            default=2.5,
            help=(
                "Maximum last-observed root distance for another person to "
                "be merged into the target's scene."
            ),
        )

    def parse_args(self, args=None, namespace=None):
        config = super().parse_args(args=args, namespace=namespace)
        expected = (
            self.INPUT_FRAMES,
            self.FUTURE_FRAMES,
            self.TARGET_FPS,
        )
        actual = (
            config.input_seq_len,
            config.output_seq_len,
            config.fps,
        )
        if actual != expected:
            raise ValueError(
                "HIK1s2sConfig uses HIK's native H25F50 setting and requires "
                f"(input_seq_len, output_seq_len, fps)={expected}, got {actual}"
            )
        if config.hik_source_fps != 25:
            raise ValueError(
                "The checked HIK raw data is 25 Hz; "
                f"got hik_source_fps={config.hik_source_fps}"
            )
        if config.sample_points != 1000:
            raise ValueError(
                "The native HUMOF HIK scene setting uses exactly 1000 points; "
                f"got sample_points={config.sample_points}"
            )
        if config.hik_max_other_people != 15:
            raise ValueError(
                "HIK supports 16 people including the target, so the fixed "
                "non-target maximum is 15"
            )
        if config.hik_max_people_scene_points != 15 * 29:
            raise ValueError(
                "The fixed last-frame skeleton budget must be 15 * 29 = 435"
            )
        return config


# A drop-in name for scripts that expect the original parser class name.
MotionFromGazeDualConfigHIK1s2s = HIK1s2sConfig
