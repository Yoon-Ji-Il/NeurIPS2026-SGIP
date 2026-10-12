"""Configuration for the isolated HIK direct-29-joint H25F50 model."""

from .config_hik_1s2s import HIK1s2sConfig


class HIKDirect29Config(HIK1s2sConfig):
    """Keep HIK at 25 Hz while replacing VPoser with native 29-joint motion."""

    N_JOINTS = 29

    def __init__(self):
        super().__init__()
        self.set_defaults(
            dataroot=(
                "data/HIK/"
                "hik_preprocessed/H25F50_nonwalking"
            ),
            save_path="runs_hik/hik_h25f50_nonwalking_direct29_seed7_gpu3",
            early_stop_patience=0,
        )
        self.input_configs.add_argument(
            "--direct29_root",
            type=str,
            default=(
                "data/HIK/"
                "hik_preprocessed/H25F50_nonwalking_direct29"
            ),
            help=(
                "Root containing SLICES_888s/{train,test}/joints29_*.pth. "
                "Scene and object caches continue to come from --dataroot."
            ),
        )
        self.train_configs.add_argument(
            "--direct29_root_weight",
            type=float,
            default=1.0,
            help="Weight for the single future root-trajectory loss.",
        )
        self.train_configs.add_argument(
            "--direct29_pose_weight",
            type=float,
            default=1.0,
            help="Weight for the single future root-relative 29-joint loss.",
        )
        self.train_configs.add_argument(
            "--direct29_yaw_weight",
            type=float,
            default=0.1,
            help="Weight for wrapped yaw supervision used by scene normalization.",
        )
        self.train_configs.add_argument(
            "--direct29_loss_mode",
            choices=("disjoint", "legacy_overlap"),
            default="disjoint",
            help=(
                "disjoint supervises root/pose once. legacy_overlap adds the "
                "same final-frame root/pose terms again, matching the previous "
                "mean-plus-destination objective as closely as direct29 allows."
            ),
        )
        self.train_configs.add_argument(
            "--direct29_destination_weight",
            type=float,
            default=1.0,
            help=(
                "Final-frame weight used only by legacy_overlap. It has no "
                "effect in disjoint mode."
            ),
        )

    def parse_args(self, args=None, namespace=None):
        config = super().parse_args(args=args, namespace=namespace)
        for name in (
            "direct29_root_weight",
            "direct29_pose_weight",
            "direct29_yaw_weight",
            "direct29_destination_weight",
        ):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        return config


MotionFromGazeDualConfigHIKDirect29 = HIKDirect29Config
