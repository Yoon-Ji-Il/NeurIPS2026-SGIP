from argparse import ArgumentParser
import os


class MotionFromGazeDualConfig(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--des_trans_weight", type=float, default=1.0,
                 help="Weight for destination root-translation loss (last-frame loss_des_trans_gcn). "
                      "Increase to push the model harder on the final-frame root position.")
        self.add_argument("--mpjpe_weight", type=float, default=1.0,
                         help="Weight for root-relative joint loss (mpjpe + des_mpjpe). "
                              "Increase to push pose harder during KD.")
        self.add_argument("--pose_loss_weight", type=float, default=1.0,
                         help="Weight for VPoser/SMPL-X pose_loss bundle when --use_pose_loss is on.")
        self.input_configs = self.add_argument_group("input")
        self.input_configs.add_argument("--batch_size", default=8, type=int)
        self.input_configs.add_argument("--num_workers", default=4, type=int)
        self.input_configs.add_argument("--input_seq_len", default=6, type=int)
        self.input_configs.add_argument("--output_seq_len", default=10, type=int)
        self.input_configs.add_argument("--dataroot", default=os.environ.get("DATAROOT", "data/GIMO"), type=str)
        self.input_configs.add_argument("--fps", default=2, type=int)
        self.input_configs.add_argument("--sample_points", default=32768, type=int)
        self.input_configs.add_argument(
            "--object_points_name",
            default="final_object_qwen3_14b_motion_c_final_points_1024.pth",
            type=str,
            help=(
                "Object tensor path relative to SLICES_888s/{train,test}; "
                "the literal {split} is replaced with train or test."
            ),
        )
        self.input_configs.add_argument("--object_mask_name", default=None, type=str)
        self.input_configs.add_argument("--require_gt_object_point_prior", action="store_true")
        self.input_configs.add_argument(
            "--student_v0_no_object",
            action="store_true",
            help=(
                "V0 student baseline: load object tensors only for dataset compatibility, "
                "but disable object embedding and object-derived scene priors in the model."
            ),
        )

        self.add_argument("--model_type", default="cross", choices=["cross", "mint", "rnn", "st_transformer", "simple"])
        self.motion_configs = self.add_argument_group("motion_transformer")
        self.motion_configs.add_argument("--dropout", default=0.0, type=float)
        self.motion_configs.add_argument("--motion_n_heads", default=8, type=int)
        self.motion_configs.add_argument("--motion_hidden_dim", default=256, type=int)
        self.motion_configs.add_argument("--motion_intermediate_dim", default=1024, type=int)
        self.motion_configs.add_argument("--motion_n_layers", default=3, type=int)
        self.motion_configs.add_argument("--motion_latent_dim", default=256, type=int)
        self.motion_configs.add_argument("--scene_feats_dim", default=256, type=int)

        self.train_configs = self.add_argument_group("train")
        self.train_configs.add_argument(
            "--save_path",
            type=str,
            default="runs_dh/teacher_motion_c_final_seed7_retrain",
        )
        self.train_configs.add_argument("--save_fre", type=int, default=5)
        self.train_configs.add_argument(
            "--compact_checkpoints",
            action="store_true",
            help=(
                "Save only best_loss_des_trans.pth during training and last.pth "
                "when training finishes. The default preserves the original "
                "per-epoch/all-best checkpoint behavior."
            ),
        )
        self.train_configs.add_argument("--val_fre", type=int, default=3)
        self.train_configs.add_argument("--load_model_dir", type=str, default=None)
        self.train_configs.add_argument("--load_optim_dir", type=str, default=None)
        self.train_configs.add_argument("--teacher_load_model_dir", type=str, default=None)

        self.train_configs.add_argument("--epoch", type=int, default=100)
        self.train_configs.add_argument("--lr", type=float, default=3e-4)
        self.train_configs.add_argument("--weight_decay", type=float, default=1e-4)
        self.train_configs.add_argument("--gamma", type=float, default=0.98)
        self.train_configs.add_argument("--seed", type=int, default=7)
        self.train_configs.add_argument("--deterministic", action="store_true")
        self.train_configs.add_argument("--use_pose_loss", action="store_true")
        self.train_configs.add_argument("--use_bone_loss", action="store_true")
        self.train_configs.add_argument("--bone_loss_weight", type=float, default=1.0)
        self.train_configs.add_argument("--bone_loss_type", type=str, default="l1", choices=["l1", "l2"])
        self.train_configs.add_argument("--use_dct_loss", action="store_true")
        self.train_configs.add_argument("--dct_loss_weight", type=float, default=1.0)
        self.train_configs.add_argument("--dct_num_coeffs", type=int, default=8)
        self.train_configs.add_argument(
            "--dct_loss_type",
            type=str,
            default="l1",
            choices=["l1", "l2"],
        )
        self.train_configs.add_argument(
            "--use_kd",
            action="store_true",
            default=False,
            help="Enable feature-level knowledge distillation from a frozen teacher.",
        )
        self.train_configs.add_argument(
            "--teacher_object_points_name",
            type=str,
            default=None,
            help=(
                "Teacher object tensor path relative to SLICES_888s/{train,test}; "
                "the literal {split} is replaced with train or test. Required "
                "when --use_kd."
            ),
        )
        self.train_configs.add_argument(
            "--teacher_object_mask_name",
            type=str,
            default=None,
            help=(
                "Optional teacher object mask filename. If omitted, derived from "
                "teacher_object_points_name by replacing .pth with _mask.pth."
            ),
        )
        self.train_configs.add_argument(
            "--lambda_des_trans",
            type=float,
            default=1.0,
            help="Multiplier for loss_des_trans_gcn in motion_loss. >1 emphasizes destination trajectory.",
        )
        self.train_configs.add_argument(
            "--lambda_des_mpjpe",
            type=float,
            default=1.0,
            help="Multiplier for des_mpjpe_gcn in motion_loss. >1 emphasizes destination joint accuracy.",
        )
        self.train_configs.add_argument("--kd_feat_out_tia_weight", type=float, default=0.0)
        self.train_configs.add_argument("--kd_feat_out_sca_weight", type=float, default=0.0)
        self.train_configs.add_argument("--kd_feat_object_weight", type=float, default=0.0)
        self.train_configs.add_argument("--kd_feat_motion_enc_weight", type=float, default=0.0)
        self.train_configs.add_argument(
            "--kd_feat_out_tia_dest_weight",
            type=float,
            default=0.0,
            help="Cosine/L1 distill on out_tia at the LAST frame only (destination).",
        )
        self.train_configs.add_argument(
            "--kd_feat_out_sca_dest_weight",
            type=float,
            default=0.0,
            help="Cosine/L1 distill on out_sca at the LAST frame only (destination).",
        )
        self.train_configs.add_argument(
            "--kd_out_traj_weight",
            type=float,
            default=0.0,
            help="Smooth-L1 distill on (ori, trans), 6-dim.",
        )
        self.train_configs.add_argument(
            "--kd_out_pose_weight",
            type=float,
            default=0.0,
            help="Smooth-L1 distill on pose latent, 32-dim.",
        )
        self.train_configs.add_argument(
            "--kd_out_traj_dest_weight",
            type=float,
            default=0.0,
            help="Smooth-L1 distill on (ori, trans) at the LAST frame only.",
        )
        self.train_configs.add_argument(
            "--kd_out_pose_dest_weight",
            type=float,
            default=0.0,
            help="Smooth-L1 distill on pose latent at the LAST frame only.",
        )
        self.train_configs.add_argument(
            "--kd_feat_kind",
            type=str,
            default="cosine",
            choices=["cosine", "smooth_l1"],
        )
        self.train_configs.add_argument(
            "--early_stop_metric",
            type=str,
            default="mpjpe",
            choices=["loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"],
        )
        self.train_configs.add_argument("--early_stop_patience", type=int, default=0)
        self.train_configs.add_argument("--early_stop_min_delta", type=float, default=0.0)
        self.train_configs.add_argument("--early_stop_warmup", type=int, default=0)

        self.eval_configs = self.add_argument_group("eval")
        self.eval_configs.add_argument("--output_path", default="results", type=str)
        self.eval_configs.add_argument("--smplx_path", default=os.environ.get("SMPLX_PATH", "data/body_models"), type=str)
        self.eval_configs.add_argument("--vposer_path", default=os.environ.get("VPOSER_PATH", "vposer_v1_0"), type=str)
        self.eval_configs.add_argument("--drop_last_eval", action="store_true")

    def get_configs(self):
        return self.parse_args()
