import time

import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from model.sif3d_final import SIF3D
from model.sif3d_sca_ori_v2 import SIF3DScaOriResidualV2
from train_dual_kd_common import (
    SMPLXKDStudentConditionedTeacherTrainer,
    _load_state_dict,
    seed_everything,
)


class ScaOriV2KDConfig(MotionFromGazeDualKDConfig):
    def __init__(self):
        super().__init__()
        self.set_defaults(
            kd_traj_weight=0.08,
            kd_dest_weight=0.15,
            kd_ori_weight=0.03,
            kd_dest_ori_weight=0.05,
            kd_pose_weight=0.0,
            kd_dest_pose_weight=0.0,
            kd_max_scale=0.2,
            kd_warmup_epochs=3,
            kd_ramp_epochs=5,
            lr=5.0e-6,
            val_fre=1,
        )
        extra = self.add_argument_group("sca_ori_v2")
        extra.add_argument("--sca_ori_residual_scale", type=float, default=0.10)
        extra.add_argument("--sca_spatial_detach_trajectory", action="store_true", default=True)
        extra.add_argument("--no_sca_spatial_detach_trajectory", dest="sca_spatial_detach_trajectory", action="store_false")
        extra.add_argument("--sca_ori_detach_base", action="store_true", default=True)
        extra.add_argument("--no_sca_ori_detach_base", dest="sca_ori_detach_base", action="store_false")
        extra.add_argument("--anchor_load_model_dir", type=str, default=None)
        extra.add_argument("--anchor_root_traj_weight", type=float, default=0.10)
        extra.add_argument("--anchor_dest_root_weight", type=float, default=0.20)


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    if epoch < config.kd_warmup_epochs:
        return 0.0
    if config.kd_ramp_epochs <= 0:
        return config.kd_max_scale
    ramp = min(1.0, float(epoch - config.kd_warmup_epochs + 1) / float(config.kd_ramp_epochs))
    return config.kd_max_scale * ramp


class ScaOriV2KDTrainer(SMPLXKDStudentConditionedTeacherTrainer):
    def __init__(self, config):
        super().__init__(config, "sca_ori_v2_student_object_kd", lambda *_: 0.0)

        self.model = SIF3DScaOriResidualV2(config, self.vposer, self.body_model).to(self.device)
        if self.config.load_model_dir is not None:
            missing, unexpected = self.model.load_state_dict(
                _load_state_dict(self.config.load_model_dir, self.device),
                strict=False,
            )
            self.logger.info("Loaded SCA-ori-v2 student init: missing=%s unexpected=%s", missing, unexpected)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        self.anchor_model = None
        if config.anchor_root_traj_weight > 0 or config.anchor_dest_root_weight > 0:
            anchor_path = config.anchor_load_model_dir or config.load_model_dir
            if not anchor_path:
                raise ValueError("anchor_load_model_dir or load_model_dir is required when anchor weights are non-zero.")
            self.anchor_model = SIF3D(config, self.vposer, self.body_model).to(self.device)
            self.anchor_model.load_state_dict(_load_state_dict(anchor_path, self.device), strict=False)
            self.anchor_model.eval()
            for param in self.anchor_model.parameters():
                param.requires_grad = False
            self.logger.info("Baseline trajectory anchor enabled with checkpoint: %s", anchor_path)

        self.logger.info(
            "SCA-ori-v2: residual_scale=%.4f detach_spatial=%s detach_ori_base=%s",
            config.sca_ori_residual_scale,
            config.sca_spatial_detach_trajectory,
            config.sca_ori_detach_base,
        )
        self.logger.info(
            "KD weights: traj=%.4f dest=%.4f ori=%.4f dest_ori=%.4f pose=%.4f dest_pose=%.4f "
            "max_scale=%.4f anchor_root=%.4f anchor_dest=%.4f",
            config.kd_traj_weight,
            config.kd_dest_weight,
            config.kd_ori_weight,
            config.kd_dest_ori_weight,
            config.kd_pose_weight,
            config.kd_dest_pose_weight,
            config.kd_max_scale,
            config.anchor_root_traj_weight,
            config.anchor_dest_root_weight,
        )

    def compute_total_loss(self, student_out, teacher_out, poses_input, poses_label, joints_input, joints_label):
        pose_loss_value = 0.0
        if self.config.use_pose_loss:
            (
                loss_ori,
                loss_trans,
                loss_latent,
                loss_des_ori,
                loss_des_trans,
                loss_des_latent,
                mpjpe,
                des_mpjpe,
            ) = self.calculate_loss(
                student_out["latent"],
                poses_label,
                poses_input,
                joints_input[:, :, :23],
                joints_label[:, :, :23],
            )
            pose_loss_value = (
                loss_ori.mean()
                + loss_trans.mean()
                + loss_latent.mean()
                + loss_des_ori
                + loss_des_trans
                + loss_des_latent
                + mpjpe.mean()
                + des_mpjpe
            )

        loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
            student_out["joints"],
            joints_label[:, :, :23],
            joints_input[:, :, :23],
        )
        gt_loss = (
            pose_loss_value
            + loss_trans_gcn.mean()
            + loss_des_trans_gcn
            + mpjpe_gcn.mean()
            + des_mpjpe_gcn
        )
        kd_loss = kd_scale(self.config) * self.teacher_kd_loss(student_out, teacher_out)
        anchor_loss = self.baseline_anchor_loss(student_out)
        total_loss = gt_loss + kd_loss + anchor_loss
        return total_loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn

    def teacher_kd_loss(self, student_out, teacher_out):
        future_slice = slice(self.config.input_seq_len, None)
        student_trans = student_out["trans"][:, future_slice]
        teacher_trans = teacher_out["trans"][:, future_slice]
        student_ori = student_out["ori"][:, future_slice]
        teacher_ori = teacher_out["ori"][:, future_slice]
        student_pose = student_out["pose"][:, future_slice]
        teacher_pose = teacher_out["pose"][:, future_slice]

        total = torch.tensor(0.0, device=student_trans.device)
        if self.config.kd_traj_weight > 0:
            total = total + self.config.kd_traj_weight * F.smooth_l1_loss(student_trans, teacher_trans)
        if self.config.kd_dest_weight > 0:
            total = total + self.config.kd_dest_weight * F.smooth_l1_loss(student_trans[:, -1], teacher_trans[:, -1])
        if self.config.kd_ori_weight > 0:
            total = total + self.config.kd_ori_weight * F.smooth_l1_loss(student_ori, teacher_ori)
        if self.config.kd_dest_ori_weight > 0:
            total = total + self.config.kd_dest_ori_weight * F.smooth_l1_loss(student_ori[:, -1], teacher_ori[:, -1])
        if self.config.kd_pose_weight > 0:
            total = total + self.config.kd_pose_weight * F.smooth_l1_loss(student_pose, teacher_pose)
        if self.config.kd_dest_pose_weight > 0:
            total = total + self.config.kd_dest_pose_weight * F.smooth_l1_loss(student_pose[:, -1], teacher_pose[:, -1])
        return total

    def baseline_anchor_loss(self, student_out):
        if self.anchor_model is None or not hasattr(self, "_anchor_batch"):
            return torch.tensor(0.0, device=student_out["joints"].device)
        with torch.no_grad():
            anchor_out = self.forward_model(self.anchor_model, *self._anchor_batch)

        future_slice = slice(self.config.input_seq_len, None)
        student_root = student_out["joints"][:, future_slice, 0]
        anchor_root = anchor_out["joints"][:, future_slice, 0]
        total = torch.tensor(0.0, device=student_root.device)
        if self.config.anchor_root_traj_weight > 0:
            total = total + self.config.anchor_root_traj_weight * F.smooth_l1_loss(student_root, anchor_root)
        if self.config.anchor_dest_root_weight > 0:
            total = total + self.config.anchor_dest_root_weight * F.smooth_l1_loss(student_root[:, -1], anchor_root[:, -1])
        return total

    def train(self):
        original_forward_model = self.forward_model

        def forward_with_anchor_context(model, poses_input, joints_input, scene_points, objects, object_mask, gt_prior, gt_prior_mask):
            if model is self.model:
                self._anchor_batch = (
                    poses_input,
                    joints_input,
                    scene_points,
                    objects,
                    object_mask,
                    gt_prior,
                    gt_prior_mask,
                )
            return original_forward_model(model, poses_input, joints_input, scene_points, objects, object_mask, gt_prior, gt_prior_mask)

        self.forward_model = forward_with_anchor_context
        super().train()


def run_training(config):
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    trainer = ScaOriV2KDTrainer(config)
    trainer.train()
    print("training done in", time.time() - start)


if __name__ == "__main__":
    config = ScaOriV2KDConfig().parse_args()
    run_training(config)
