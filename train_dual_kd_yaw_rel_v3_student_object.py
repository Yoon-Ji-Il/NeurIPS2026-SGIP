import time

import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from model.sif3d_final import SIF3D
from model.sif3d_sca_ori_v2 import SIF3DScaOriResidualV2
from train_dual_kd_common import (
    SMPLXKDStudentConditionedTeacherTrainer,
    seed_everything,
)


class YawRelV3KDConfig(MotionFromGazeDualKDConfig):
    def __init__(self):
        super().__init__()
        self.set_defaults(
            kd_traj_weight=0.0,
            kd_dest_weight=0.0,
            kd_ori_weight=0.02,
            kd_dest_ori_weight=0.03,
            kd_pose_weight=0.0,
            kd_dest_pose_weight=0.0,
            kd_max_scale=0.2,
            kd_warmup_epochs=3,
            kd_ramp_epochs=5,
            lr=5.0e-6,
            val_fre=1,
        )
        extra = self.add_argument_group("yaw_rel_v3")
        extra.add_argument("--sca_ori_residual_scale", type=float, default=0.05)
        extra.add_argument("--sca_spatial_detach_trajectory", action="store_true", default=True)
        extra.add_argument("--no_sca_spatial_detach_trajectory", dest="sca_spatial_detach_trajectory", action="store_false")
        extra.add_argument("--sca_ori_detach_base", action="store_true", default=True)
        extra.add_argument("--no_sca_ori_detach_base", dest="sca_ori_detach_base", action="store_false")
        extra.add_argument("--anchor_load_model_dir", type=str, default=None)
        extra.add_argument("--anchor_root_traj_weight", type=float, default=0.30)
        extra.add_argument("--anchor_dest_root_weight", type=float, default=0.60)
        extra.add_argument("--kd_yaw_rel_joints_weight", type=float, default=0.20)
        extra.add_argument("--kd_dest_yaw_rel_joints_weight", type=float, default=0.40)
        extra.add_argument("--detach_root_in_gt_mpjpe", action="store_true", default=True)
        extra.add_argument("--no_detach_root_in_gt_mpjpe", dest="detach_root_in_gt_mpjpe", action="store_false")


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    if epoch < config.kd_warmup_epochs:
        return 0.0
    if config.kd_ramp_epochs <= 0:
        return config.kd_max_scale
    ramp = min(1.0, float(epoch - config.kd_warmup_epochs + 1) / float(config.kd_ramp_epochs))
    return config.kd_max_scale * ramp


def yaw_local_joints(relative_joints, ori):
    """Rotate root-relative joints into the model's yaw-aligned local frame."""
    yaw = ori[..., 1]
    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    x = relative_joints[..., 0]
    y = relative_joints[..., 1]
    z = relative_joints[..., 2]
    local_x = sin_yaw * x - cos_yaw * z
    local_z = cos_yaw * x + sin_yaw * z
    return torch.stack([local_x, y, local_z], dim=-1)


def load_state_dict_cpu(path):
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}


class YawRelV3KDTrainer(SMPLXKDStudentConditionedTeacherTrainer):
    def __init__(self, config):
        student_init_path = config.load_model_dir
        config.load_model_dir = None
        super().__init__(config, "yaw_rel_v3_student_object_kd", lambda *_: 0.0)
        config.load_model_dir = student_init_path

        self.model = SIF3DScaOriResidualV2(config, self.vposer, self.body_model).to(self.device)
        if student_init_path is not None:
            missing, unexpected = self.model.load_state_dict(
                load_state_dict_cpu(student_init_path),
                strict=False,
            )
            self.logger.info("Loaded yaw-rel-v3 student init: missing=%s unexpected=%s", missing, unexpected)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        self.anchor_model = None
        if config.anchor_root_traj_weight > 0 or config.anchor_dest_root_weight > 0:
            anchor_path = config.anchor_load_model_dir or student_init_path
            if not anchor_path:
                raise ValueError("anchor_load_model_dir or load_model_dir is required when anchor weights are non-zero.")
            self.anchor_model = SIF3D(config, self.vposer, self.body_model).to(self.device)
            self.anchor_model.load_state_dict(load_state_dict_cpu(anchor_path), strict=False)
            self.anchor_model.eval()
            for param in self.anchor_model.parameters():
                param.requires_grad = False
            self.logger.info("Baseline trajectory anchor enabled with checkpoint: %s", anchor_path)

        self.logger.info(
            "Yaw-rel-v3: residual_scale=%.4f detach_spatial=%s detach_ori_base=%s detach_root_gt_mpjpe=%s",
            config.sca_ori_residual_scale,
            config.sca_spatial_detach_trajectory,
            config.sca_ori_detach_base,
            config.detach_root_in_gt_mpjpe,
        )
        self.logger.info(
            "KD weights: yaw_rel=%.4f dest_yaw_rel=%.4f ori=%.4f dest_ori=%.4f max_scale=%.4f "
            "anchor_root=%.4f anchor_dest=%.4f",
            config.kd_yaw_rel_joints_weight,
            config.kd_dest_yaw_rel_joints_weight,
            config.kd_ori_weight,
            config.kd_dest_ori_weight,
            config.kd_max_scale,
            config.anchor_root_traj_weight,
            config.anchor_dest_root_weight,
        )

    def compute_total_loss(self, student_out, teacher_out, poses_input, poses_label, joints_input, joints_label):
        loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
            student_out["joints"],
            joints_label[:, :, :23],
            joints_input[:, :, :23],
        )
        gt_loss = self.root_decoupled_gt_loss(
            student_out["joints"],
            joints_input[:, :, :23],
            joints_label[:, :, :23],
            loss_trans_gcn,
            loss_des_trans_gcn,
            mpjpe_gcn,
            des_mpjpe_gcn,
        )
        kd_loss = kd_scale(self.config) * self.teacher_kd_loss(student_out, teacher_out)
        anchor_loss = self.baseline_anchor_loss(student_out)
        total_loss = gt_loss + kd_loss + anchor_loss
        return total_loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn

    def root_decoupled_gt_loss(
        self,
        student_joints,
        joints_input,
        joints_label,
        loss_trans_gcn,
        loss_des_trans_gcn,
        mpjpe_gcn,
        des_mpjpe_gcn,
    ):
        if not self.config.detach_root_in_gt_mpjpe:
            return loss_trans_gcn[:, self.config.input_seq_len :].mean() + loss_des_trans_gcn + mpjpe_gcn[
                :, self.config.input_seq_len :
            ].mean() + des_mpjpe_gcn

        target_joints = torch.cat([joints_input, joints_label], dim=1)
        pred_root = student_joints[:, :, [0]].detach()
        target_root = target_joints[:, :, [0]]
        pred_rel = student_joints - pred_root
        target_rel = target_joints - target_root
        rel_mpjpe = torch.norm(pred_rel[:, :, 1:] - target_rel[:, :, 1:], dim=-1)
        future_slice = slice(self.config.input_seq_len, None)
        return (
            loss_trans_gcn[:, future_slice].mean()
            + loss_des_trans_gcn
            + rel_mpjpe[:, future_slice].mean()
            + rel_mpjpe[:, -1].mean()
        )

    def teacher_kd_loss(self, student_out, teacher_out):
        future_slice = slice(self.config.input_seq_len, None)
        student_joints = student_out["joints"][:, future_slice]
        teacher_joints = teacher_out["joints"][:, future_slice]
        student_ori = student_out["ori"][:, future_slice]
        teacher_ori = teacher_out["ori"][:, future_slice]

        student_rel = student_joints - student_joints[:, :, [0]].detach()
        teacher_rel = teacher_joints - teacher_joints[:, :, [0]]
        student_local = yaw_local_joints(student_rel, student_ori)
        teacher_local = yaw_local_joints(teacher_rel, teacher_ori)

        total = torch.tensor(0.0, device=student_joints.device)
        if self.config.kd_yaw_rel_joints_weight > 0:
            total = total + self.config.kd_yaw_rel_joints_weight * F.smooth_l1_loss(
                student_local[:, :, 1:],
                teacher_local[:, :, 1:],
            )
        if self.config.kd_dest_yaw_rel_joints_weight > 0:
            total = total + self.config.kd_dest_yaw_rel_joints_weight * F.smooth_l1_loss(
                student_local[:, -1, 1:],
                teacher_local[:, -1, 1:],
            )
        if self.config.kd_ori_weight > 0:
            total = total + self.config.kd_ori_weight * F.smooth_l1_loss(student_ori, teacher_ori)
        if self.config.kd_dest_ori_weight > 0:
            total = total + self.config.kd_dest_ori_weight * F.smooth_l1_loss(student_ori[:, -1], teacher_ori[:, -1])
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
            total = total + self.config.anchor_dest_root_weight * F.smooth_l1_loss(
                student_root[:, -1],
                anchor_root[:, -1],
            )
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
    trainer = YawRelV3KDTrainer(config)
    trainer.train()
    print("training done in", time.time() - start)


if __name__ == "__main__":
    config = YawRelV3KDConfig().parse_args()
    run_training(config)
