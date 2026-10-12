import time

import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from model.sif3d_final import SIF3D
from train_dual_kd_common import (
    SMPLXKDStudentConditionedTeacherTrainer,
    _load_state_dict,
    seed_everything,
)


class PoseAnchorKDConfig(MotionFromGazeDualKDConfig):
    def __init__(self):
        super().__init__()
        extra = self.add_argument_group("pose_anchor_kd")
        extra.add_argument("--kd_pose_delta_weight", type=float, default=0.0)
        extra.add_argument("--kd_dest_pose_delta_weight", type=float, default=0.0)
        extra.add_argument("--anchor_load_model_dir", type=str, default=None)
        extra.add_argument("--anchor_root_traj_weight", type=float, default=0.0)
        extra.add_argument("--anchor_dest_root_weight", type=float, default=0.0)


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    if epoch < config.kd_warmup_epochs:
        return 0.0
    if config.kd_ramp_epochs <= 0:
        return config.kd_max_scale
    ramp = min(1.0, float(epoch - config.kd_warmup_epochs + 1) / float(config.kd_ramp_epochs))
    return config.kd_max_scale * ramp


def pose_loss(config, student_pose, teacher_pose):
    loss_type = getattr(config, "kd_pose_loss_type", "smooth_l1")
    eps = getattr(config, "kd_pose_loss_eps", 1.0e-6)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_pose, teacher_pose)
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(student_pose, teacher_pose, dim=-1, eps=eps)).mean()
    if loss_type == "normalized_mse":
        student_pose = F.normalize(student_pose, dim=-1, eps=eps)
        teacher_pose = F.normalize(teacher_pose, dim=-1, eps=eps)
        return F.mse_loss(student_pose, teacher_pose)
    raise ValueError(f"Unsupported kd_pose_loss_type: {loss_type}")


class PoseAnchorKDTrainer(SMPLXKDStudentConditionedTeacherTrainer):
    def __init__(self, config):
        super().__init__(config, "pose_anchor_student_object_kd", lambda *_: 0.0)
        self.anchor_model = None
        if config.anchor_root_traj_weight > 0 or config.anchor_dest_root_weight > 0:
            anchor_path = config.anchor_load_model_dir or config.load_model_dir
            if not anchor_path:
                raise ValueError("anchor_load_model_dir or load_model_dir is required for trajectory anchor.")
            self.anchor_model = SIF3D(config, self.vposer, self.body_model).to(self.device)
            self.anchor_model.load_state_dict(_load_state_dict(anchor_path, self.device), strict=False)
            self.anchor_model.eval()
            for param in self.anchor_model.parameters():
                param.requires_grad = False
            self.logger.info("Baseline trajectory anchor enabled with checkpoint: %s", anchor_path)

        self.logger.info(
            "Pose-anchor KD weights: pose=%.4f dest_pose=%.4f pose_delta=%.4f "
            "dest_pose_delta=%.4f anchor_root=%.4f anchor_dest_root=%.4f",
            config.kd_pose_weight,
            config.kd_dest_pose_weight,
            config.kd_pose_delta_weight,
            config.kd_dest_pose_delta_weight,
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
        kd_loss = kd_scale(self.config) * self.teacher_pose_kd_loss(student_out, teacher_out, poses_input)
        anchor_loss = self.baseline_anchor_loss(student_out, poses_input, joints_input)
        total_loss = gt_loss + kd_loss + anchor_loss
        return total_loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn

    def teacher_pose_kd_loss(self, student_out, teacher_out, poses_input):
        future_slice = slice(self.config.input_seq_len, None)
        student_pose = student_out["pose"][:, future_slice]
        teacher_pose = teacher_out["pose"][:, future_slice]
        total = torch.tensor(0.0, device=student_pose.device)

        if self.config.kd_pose_weight > 0:
            total = total + self.config.kd_pose_weight * pose_loss(self.config, student_pose, teacher_pose)
        if self.config.kd_dest_pose_weight > 0:
            total = total + self.config.kd_dest_pose_weight * pose_loss(
                self.config,
                student_pose[:, -1:],
                teacher_pose[:, -1:],
            )
        if self.config.kd_pose_delta_weight > 0 or self.config.kd_dest_pose_delta_weight > 0:
            observed_last_pose = poses_input[:, [-1], 6:]
            student_delta = student_pose - observed_last_pose
            teacher_delta = teacher_pose - observed_last_pose
            if self.config.kd_pose_delta_weight > 0:
                total = total + self.config.kd_pose_delta_weight * F.smooth_l1_loss(student_delta, teacher_delta)
            if self.config.kd_dest_pose_delta_weight > 0:
                total = total + self.config.kd_dest_pose_delta_weight * F.smooth_l1_loss(
                    student_delta[:, -1],
                    teacher_delta[:, -1],
                )
        return total

    def baseline_anchor_loss(self, student_out, poses_input, joints_input):
        if self.anchor_model is None:
            return torch.tensor(0.0, device=student_out["joints"].device)

        # The anchor output is computed in train/test with the same student object condition.
        # It keeps the root trajectory close to the already strong baseline12 solution.
        # Inputs are cached from the current batch by overriding forward pass context in compute loops.
        if not hasattr(self, "_anchor_batch"):
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
    trainer = PoseAnchorKDTrainer(config)
    trainer.train()
    print("training done in", time.time() - start)


if __name__ == "__main__":
    config = PoseAnchorKDConfig().parse_args()
    run_training(config)
