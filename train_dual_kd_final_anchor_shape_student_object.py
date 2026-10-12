import time

import torch
import torch.nn.functional as F

from config.config_dual_kd import MotionFromGazeDualKDConfig
from train_dual_kd_common import SMPLXKDStudentConditionedTeacherTrainer, seed_everything


class FinalAnchorShapeKDConfig(MotionFromGazeDualKDConfig):
    def __init__(self):
        super().__init__()
        extra = self.add_argument_group("final_anchor_shape_kd")
        extra.add_argument("--gt_final_root_weight", type=float, default=0.0)
        extra.add_argument("--gt_final_rel_joints_weight", type=float, default=0.0)
        extra.add_argument("--gt_final_pose_weight", type=float, default=0.0)
        extra.add_argument("--gt_final_ori_weight", type=float, default=0.0)
        extra.add_argument("--gt_object_distance_weight", type=float, default=0.0)
        extra.add_argument("--final_window", type=int, default=1)
        extra.add_argument("--final_time_boost", type=float, default=1.0)
        extra.add_argument("--kd_root_shape_weight", type=float, default=0.0)
        extra.add_argument("--kd_root_velocity_weight", type=float, default=0.0)
        extra.add_argument("--kd_object_distance_weight", type=float, default=0.0)
        extra.add_argument("--kd_teacher_advantage_gated", action="store_true")


def kd_scale(config):
    epoch = getattr(config, "_current_epoch", 0) + getattr(config, "kd_epoch_offset", 0)
    if epoch < config.kd_warmup_epochs:
        return 0.0
    if config.kd_ramp_epochs <= 0:
        return config.kd_max_scale
    ramp = min(1.0, float(epoch - config.kd_warmup_epochs + 1) / float(config.kd_ramp_epochs))
    return config.kd_max_scale * ramp


def weighted_time_mean(loss_by_time, final_time_boost, mask=None):
    if mask is not None:
        loss_by_time = loss_by_time * mask.to(loss_by_time.dtype)
    if loss_by_time.shape[1] == 1:
        denom = mask.to(loss_by_time.dtype).sum().clamp_min(1.0) if mask is not None else loss_by_time.numel()
        return loss_by_time.sum() / denom
    weights = torch.linspace(
        1.0,
        final_time_boost,
        steps=loss_by_time.shape[1],
        device=loss_by_time.device,
        dtype=loss_by_time.dtype,
    )
    weighted = loss_by_time * weights.view(1, -1)
    if mask is not None:
        denom = (mask.to(loss_by_time.dtype) * weights.view(1, -1)).sum().clamp_min(1.0)
        return weighted.sum() / denom
    return weighted.sum() / (loss_by_time.shape[0] * weights.sum())


def object_centroid(objects, object_mask):
    if object_mask.ndim == 1:
        reduce_dims = tuple(range(1, objects.ndim - 1))
        centroid = objects.mean(dim=reduce_dims)
        return centroid * object_mask.to(objects.dtype).view(-1, 1)
    mask = object_mask
    while mask.ndim < objects.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(objects.dtype)
    reduce_dims = tuple(range(1, objects.ndim - 1))
    centroid = (objects * mask).sum(dim=reduce_dims)
    denom = mask.sum(dim=reduce_dims).clamp_min(1.0)
    return centroid / denom


def relative_joint_features(joints):
    joints = joints - joints[:, :, [0]]
    joints = joints[:, :, 1:]
    return joints.reshape(joints.shape[0], joints.shape[1], -1)


def pose_loss_by_time(student_pose, teacher_pose, loss_type, eps):
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_pose, teacher_pose, reduction="none").mean(dim=-1)
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(student_pose, teacher_pose, dim=-1, eps=eps)
    if loss_type == "normalized_mse":
        student_pose = F.normalize(student_pose, dim=-1, eps=eps)
        teacher_pose = F.normalize(teacher_pose, dim=-1, eps=eps)
        return F.mse_loss(student_pose, teacher_pose, reduction="none").mean(dim=-1)
    raise ValueError(f"Unsupported kd_pose_loss_type: {loss_type}")


class FinalAnchorShapeKDTrainer(SMPLXKDStudentConditionedTeacherTrainer):
    def forward_model(self, model, poses_input, joints_input, scene_points, objects, object_mask, gt_prior, gt_prior_mask):
        out = super().forward_model(
            model,
            poses_input,
            joints_input,
            scene_points,
            objects,
            object_mask,
            gt_prior,
            gt_prior_mask,
        )
        out["object_centroid"] = object_centroid(objects, object_mask)
        return out

    def compute_total_loss(self, student_out, teacher_out, poses_input, poses_label, joints_input, joints_label):
        pose_loss = 0.0
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
            pose_loss = (
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
            pose_loss
            + loss_trans_gcn.mean()
            + loss_des_trans_gcn
            + mpjpe_gcn.mean()
            + des_mpjpe_gcn
        )
        extra_gt = self.gt_final_anchor_loss(student_out, poses_input, poses_label, joints_input, joints_label)
        kd_loss = kd_scale(self.config) * self.teacher_kd_loss(
            student_out,
            teacher_out,
            poses_input,
            poses_label,
            joints_input,
            joints_label,
        )
        total_loss = gt_loss + extra_gt + kd_loss
        return total_loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn

    def gt_final_anchor_loss(self, student_out, poses_input, poses_label, joints_input, joints_label):
        window = max(1, min(self.config.final_window, self.config.output_seq_len))
        poses_target = torch.cat([poses_input, poses_label], dim=1)
        joints_target = torch.cat([joints_input[:, :, :23], joints_label[:, :, :23]], dim=1)
        pred_joints = student_out["joints"][:, -window:]
        target_joints = joints_target[:, -window:]

        root_loss = torch.norm(pred_joints[:, :, 0] - target_joints[:, :, 0], dim=-1)
        rel_loss = torch.norm(
            relative_joint_features(pred_joints) - relative_joint_features(target_joints),
            dim=-1,
        )
        pose_loss = F.smooth_l1_loss(
            student_out["pose"][:, -window:],
            poses_target[:, -window:, 6:],
            reduction="none",
        ).mean(dim=-1)
        ori_loss = F.smooth_l1_loss(
            student_out["ori"][:, -window:],
            poses_target[:, -window:, :3],
            reduction="none",
        ).mean(dim=-1)
        object_distance_loss = torch.abs(
            torch.norm(pred_joints[:, :, 0] - student_out["object_centroid"].unsqueeze(1), dim=-1)
            - torch.norm(target_joints[:, :, 0] - student_out["object_centroid"].unsqueeze(1), dim=-1)
        )

        return (
            self.config.gt_final_root_weight * weighted_time_mean(root_loss, self.config.final_time_boost)
            + self.config.gt_final_rel_joints_weight * weighted_time_mean(rel_loss, self.config.final_time_boost)
            + self.config.gt_final_pose_weight * weighted_time_mean(pose_loss, self.config.final_time_boost)
            + self.config.gt_final_ori_weight * weighted_time_mean(ori_loss, self.config.final_time_boost)
            + self.config.gt_object_distance_weight
            * weighted_time_mean(object_distance_loss, self.config.final_time_boost)
        )

    def teacher_kd_loss(self, student_out, teacher_out, poses_input, poses_label, joints_input, joints_label):
        future_slice = slice(self.config.input_seq_len, None)
        student_joints = student_out["joints"][:, future_slice]
        teacher_joints = teacher_out["joints"][:, future_slice]
        joints_target = joints_label[:, :, :23]
        poses_target = poses_label

        student_root = student_joints[:, :, 0]
        teacher_root = teacher_joints[:, :, 0]
        root_shape_loss = F.smooth_l1_loss(
            student_root - student_root[:, -1:],
            teacher_root - teacher_root[:, -1:],
            reduction="none",
        ).mean(dim=-1)
        root_velocity_loss = F.smooth_l1_loss(
            student_root[:, 1:] - student_root[:, :-1],
            teacher_root[:, 1:] - teacher_root[:, :-1],
            reduction="none",
        ).mean(dim=-1)
        teacher_object_distance_loss = torch.abs(
            torch.norm(student_root - student_out["object_centroid"].unsqueeze(1), dim=-1)
            - torch.norm(teacher_root - teacher_out["object_centroid"].unsqueeze(1), dim=-1)
        )

        student_rel = relative_joint_features(student_joints)
        teacher_rel = relative_joint_features(teacher_joints)
        target_rel = relative_joint_features(joints_target)
        rel_loss = F.smooth_l1_loss(student_rel, teacher_rel, reduction="none").mean(dim=-1)
        dest_rel_loss = F.smooth_l1_loss(student_rel[:, -1], teacher_rel[:, -1], reduction="none").mean(dim=-1, keepdim=True)

        student_pose = student_out["pose"][:, future_slice]
        teacher_pose = teacher_out["pose"][:, future_slice]
        pose_loss = pose_loss_by_time(
            student_pose,
            teacher_pose,
            self.config.kd_pose_loss_type,
            self.config.kd_pose_loss_eps,
        )
        dest_pose_loss = pose_loss_by_time(
            student_pose[:, -1:],
            teacher_pose[:, -1:],
            self.config.kd_pose_loss_type,
            self.config.kd_pose_loss_eps,
        )
        dest_ori_loss = F.smooth_l1_loss(
            student_out["ori"][:, -1],
            teacher_out["ori"][:, -1],
            reduction="none",
        ).mean(dim=-1, keepdim=True)

        rel_gate = None
        pose_gate = None
        if self.config.kd_teacher_advantage_gated:
            with torch.no_grad():
                student_rel_err = F.smooth_l1_loss(student_rel, target_rel, reduction="none").mean(dim=-1)
                teacher_rel_err = F.smooth_l1_loss(teacher_rel, target_rel, reduction="none").mean(dim=-1)
                rel_gate = teacher_rel_err < student_rel_err
                student_pose_err = F.smooth_l1_loss(student_pose, poses_target[:, :, 6:], reduction="none").mean(dim=-1)
                teacher_pose_err = F.smooth_l1_loss(teacher_pose, poses_target[:, :, 6:], reduction="none").mean(dim=-1)
                pose_gate = teacher_pose_err < student_pose_err

        return (
            self.config.kd_root_shape_weight * weighted_time_mean(root_shape_loss, self.config.final_time_boost)
            + self.config.kd_root_velocity_weight * weighted_time_mean(root_velocity_loss, self.config.final_time_boost)
            + self.config.kd_object_distance_weight
            * weighted_time_mean(teacher_object_distance_loss, self.config.final_time_boost)
            + self.config.kd_joints_weight
            * weighted_time_mean(rel_loss, self.config.final_time_boost, mask=rel_gate)
            + self.config.kd_dest_joints_weight
            * weighted_time_mean(dest_rel_loss, self.config.final_time_boost, mask=rel_gate[:, -1:] if rel_gate is not None else None)
            + self.config.kd_pose_weight
            * weighted_time_mean(pose_loss, self.config.final_time_boost, mask=pose_gate)
            + self.config.kd_dest_pose_weight
            * weighted_time_mean(dest_pose_loss, self.config.final_time_boost, mask=pose_gate[:, -1:] if pose_gate is not None else None)
            + self.config.kd_dest_ori_weight * dest_ori_loss.mean()
        )


def run_training(config):
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    trainer = FinalAnchorShapeKDTrainer(config, "final_anchor_shape_student_object_kd", lambda *_: 0.0)
    trainer.train()
    print("training done in", time.time() - start)


if __name__ == "__main__":
    config = FinalAnchorShapeKDConfig().parse_args()
    run_training(config)
