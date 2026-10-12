import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.config_dual import MotionFromGazeDualConfig
from train_dual import SMPLX_evalutor, seed_everything
from utils.logger import MetricTracker


class FinalTargetConfig(MotionFromGazeDualConfig):
    def __init__(self):
        super().__init__()
        target_configs = self.add_argument_group("final_target_finetune")
        target_configs.add_argument("--extra_dest_trans_weight", type=float, default=0.0)
        target_configs.add_argument("--extra_dest_mpjpe_weight", type=float, default=0.0)
        target_configs.add_argument("--extra_final_pose_weight", type=float, default=0.0)
        target_configs.add_argument("--extra_final_ori_weight", type=float, default=0.0)
        target_configs.add_argument("--final_window", type=int, default=1)
        target_configs.add_argument("--final_time_boost", type=float, default=1.0)
        target_configs.add_argument("--target_warmup_epochs", type=int, default=0)
        target_configs.add_argument("--target_ramp_epochs", type=int, default=0)
        target_configs.add_argument("--target_max_scale", type=float, default=1.0)


def target_scale(config, epoch):
    if epoch < config.target_warmup_epochs:
        return 0.0
    if config.target_ramp_epochs <= 0:
        return config.target_max_scale
    ramp = min(1.0, float(epoch - config.target_warmup_epochs + 1) / float(config.target_ramp_epochs))
    return config.target_max_scale * ramp


def weighted_time_mean(loss_by_time, final_time_boost):
    if loss_by_time.shape[1] == 1:
        return loss_by_time.mean()
    weights = torch.linspace(
        1.0,
        final_time_boost,
        steps=loss_by_time.shape[1],
        device=loss_by_time.device,
        dtype=loss_by_time.dtype,
    )
    return (loss_by_time * weights.view(1, -1)).sum() / (loss_by_time.shape[0] * weights.sum())


class FinalTargetFinetuneEvaluator(SMPLX_evalutor):
    def __init__(self, config):
        super().__init__(config)
        self.logger.info(
            "Final-target fine-tune: extra_dest_trans=%.3f extra_dest_mpjpe=%.3f "
            "extra_final_pose=%.3f extra_final_ori=%.3f final_window=%d final_time_boost=%.3f "
            "target_warmup=%d target_ramp=%d target_max_scale=%.3f",
            config.extra_dest_trans_weight,
            config.extra_dest_mpjpe_weight,
            config.extra_final_pose_weight,
            config.extra_final_ori_weight,
            config.final_window,
            config.final_time_boost,
            config.target_warmup_epochs,
            config.target_ramp_epochs,
            config.target_max_scale,
        )

    def final_target_loss(self, poses_predict, joints_predict, poses_input, poses_label, joints_input, joints_label):
        poses_target = torch.cat([poses_input, poses_label], dim=1)
        joints_target = torch.cat([joints_input[:, :, :23], joints_label[:, :, :23]], dim=1)
        window = max(1, min(self.config.final_window, self.config.output_seq_len))

        pred_joints = joints_predict[:, -window:]
        target_joints = joints_target[:, -window:]
        root_dist = torch.norm(pred_joints[:, :, 0] - target_joints[:, :, 0], dim=-1)
        dest_trans_loss = weighted_time_mean(root_dist, self.config.final_time_boost)

        pred_relative = pred_joints - pred_joints[:, :, [0]]
        target_relative = target_joints - target_joints[:, :, [0]]
        mpjpe_by_time = torch.norm(pred_relative - target_relative, dim=-1).mean(dim=-1)
        dest_mpjpe_loss = weighted_time_mean(mpjpe_by_time, self.config.final_time_boost)

        pose_loss = torch.tensor(0.0, device=poses_predict.device)
        if self.config.extra_final_pose_weight > 0:
            pose_loss_by_time = F.smooth_l1_loss(
                poses_predict[:, -window:, 6:],
                poses_target[:, -window:, 6:],
                reduction="none",
            ).mean(dim=-1)
            pose_loss = weighted_time_mean(pose_loss_by_time, self.config.final_time_boost)

        ori_loss = torch.tensor(0.0, device=poses_predict.device)
        if self.config.extra_final_ori_weight > 0:
            ori_loss_by_time = F.smooth_l1_loss(
                poses_predict[:, -window:, :3],
                poses_target[:, -window:, :3],
                reduction="none",
            ).mean(dim=-1)
            ori_loss = weighted_time_mean(ori_loss_by_time, self.config.final_time_boost)

        return (
            self.config.extra_dest_trans_weight * dest_trans_loss
            + self.config.extra_dest_mpjpe_weight * dest_mpjpe_loss
            + self.config.extra_final_pose_weight * pose_loss
            + self.config.extra_final_ori_weight * ori_loss
        )

    def train(self):
        train_metric_keys = ["loss_total", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        train_metrics = MetricTracker(*train_metric_keys)

        for epoch in range(self.config.epoch):
            scale = target_scale(self.config, epoch)
            for data in tqdm(self.train_loader):
                (
                    objects,
                    object_mask,
                    gt_object_point_prior,
                    gt_object_point_prior_mask,
                    poses_input,
                    poses_label,
                    joints_input,
                    joints_label,
                    scene_points,
                    seq,
                    scene,
                ) = data

                objects = objects.to(self.device)
                object_mask = object_mask.to(self.device)
                gt_object_point_prior = gt_object_point_prior.to(self.device)
                gt_object_point_prior_mask = gt_object_point_prior_mask.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)

                poses_predict, joints_predict = self.model(
                    poses_input,
                    joints_input[:, :, :23],
                    scene_points,
                    objects,
                    object_mask,
                    gt_object_point_prior,
                    gt_object_point_prior_mask,
                )

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
                        poses_predict,
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
                    joints_predict,
                    joints_label[:, :, :23],
                    joints_input[:, :, :23],
                )
                motion_loss = (
                    pose_loss
                    + loss_trans_gcn.mean()
                    + loss_des_trans_gcn
                    + mpjpe_gcn.mean()
                    + des_mpjpe_gcn
                )
                extra_loss = scale * self.final_target_loss(
                    poses_predict,
                    joints_predict,
                    poses_input,
                    poses_label,
                    joints_input,
                    joints_label,
                )
                loss = motion_loss + extra_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_size = objects.shape[0]
                train_metrics.update("loss_total", loss, batch_size)
                train_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), batch_size)
                train_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
                train_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), batch_size)
                train_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch)
            train_metrics.reset()
            self.save_checkpoint("last.pth")

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    val_result = self.test(epoch)
                self.model.train()
                should_stop = self.update_best_checkpoints(epoch, val_result)
                if should_stop:
                    break
            if epoch % self.config.save_fre == 0:
                self.save_checkpoint(f"{epoch}.pth")


if __name__ == "__main__":
    config = FinalTargetConfig().parse_args()
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    evaluator = FinalTargetFinetuneEvaluator(config)
    evaluator.train()
    print("training done in", time.time() - start)
