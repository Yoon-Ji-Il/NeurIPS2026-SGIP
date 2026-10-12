import os
import random
import time

import numpy as np
import smplx
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_dual import MotionFromGazeDualConfig
from dataset import gimo_dataset_dual
from human_body_prior.tools.model_loader import load_vposer
from model.sif3d_final import SIF3D
from utils.bone_loss import BONE_PAIRS, calc_loss_bone
from utils.dct_loss import calc_loss_dct
from utils.kd_loss import feature_distill
from utils.logger import MetricTracker, create_logger


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class SMPLX_evalutor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vposer, _ = load_vposer(self.config.vposer_path, vp_model="snapshot")
        self.vposer = vposer.to(self.device)
        self.body_model = smplx.create(
            self.config.smplx_path,
            model_type="smplx",
            gender="neutral",
            ext="npz",
            num_pca_comps=12,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            num_betas=10,
            num_expression_coeffs=10,
            batch_size=config.batch_size * (config.output_seq_len + config.input_seq_len),
        ).to(self.device)

        self.model = SIF3D(config, self.vposer, self.body_model).to(self.device)

        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir, map_location=self.device, weights_only=False)
            # VPoser is trainable, so restore its learned decoder weights.
            # SMPL-X keeps its freshly-created state because its internal
            # parameters are batch-size dependent and are intentionally not
            # restored from the training checkpoint.
            state_dict = {k: v for k, v in state_dict.items() if "smplx_model" not in k}
            self.model.load_state_dict(state_dict, strict=False)

        os.makedirs(self.config.save_path, exist_ok=True)
        self.logger = create_logger(self.config.save_path)

        if config.use_kd:
            DatasetCls = gimo_dataset_dual.EgoEvalTeacherKDDataset
        else:
            DatasetCls = gimo_dataset_dual.EgoEvalDataset
        self.train_dataset = DatasetCls(config, train=True)
        self.test_dataset = DatasetCls(config, train=False)
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(config.seed)
        self.test_generator = torch.Generator()
        self.test_generator.manual_seed(config.seed + 1)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=self.train_generator,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=max(1, config.num_workers // 2),
            drop_last=config.drop_last_eval,
            worker_init_fn=seed_worker,
            generator=self.test_generator,
        )
        self.logger.info(
            "Data loaders: train_samples=%d train_batches=%d test_samples=%d test_batches=%d drop_last_eval=%s",
            len(self.train_dataset),
            len(self.train_loader),
            len(self.test_dataset),
            len(self.test_loader),
            config.drop_last_eval,
        )

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.teacher_model = None
        if config.use_kd:
            if config.teacher_load_model_dir is None:
                raise ValueError(
                    "--teacher_load_model_dir is required when --use_kd is set."
                )
            self.teacher_model = SIF3D(config, self.vposer, self.body_model).to(self.device)
            teacher_state = torch.load(
                config.teacher_load_model_dir,
                map_location=self.device,
                weights_only=False,
            )
            teacher_state = {
                k: v for k, v in teacher_state.items()
                if ("smplx_model" not in k and "vposer" not in k)
            }
            self.teacher_model.load_state_dict(teacher_state, strict=False)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.logger.info(
                "KD enabled: teacher=%s teacher_object=%s",
                config.teacher_load_model_dir,
                config.teacher_object_points_name,
            )
            self.logger.info(
                "KD weights: out_tia=%.2f out_sca=%.2f object=%.2f motion_enc=%.2f "
                "out_traj=%.2f out_pose=%.2f kind=%s",
                config.kd_feat_out_tia_weight,
                config.kd_feat_out_sca_weight,
                config.kd_feat_object_weight,
                config.kd_feat_motion_enc_weight,
                config.kd_out_traj_weight,
                config.kd_out_pose_weight,
                config.kd_feat_kind,
            )

        self.best_metrics = {
            "loss_trans": float("inf"),
            "loss_des_trans": float("inf"),
            "mpjpe": float("inf"),
            "des_mpjpe": float("inf"),
        }
        self.early_stop_best = float("inf")
        self.early_stop_bad_epochs = 0

        self.logger.info(f"Seed fixed: seed={self.config.seed}, deterministic={self.config.deterministic}")
        if self.config.use_kd:
            self.logger.info("KD distillation enabled.")
        else:
            self.logger.info("Supervised training enabled: no teacher/KD loss is used.")
        self.logger.info(
            "student_v0_no_object=%s object_points_name=%s object_mask_name=%s",
            getattr(self.config, "student_v0_no_object", False),
            self.config.object_points_name,
            self.config.object_mask_name,
        )
        if getattr(self.config, "student_v0_no_object", False):
            self.logger.info(
                "V0 no-object mode: object tensors may still be loaded by the dataset, "
                "but object embedding and object-derived priors are disabled in SIF3D."
            )

        if self.config.use_pose_loss:
            self.logger.info("Pose loss enabled: using single-stream-style pose and SMPL-X losses in dual training.")
        if self.config.use_dct_loss:
            self.logger.info(
                "DCT loss enabled: metric=loss_dct, weight=%s, K=%s, kind=%s",
                self.config.dct_loss_weight,
                self.config.dct_num_coeffs,
                self.config.dct_loss_type,
            )
        if self.config.use_bone_loss:
            self.bone_pairs = torch.tensor(BONE_PAIRS, dtype=torch.long, device=self.device)
            self.logger.info(
                "Bone loss enabled: weight=%s, kind=%s, num_bones=%d",
                self.config.bone_loss_weight,
                self.config.bone_loss_type,
                len(BONE_PAIRS),
            )

        if self.config.early_stop_patience > 0:
            self.logger.info(
                "Early stopping enabled: "
                f"metric={self.config.early_stop_metric}, "
                f"patience={self.config.early_stop_patience}, "
                f"min_delta={self.config.early_stop_min_delta}, "
                f"warmup={self.config.early_stop_warmup}"
            )

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, filename))

    def train(self):
        train_metric_keys = ["loss_total", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        if self.config.use_dct_loss:
            train_metric_keys.append("loss_dct")
        if self.config.use_bone_loss:
            train_metric_keys.append("loss_bone")
        if self.config.use_kd:
            train_metric_keys.append("loss_kd")
        train_metrics = MetricTracker(*train_metric_keys)

        for epoch in range(self.config.epoch):
            for data in tqdm(self.train_loader):
                if self.config.use_kd:
                    (
                        objects,
                        object_mask,
                        teacher_objects,
                        teacher_object_mask,
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
                    teacher_objects = teacher_objects.to(self.device)
                    teacher_object_mask = teacher_object_mask.to(self.device)
                else:
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
                    teacher_objects = None
                    teacher_object_mask = None

                objects = objects.to(self.device)
                object_mask = object_mask.to(self.device)
                gt_object_point_prior = gt_object_point_prior.to(self.device)
                gt_object_point_prior_mask = gt_object_point_prior_mask.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)

                if self.config.use_kd:
                    poses_predict, joints_predict, student_feats = self.model(
                        poses_input,
                        joints_input[:, :, :23],
                        scene_points,
                        objects,
                        object_mask,
                        gt_object_point_prior,
                        gt_object_point_prior_mask,
                        return_features=True,
                    )
                else:
                    poses_predict, joints_predict = self.model(
                        poses_input,
                        joints_input[:, :, :23],
                        scene_points,
                        objects,
                        object_mask,
                        gt_object_point_prior,
                        gt_object_point_prior_mask,
                    )
                    student_feats = None

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
                if self.config.use_dct_loss:
                    loss_dct = calc_loss_dct(
                        joints_predict,
                        joints_label[:, :, :23],
                        joints_input[:, :, :23],
                        K=self.config.dct_num_coeffs,
                        kind=self.config.dct_loss_type,
                    )
                if self.config.use_bone_loss:
                    loss_bone = calc_loss_bone(
                        joints_predict,
                        joints_label[:, :, :23],
                        joints_input[:, :, :23],
                        self.bone_pairs,
                        kind=self.config.bone_loss_type,
                    )

                teacher_feats = None
                if self.teacher_model is not None and self.config.use_kd:
                    with torch.no_grad():
                        _, _, teacher_feats = self.teacher_model(
                            poses_input,
                            joints_input[:, :, :23],
                            scene_points,
                            teacher_objects,
                            teacher_object_mask,
                            gt_object_point_prior,
                            gt_object_point_prior_mask,
                            return_features=True,
                        )

                motion_loss = (
                    pose_loss
                    + loss_trans_gcn.mean()
                    + loss_des_trans_gcn
                    + mpjpe_gcn.mean()
                    + des_mpjpe_gcn
                )
                loss = motion_loss
                if self.config.use_dct_loss:
                    loss = loss + self.config.dct_loss_weight * loss_dct
                if self.config.use_bone_loss:
                    loss = loss + self.config.bone_loss_weight * loss_bone
                if self.config.use_kd:
                    loss = loss + self.kd_loss(student_feats, teacher_feats)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_size = objects.shape[0]
                train_metrics.update("loss_total", loss, batch_size)
                train_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), batch_size)
                train_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
                train_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), batch_size)
                train_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)
                if self.config.use_dct_loss:
                    train_metrics.update("loss_dct", loss_dct, batch_size)
                if self.config.use_bone_loss:
                    train_metrics.update("loss_bone", loss_bone, batch_size)
                if self.config.use_kd:
                    loss_kd_value = self.kd_loss(student_feats, teacher_feats)
                    train_metrics.update("loss_kd", loss_kd_value, batch_size)

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

    def update_best_checkpoints(self, epoch, val_result):
        for metric_name, filename in (
            ("loss_trans", "best_loss_trans.pth"),
            ("loss_des_trans", "best_loss_des_trans.pth"),
            ("mpjpe", "best_mpjpe.pth"),
            ("des_mpjpe", "best_des_mpjpe.pth"),
        ):
            metric_value = val_result[metric_name]
            if metric_value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = metric_value
                self.save_checkpoint(filename)
                self.logger.info(
                    f"Saved {filename} at epoch {epoch} with {metric_name}: {metric_value:.4f}"
                )

        if self.config.early_stop_patience <= 0 or epoch < self.config.early_stop_warmup:
            return False

        monitor_metric = self.config.early_stop_metric
        monitor_value = val_result[monitor_metric]
        improved = monitor_value < (self.early_stop_best - self.config.early_stop_min_delta)

        if improved:
            self.early_stop_best = monitor_value
            self.early_stop_bad_epochs = 0
            self.save_checkpoint("best_early_stop.pth")
            self.logger.info(
                f"Early-stop metric improved at epoch {epoch}: "
                f"{monitor_metric}={monitor_value:.4f}"
            )
            return False

        self.early_stop_bad_epochs += 1
        self.logger.info(
            f"Early-stop patience {self.early_stop_bad_epochs}/{self.config.early_stop_patience} "
            f"without improvement on {monitor_metric} "
            f"(best={self.early_stop_best:.4f}, current={monitor_value:.4f})"
        )

        if self.early_stop_bad_epochs >= self.config.early_stop_patience:
            self.logger.info(
                f"Early stopping triggered at epoch {epoch} on {monitor_metric}."
            )
            return True

        return False

    def test(self, epoch):
        test_metric_keys = ["loss_total", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        if self.config.use_dct_loss:
            test_metric_keys.append("loss_dct")
        if self.config.use_bone_loss:
            test_metric_keys.append("loss_bone")
        if self.config.use_kd:
            test_metric_keys.append("loss_kd")
        test_metrics = MetricTracker(*test_metric_keys)

        for data in self.test_loader:
            if self.config.use_kd:
                (
                    objects,
                    object_mask,
                    teacher_objects,
                    teacher_object_mask,
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
                teacher_objects = teacher_objects.to(self.device)
                teacher_object_mask = teacher_object_mask.to(self.device)
            else:
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
                teacher_objects = None
                teacher_object_mask = None

            objects = objects.to(self.device)
            object_mask = object_mask.to(self.device)
            gt_object_point_prior = gt_object_point_prior.to(self.device)
            gt_object_point_prior_mask = gt_object_point_prior_mask.to(self.device)
            poses_input = poses_input.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)

            if self.config.use_kd:
                _, joints_predict, student_feats = self.model(
                    poses_input,
                    joints_input[:, :, :23],
                    scene_points,
                    objects,
                    object_mask,
                    gt_object_point_prior,
                    gt_object_point_prior_mask,
                    return_features=True,
                )
            else:
                _, joints_predict = self.model(
                    poses_input,
                    joints_input[:, :, :23],
                    scene_points,
                    objects,
                    object_mask,
                    gt_object_point_prior,
                    gt_object_point_prior_mask,
                )
                student_feats = None

            loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
                joints_predict,
                joints_label[:, :, :23],
                joints_input[:, :, :23],
            )
            if self.config.use_dct_loss:
                loss_dct = calc_loss_dct(
                    joints_predict,
                    joints_label[:, :, :23],
                    joints_input[:, :, :23],
                    K=self.config.dct_num_coeffs,
                    kind=self.config.dct_loss_type,
                )
            if self.config.use_bone_loss:
                loss_bone = calc_loss_bone(
                    joints_predict,
                    joints_label[:, :, :23],
                    joints_input[:, :, :23],
                    self.bone_pairs,
                    kind=self.config.bone_loss_type,
                )

            teacher_feats = None
            if self.teacher_model is not None and self.config.use_kd:
                with torch.no_grad():
                    _, _, teacher_feats = self.teacher_model(
                        poses_input,
                        joints_input[:, :, :23],
                        scene_points,
                        teacher_objects,
                        teacher_object_mask,
                        gt_object_point_prior,
                        gt_object_point_prior_mask,
                        return_features=True,
                    )

            motion_loss = (
                loss_trans_gcn.mean()
                + loss_des_trans_gcn
                + mpjpe_gcn.mean()
                + des_mpjpe_gcn
            )
            loss = motion_loss
            if self.config.use_dct_loss:
                loss = loss + self.config.dct_loss_weight * loss_dct
            if self.config.use_bone_loss:
                loss = loss + self.config.bone_loss_weight * loss_bone
            if self.config.use_kd:
                loss = loss + self.kd_loss(student_feats, teacher_feats)

            batch_size = objects.shape[0]
            test_metrics.update("loss_total", loss, batch_size)
            test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), batch_size)
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
            test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), batch_size)
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)
            if self.config.use_dct_loss:
                test_metrics.update("loss_dct", loss_dct, batch_size)
            if self.config.use_bone_loss:
                test_metrics.update("loss_bone", loss_bone, batch_size)
            if self.config.use_kd:
                loss_kd_value = self.kd_loss(student_feats, teacher_feats)
                test_metrics.update("loss_kd", loss_kd_value, batch_size)

        result = test_metrics.result()
        test_metrics.log(self.logger, epoch, train=False)
        test_metrics.reset()
        return result

    def calculate_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)[:, -1].mean()
        loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

        loss_all = F.l1_loss(poses_predict[:, :, :], poses_label[:, :, :], reduction="none")

        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)
        loss_latent = loss_all[:, :, 6:]

        pred_pose = self.vposer.decode(poses_predict[:, :, 6:], output_type="aa").view(-1, 63)
        pred_smplx = self.body_model(
            return_verts=True,
            body_pose=pred_pose,
            global_orient=poses_predict[:, :, :3].view(-1, 3),
            transl=poses_predict[:, :, 3:6].view(-1, 3),
            pose_embedding=poses_predict[:, :, 6:].view(-1, 32),
        ).joints

        gt_smplx = torch.cat([joints_input, joints_label], dim=1)
        pred_smplx = pred_smplx.view(-1, self.config.input_seq_len + self.config.output_seq_len, 127, 3)[:, :, :23]
        gt_smplx -= gt_smplx[:, :, [0]]
        pred_smplx -= pred_smplx[:, :, [0]]

        mpjpe = torch.norm(gt_smplx - pred_smplx, dim=-1)

        return loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, mpjpe[:, -1].mean()

    def kd_loss(self, student_feats, teacher_feats):
        if not self.config.use_kd or teacher_feats is None:
            return torch.tensor(0.0, device=self.device)
        kind = self.config.kd_feat_kind
        loss = torch.tensor(0.0, device=self.device)
        if self.config.kd_feat_out_tia_weight > 0:
            loss = loss + self.config.kd_feat_out_tia_weight * feature_distill(
                student_feats["out_tia"], teacher_feats["out_tia"], kind
            )
        if self.config.kd_feat_out_sca_weight > 0:
            loss = loss + self.config.kd_feat_out_sca_weight * feature_distill(
                student_feats["out_sca"], teacher_feats["out_sca"], kind
            )
        if self.config.kd_feat_object_weight > 0:
            loss = loss + self.config.kd_feat_object_weight * feature_distill(
                student_feats["object_emb"], teacher_feats["object_emb"], kind
            )
        if self.config.kd_feat_motion_enc_weight > 0:
            loss = loss + self.config.kd_feat_motion_enc_weight * feature_distill(
                student_feats["motion_enc"], teacher_feats["motion_enc"], kind
            )
        if self.config.kd_out_traj_weight > 0:
            loss = loss + self.config.kd_out_traj_weight * F.smooth_l1_loss(
                student_feats["ori_trans"], teacher_feats["ori_trans"].detach()
            )
        if self.config.kd_out_pose_weight > 0:
            loss = loss + self.config.kd_out_pose_weight * F.smooth_l1_loss(
                student_feats["pose_latent"], teacher_feats["pose_latent"].detach()
            )
        return loss

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)

        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == "__main__":
    config = MotionFromGazeDualConfig().parse_args()
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    evaluator = SMPLX_evalutor(config)
    evaluator.train()
    print("training done in", time.time() - start)
