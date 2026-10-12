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

from dataset.gimo_dataset_dual_teacher_kd import EgoEvalTeacherKDDataset
from human_body_prior.tools.model_loader import load_vposer
from model.sif3d_final import SIF3D
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


def _load_state_dict(path, device):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    return {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}


class SMPLXKDTrainer:
    def __init__(self, config, kd_name, kd_loss_fn):
        self.config = config
        self.kd_name = kd_name
        self.kd_loss_fn = kd_loss_fn
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
            self.model.load_state_dict(_load_state_dict(self.config.load_model_dir, self.device), strict=False)

        self.teacher_model = SIF3D(config, self.vposer, self.body_model).to(self.device)
        self.teacher_model.load_state_dict(_load_state_dict(self.config.teacher_load_model_dir, self.device), strict=False)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        os.makedirs(self.config.save_path, exist_ok=True)
        self.logger = create_logger(self.config.save_path)

        self.train_dataset = EgoEvalTeacherKDDataset(config, train=True)
        self.test_dataset = EgoEvalTeacherKDDataset(config, train=False)
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
            "KD=%s train_samples=%d train_batches=%d test_samples=%d test_batches=%d",
            self.kd_name,
            len(self.train_dataset),
            len(self.train_loader),
            len(self.test_dataset),
            len(self.test_loader),
        )

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.best_metrics = {
            "loss_trans": float("inf"),
            "loss_des_trans": float("inf"),
            "mpjpe": float("inf"),
            "des_mpjpe": float("inf"),
        }
        self.early_stop_best = float("inf")
        self.early_stop_bad_epochs = 0

        self.logger.info("Seed fixed: seed=%s deterministic=%s", self.config.seed, self.config.deterministic)
        self.logger.info("Teacher KD enabled with frozen checkpoint: %s", self.config.teacher_load_model_dir)
        self.logger.info("Teacher object points: %s", self.config.teacher_object_points_name)
        self.logger.info("Teacher conditioning source: %s", self.teacher_condition_name())
        self.logger.info("KD only training: %s", getattr(self.config, "kd_only", False))

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, filename))

    def teacher_condition_name(self):
        return "teacher_objects"

    def teacher_condition_inputs(self, objects, object_mask, teacher_objects, teacher_object_mask):
        return teacher_objects, teacher_object_mask

    def unpack_batch(self, data):
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
        return (
            objects.to(self.device),
            object_mask.to(self.device),
            teacher_objects.to(self.device),
            teacher_object_mask.to(self.device),
            gt_object_point_prior.to(self.device),
            gt_object_point_prior_mask.to(self.device),
            poses_input.to(self.device),
            poses_label.to(self.device),
            joints_input.to(self.device),
            joints_label.to(self.device),
            scene_points.to(self.device).contiguous(),
            seq,
            scene,
        )

    def forward_model(self, model, poses_input, joints_input, scene_points, objects, object_mask, gt_prior, gt_prior_mask):
        latent, joints = model(
            poses_input,
            joints_input[:, :, :23],
            scene_points,
            objects,
            object_mask,
            gt_prior,
            gt_prior_mask,
        )
        return {
            "latent": latent,
            "joints": joints,
            "ori": latent[:, :, :3],
            "trans": latent[:, :, 3:6],
            "pose": latent[:, :, 6:],
        }

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
        kd_loss = self.kd_loss_fn(self.config, student_out, teacher_out)
        if getattr(self.config, "kd_only", False):
            total_loss = kd_loss
        else:
            total_loss = gt_loss + kd_loss
        return total_loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn

    def train(self):
        metric_keys = ["loss_total", "loss_kd", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        train_metrics = MetricTracker(*metric_keys)

        for epoch in range(self.config.epoch):
            self.config._current_epoch = epoch
            for data in tqdm(self.train_loader):
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
                ) = self.unpack_batch(data)

                student_out = self.forward_model(
                    self.model,
                    poses_input,
                    joints_input,
                    scene_points,
                    objects,
                    object_mask,
                    gt_object_point_prior,
                    gt_object_point_prior_mask,
                )
                teacher_condition_objects, teacher_condition_mask = self.teacher_condition_inputs(
                    objects,
                    object_mask,
                    teacher_objects,
                    teacher_object_mask,
                )
                with torch.no_grad():
                    teacher_out = self.forward_model(
                        self.teacher_model,
                        poses_input,
                        joints_input,
                        scene_points,
                        teacher_condition_objects,
                        teacher_condition_mask,
                        gt_object_point_prior,
                        gt_object_point_prior_mask,
                    )

                loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.compute_total_loss(
                    student_out,
                    teacher_out,
                    poses_input,
                    poses_label,
                    joints_input,
                    joints_label,
                )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_size = objects.shape[0]
                train_metrics.update("loss_total", loss, batch_size)
                train_metrics.update("loss_kd", kd_loss, batch_size)
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
                self.logger.info("Saved %s at epoch %d with %s: %.4f", filename, epoch, metric_name, metric_value)

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
                "Early-stop metric improved at epoch %d: %s=%.4f",
                epoch,
                monitor_metric,
                monitor_value,
            )
            return False

        self.early_stop_bad_epochs += 1
        self.logger.info(
            "Early-stop patience %d/%d without improvement on %s (best=%.4f, current=%.4f)",
            self.early_stop_bad_epochs,
            self.config.early_stop_patience,
            monitor_metric,
            self.early_stop_best,
            monitor_value,
        )
        if self.early_stop_bad_epochs >= self.config.early_stop_patience:
            self.logger.info("Early stopping triggered at epoch %d on %s.", epoch, monitor_metric)
            return True
        return False

    def test(self, epoch):
        metric_keys = ["loss_total", "loss_kd", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        test_metrics = MetricTracker(*metric_keys)
        self.config._current_epoch = epoch

        for data in self.test_loader:
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
            ) = self.unpack_batch(data)

            student_out = self.forward_model(
                self.model,
                poses_input,
                joints_input,
                scene_points,
                objects,
                object_mask,
                gt_object_point_prior,
                gt_object_point_prior_mask,
            )
            teacher_condition_objects, teacher_condition_mask = self.teacher_condition_inputs(
                objects,
                object_mask,
                teacher_objects,
                teacher_object_mask,
            )
            teacher_out = self.forward_model(
                self.teacher_model,
                poses_input,
                joints_input,
                scene_points,
                teacher_condition_objects,
                teacher_condition_mask,
                gt_object_point_prior,
                gt_object_point_prior_mask,
            )
            loss, kd_loss, loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.compute_total_loss(
                student_out,
                teacher_out,
                poses_input,
                poses_label,
                joints_input,
                joints_label,
            )

            batch_size = objects.shape[0]
            test_metrics.update("loss_total", loss, batch_size)
            test_metrics.update("loss_kd", kd_loss, batch_size)
            test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), batch_size)
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
            test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), batch_size)
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)

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

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)
        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


class SMPLXKDStudentConditionedTeacherTrainer(SMPLXKDTrainer):
    def teacher_condition_name(self):
        return "student_objects"

    def teacher_condition_inputs(self, objects, object_mask, teacher_objects, teacher_object_mask):
        del teacher_objects, teacher_object_mask
        return objects, object_mask


def run_kd_training(config, kd_name, kd_loss_fn):
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    trainer = SMPLXKDTrainer(config, kd_name, kd_loss_fn)
    trainer.train()
    print("training done in", time.time() - start)


def run_kd_training_student_conditioned(config, kd_name, kd_loss_fn):
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    trainer = SMPLXKDStudentConditionedTeacherTrainer(config, kd_name, kd_loss_fn)
    trainer.train()
    print("training done in", time.time() - start)
