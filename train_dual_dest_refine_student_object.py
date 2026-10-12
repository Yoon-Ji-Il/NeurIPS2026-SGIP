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
from model.sif3d_dest_refine import SIF3DDestinationRefiner
from model.sif3d_final import SIF3D
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


def load_state_dict_cpu(path):
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}


class DestinationRefineConfig(MotionFromGazeDualConfig):
    def __init__(self):
        super().__init__()
        self.set_defaults(
            lr=2.0e-4,
            weight_decay=1.0e-4,
            gamma=0.98,
            epoch=40,
            val_fre=1,
            save_fre=10,
            early_stop_patience=8,
            early_stop_warmup=3,
            early_stop_metric="loss_des_trans",
        )
        extra = self.add_argument_group("destination_refine")
        extra.add_argument(
            "--dest_refine_variant",
            choices=["root", "pose", "both", "mix_b8_pose_b5"],
            default="root",
        )
        extra.add_argument("--dest_refine_hidden_dim", type=int, default=128)
        extra.add_argument("--dest_root_residual_scale", type=float, default=0.05)
        extra.add_argument("--dest_pose_residual_scale", type=float, default=0.05)
        extra.add_argument("--root_seq_weight", type=float, default=0.10)
        extra.add_argument("--dest_root_weight", type=float, default=2.00)
        extra.add_argument("--pose_seq_weight", type=float, default=0.10)
        extra.add_argument("--dest_pose_weight", type=float, default=2.00)
        extra.add_argument("--anchor_root_weight", type=float, default=0.20)
        extra.add_argument("--anchor_pose_weight", type=float, default=0.20)
        extra.add_argument("--delta_root_weight", type=float, default=0.02)
        extra.add_argument("--delta_pose_weight", type=float, default=0.02)
        extra.add_argument("--mix_target_root_weight", type=float, default=1.00)
        extra.add_argument("--mix_target_pose_weight", type=float, default=1.00)
        extra.add_argument("--layer_mix_root", type=str, default="after_block_8")
        extra.add_argument("--layer_mix_pose", type=str, default="after_block_5")
        extra.add_argument("--teacher_dest_pose_kd_weight", type=float, default=0.0)
        extra.add_argument(
            "--pose_loss_weight",
            type=float,
            default=1.0,
            help="Weight for --use_pose_loss latent/SMPL-X pose supervision copied from teacher train_dual.py.",
        )
        extra.add_argument("--freeze_base", action="store_true", default=True)
        extra.add_argument("--no_freeze_base", dest="freeze_base", action="store_false")
        extra.add_argument("--eval_initial", action="store_true", default=True)
        extra.add_argument("--no_eval_initial", dest="eval_initial", action="store_false")


class DestinationRefineTrainer:
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

        self.model = SIF3DDestinationRefiner(config, self.vposer, self.body_model).to(self.device)
        if config.load_model_dir is None:
            raise ValueError("--load_model_dir must point to the baseline checkpoint.")
        missing, unexpected = self.model.load_state_dict(load_state_dict_cpu(config.load_model_dir), strict=False)
        if config.dest_refine_variant == "mix_b8_pose_b5":
            self.model.enable_layer_capture()
        if config.freeze_base:
            self.model.freeze_base()

        self.teacher_model = None
        needs_teacher_for_dest_kd = (
            config.teacher_load_model_dir is not None
            and config.teacher_dest_pose_kd_weight > 0
        )
        needs_teacher_for_feature_kd = config.use_kd
        if needs_teacher_for_dest_kd or needs_teacher_for_feature_kd:
            if config.teacher_load_model_dir is None:
                raise ValueError(
                    "--teacher_load_model_dir is required when --use_kd is set."
                )
            self.teacher_model = SIF3D(config, self.vposer, self.body_model).to(self.device)
            self.teacher_model.load_state_dict(load_state_dict_cpu(config.teacher_load_model_dir), strict=False)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        self.logger.info("Destination refine variant=%s", config.dest_refine_variant)
        self.logger.info("Loaded baseline checkpoint: %s", config.load_model_dir)
        self.logger.info("load_state_dict missing=%s unexpected=%s", missing, unexpected)
        self.logger.info("freeze_base=%s", config.freeze_base)
        if needs_teacher_for_dest_kd:
            self.logger.info("Teacher destination pose KD enabled: %s", config.teacher_load_model_dir)
            self.logger.info("teacher_dest_pose_kd_weight=%.4f", config.teacher_dest_pose_kd_weight)
        if config.use_kd:
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
        if config.use_pose_loss:
            self.logger.info(
                "Pose loss enabled: using teacher train_dual.py latent/SMPL-X pose loss with weight=%.4f",
                config.pose_loss_weight,
            )
            if config.freeze_base:
                self.logger.warning(
                    "Pose loss is attached to the base latent prediction. With freeze_base=True, "
                    "it is logged/included in loss_total but does not directly update the destination refiner heads. "
                    "Use --no_freeze_base if you want this pose loss to train the base SIF3D layers."
                )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Trainable params: %d / %d", trainable, total)

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
            "Data: train_samples=%d train_batches=%d test_samples=%d test_batches=%d",
            len(self.train_dataset),
            len(self.train_loader),
            len(self.test_dataset),
            len(self.test_loader),
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters.")
        self.optim = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)
        self.best_metrics = {
            "loss_trans": float("inf"),
            "loss_des_trans": float("inf"),
            "mpjpe": float("inf"),
            "des_mpjpe": float("inf"),
        }
        self.early_stop_best = float("inf")
        self.early_stop_bad_epochs = 0

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, filename))

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)
        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()

    def calculate_pose_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        batch_size, seq_len = poses_predict.shape[:2]
        smplx_batch_size = batch_size * seq_len
        dtype = poses_predict.dtype
        device = poses_predict.device

        loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)[:, -1].mean()
        loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

        loss_all = F.l1_loss(poses_predict, poses_label, reduction="none")
        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)
        loss_latent = loss_all[:, :, 6:]

        pred_pose = self.vposer.decode(poses_predict[:, :, 6:], output_type="aa").view(-1, 63)
        pred_smplx = self.body_model(
            return_verts=True,
            body_pose=pred_pose,
            global_orient=poses_predict[:, :, :3].reshape(-1, 3),
            transl=poses_predict[:, :, 3:6].reshape(-1, 3),
            pose_embedding=poses_predict[:, :, 6:].reshape(-1, 32),
            betas=torch.zeros(
                smplx_batch_size,
                getattr(self.body_model, "num_betas", 10),
                dtype=dtype,
                device=device,
            ),
            left_hand_pose=torch.zeros(
                smplx_batch_size,
                getattr(self.body_model, "num_pca_comps", 12),
                dtype=dtype,
                device=device,
            ),
            right_hand_pose=torch.zeros(
                smplx_batch_size,
                getattr(self.body_model, "num_pca_comps", 12),
                dtype=dtype,
                device=device,
            ),
            expression=torch.zeros(
                smplx_batch_size,
                getattr(self.body_model, "num_expression_coeffs", 10),
                dtype=dtype,
                device=device,
            ),
            jaw_pose=torch.zeros(smplx_batch_size, 3, dtype=dtype, device=device),
            leye_pose=torch.zeros(smplx_batch_size, 3, dtype=dtype, device=device),
            reye_pose=torch.zeros(smplx_batch_size, 3, dtype=dtype, device=device),
        ).joints

        gt_smplx = torch.cat([joints_input, joints_label], dim=1)
        pred_smplx = pred_smplx.view(
            batch_size,
            seq_len,
            127,
            3,
        )[:, :, :23]
        gt_smplx = gt_smplx - gt_smplx[:, :, [0]]
        pred_smplx = pred_smplx - pred_smplx[:, :, [0]]
        mpjpe = torch.norm(gt_smplx - pred_smplx, dim=-1)

        return (
            loss_ori.mean()
            + loss_trans.mean()
            + loss_latent.mean()
            + loss_des_ori
            + loss_des_trans
            + loss_des_latent
            + mpjpe.mean()
            + mpjpe[:, -1].mean()
        )

    def supervised_dest_loss(self, joints_predict, joints_input, joints_label):
        target = torch.cat([joints_input, joints_label], dim=1)
        future_slice = slice(self.config.input_seq_len, None)

        root_dist = torch.norm(joints_predict[:, :, 0] - target[:, :, 0], dim=-1)
        pred_rel = joints_predict - joints_predict[:, :, [0]]
        target_rel = target - target[:, :, [0]]
        rel_dist = torch.norm(pred_rel[:, :, 1:] - target_rel[:, :, 1:], dim=-1)

        return (
            self.config.root_seq_weight * root_dist[:, future_slice].mean()
            + self.config.dest_root_weight * root_dist[:, -1].mean()
            + self.config.pose_seq_weight * rel_dist[:, future_slice].mean()
            + self.config.dest_pose_weight * rel_dist[:, -1].mean()
        )

    def anchor_loss(self, joints_predict):
        base = self.model.last_base_joints
        if base is None:
            return torch.tensor(0.0, device=joints_predict.device)
        future_slice = slice(self.config.input_seq_len, None)
        total = torch.tensor(0.0, device=joints_predict.device)
        if self.config.anchor_root_weight > 0:
            total = total + self.config.anchor_root_weight * F.smooth_l1_loss(
                joints_predict[:, future_slice, 0],
                base[:, future_slice, 0],
            )
        if self.config.anchor_pose_weight > 0:
            pred_rel = joints_predict - joints_predict[:, :, [0]]
            base_rel = base - base[:, :, [0]]
            total = total + self.config.anchor_pose_weight * F.smooth_l1_loss(
                pred_rel[:, future_slice, 1:],
                base_rel[:, future_slice, 1:],
            )
        return total

    def delta_loss(self):
        total = torch.tensor(0.0, device=self.device)
        if self.model.last_root_delta is not None and self.config.delta_root_weight > 0:
            total = total + self.config.delta_root_weight * self.model.last_root_delta.pow(2).mean()
        if self.model.last_pose_delta is not None and self.config.delta_pose_weight > 0:
            total = total + self.config.delta_pose_weight * self.model.last_pose_delta.pow(2).mean()
        return total

    def mix_target_loss(self, joints_predict):
        mix_target = self.model.last_mix_target
        if mix_target is None:
            return torch.tensor(0.0, device=joints_predict.device)
        total = torch.tensor(0.0, device=joints_predict.device)
        if self.config.mix_target_root_weight > 0:
            total = total + self.config.mix_target_root_weight * F.smooth_l1_loss(
                joints_predict[:, -1, 0],
                mix_target[:, -1, 0],
            )
        if self.config.mix_target_pose_weight > 0:
            pred_rel = joints_predict[:, -1] - joints_predict[:, -1, [0]]
            target_rel = mix_target[:, -1] - mix_target[:, -1, [0]]
            total = total + self.config.mix_target_pose_weight * F.smooth_l1_loss(
                pred_rel[:, 1:],
                target_rel[:, 1:],
            )
        return total

    def teacher_pose_kd_loss(self, joints_predict, teacher_joints):
        if teacher_joints is None or self.config.teacher_dest_pose_kd_weight <= 0:
            return torch.tensor(0.0, device=joints_predict.device)
        pred_rel = joints_predict[:, -1] - joints_predict[:, -1, [0]]
        teacher_rel = teacher_joints[:, -1] - teacher_joints[:, -1, [0]]
        return self.config.teacher_dest_pose_kd_weight * F.smooth_l1_loss(
            pred_rel[:, 1:],
            teacher_rel[:, 1:],
        )

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

    def total_loss(
        self,
        joints_predict,
        joints_input,
        joints_label,
        teacher_joints=None,
        pose_loss=None,
        student_feats=None,
        teacher_feats=None,
    ):
        loss = self.supervised_dest_loss(joints_predict, joints_input, joints_label)
        loss = loss + self.anchor_loss(joints_predict) + self.delta_loss()
        loss = loss + self.teacher_pose_kd_loss(joints_predict, teacher_joints)
        if pose_loss is not None:
            loss = loss + self.config.pose_loss_weight * pose_loss
        if self.config.use_kd:
            loss = loss + self.kd_loss(student_feats, teacher_feats)
        if self.config.dest_refine_variant == "mix_b8_pose_b5":
            loss = loss + self.mix_target_loss(joints_predict)
        return loss

    def forward_batch(self, data):
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
                _seq,
                _scene,
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
                _seq,
                _scene,
            ) = data
            teacher_objects = None
            teacher_object_mask = None
        objects = objects.to(self.device)
        object_mask = object_mask.to(self.device)
        gt_object_point_prior = gt_object_point_prior.to(self.device)
        gt_object_point_prior_mask = gt_object_point_prior_mask.to(self.device)
        poses_input = poses_input.to(self.device)
        poses_label = poses_label.to(self.device)
        joints_input = joints_input.to(self.device)
        joints_label = joints_label.to(self.device)
        scene_points = scene_points.to(self.device).contiguous()

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
        teacher_joints = None
        teacher_feats = None
        if self.teacher_model is not None:
            with torch.no_grad():
                if self.config.use_kd:
                    _, teacher_joints, teacher_feats = self.teacher_model(
                        poses_input,
                        joints_input[:, :, :23],
                        scene_points,
                        teacher_objects,
                        teacher_object_mask,
                        gt_object_point_prior,
                        gt_object_point_prior_mask,
                        return_features=True,
                    )
                else:
                    _, teacher_joints = self.teacher_model(
                        poses_input,
                        joints_input[:, :, :23],
                        scene_points,
                        objects,
                        object_mask,
                        gt_object_point_prior,
                        gt_object_point_prior_mask,
                    )
        return (
            objects.shape[0],
            poses_predict,
            joints_predict,
            poses_input,
            poses_label,
            joints_input[:, :, :23],
            joints_label[:, :, :23],
            teacher_joints,
            student_feats,
            teacher_feats,
        )

    def train(self):
        metric_keys = ["loss_total", "loss_pose", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        if self.config.use_kd:
            metric_keys.append("loss_kd")
        train_metrics = MetricTracker(*metric_keys)

        if self.config.eval_initial:
            self.model.eval()
            with torch.no_grad():
                val_result = self.test(-1)
            self.update_best_checkpoints(-1, val_result)

        for epoch in range(self.config.epoch):
            self.set_train_mode()
            for data in tqdm(self.train_loader):
                (
                    batch_size,
                    poses_predict,
                    joints_predict,
                    poses_input,
                    poses_label,
                    joints_input,
                    joints_label,
                    teacher_joints,
                    student_feats,
                    teacher_feats,
                ) = self.forward_batch(data)
                pose_loss = None
                if self.config.use_pose_loss:
                    pose_loss = self.calculate_pose_loss(
                        poses_predict,
                        poses_label,
                        poses_input,
                        joints_input,
                        joints_label,
                    )
                loss = self.total_loss(
                    joints_predict,
                    joints_input,
                    joints_label,
                    teacher_joints,
                    pose_loss,
                    student_feats=student_feats,
                    teacher_feats=teacher_feats,
                )
                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
                    joints_predict,
                    joints_label,
                    joints_input,
                )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("loss_total", loss, batch_size)
                train_metrics.update(
                    "loss_pose",
                    pose_loss if pose_loss is not None else torch.tensor(0.0, device=self.device),
                    batch_size,
                )
                if self.config.use_kd:
                    loss_kd_value = self.kd_loss(student_feats, teacher_feats)
                    train_metrics.update("loss_kd", loss_kd_value, batch_size)
                train_metrics.update("loss_trans", loss_trans_gcn[:, self.config.input_seq_len :].mean(), batch_size)
                train_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
                train_metrics.update("mpjpe", mpjpe_gcn[:, self.config.input_seq_len :].mean(), batch_size)
                train_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch)
            train_metrics.reset()
            self.save_checkpoint("last.pth")

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    val_result = self.test(epoch)
                should_stop = self.update_best_checkpoints(epoch, val_result)
                if should_stop:
                    break
            if epoch % self.config.save_fre == 0:
                self.save_checkpoint(f"{epoch}.pth")

    def set_train_mode(self):
        self.model.train()
        if self.config.freeze_base:
            self.model.eval()
            self.model.dest_root_refiner.train()
            self.model.dest_pose_refiner.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def test(self, epoch):
        metric_keys = ["loss_total", "loss_pose", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe"]
        if self.config.use_kd:
            metric_keys.append("loss_kd")
        test_metrics = MetricTracker(*metric_keys)
        for data in self.test_loader:
            (
                batch_size,
                poses_predict,
                joints_predict,
                poses_input,
                poses_label,
                joints_input,
                joints_label,
                teacher_joints,
                student_feats,
                teacher_feats,
            ) = self.forward_batch(data)
            pose_loss = None
            if self.config.use_pose_loss:
                pose_loss = self.calculate_pose_loss(
                    poses_predict,
                    poses_label,
                    poses_input,
                    joints_input,
                    joints_label,
                )
            loss = self.total_loss(
                joints_predict,
                joints_input,
                joints_label,
                teacher_joints,
                pose_loss,
                student_feats=student_feats,
                teacher_feats=teacher_feats,
            )
            loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
                joints_predict,
                joints_label,
                joints_input,
            )
            test_metrics.update("loss_total", loss, batch_size)
            test_metrics.update(
                "loss_pose",
                pose_loss if pose_loss is not None else torch.tensor(0.0, device=self.device),
                batch_size,
            )
            if self.config.use_kd:
                loss_kd_value = self.kd_loss(student_feats, teacher_feats)
                test_metrics.update("loss_kd", loss_kd_value, batch_size)
            test_metrics.update("loss_trans", loss_trans_gcn[:, self.config.input_seq_len :].mean(), batch_size)
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
            test_metrics.update("mpjpe", mpjpe_gcn[:, self.config.input_seq_len :].mean(), batch_size)
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)

        result = test_metrics.result()
        test_metrics.log(self.logger, epoch, train=False)
        test_metrics.reset()
        return result

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

        monitor_value = val_result[self.config.early_stop_metric]
        improved = monitor_value < (self.early_stop_best - self.config.early_stop_min_delta)
        if improved:
            self.early_stop_best = monitor_value
            self.early_stop_bad_epochs = 0
            self.save_checkpoint("best_early_stop.pth")
            self.logger.info(
                "Early-stop metric improved at epoch %d: %s=%.4f",
                epoch,
                self.config.early_stop_metric,
                monitor_value,
            )
            return False

        self.early_stop_bad_epochs += 1
        self.logger.info(
            "Early-stop patience %d/%d without improvement on %s (best=%.4f, current=%.4f)",
            self.early_stop_bad_epochs,
            self.config.early_stop_patience,
            self.config.early_stop_metric,
            self.early_stop_best,
            monitor_value,
        )
        if self.early_stop_bad_epochs >= self.config.early_stop_patience:
            self.logger.info("Early stopping triggered at epoch %d on %s.", epoch, self.config.early_stop_metric)
            return True
        return False


def run_training(config):
    seed_everything(config.seed, config.deterministic)
    start = time.time()
    trainer = DestinationRefineTrainer(config)
    trainer.train()
    print("training done in", time.time() - start)


if __name__ == "__main__":
    config = DestinationRefineConfig().parse_args()
    run_training(config)
