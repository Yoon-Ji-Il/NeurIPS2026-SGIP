import random
import time
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_dual import MotionFromGazeDualConfig
from dataset import gimo_dataset_dual
from human_body_prior.tools.model_loader import load_vposer
from model.sif3d_final import SIF3D
from utils.logger import MetricTracker


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


class SMPLXEvaluatorMmap:
    def __init__(self, config):
        self.config = config
        print("[eval] build dataloader", flush=True)
        self.test_generator = torch.Generator()
        self.test_generator.manual_seed(config.seed + 1)
        self.test_loader = DataLoader(
            gimo_dataset_dual.EgoEvalDataset(config, train=False),
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=self.test_generator,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[eval] device={self.device}", flush=True)
        print("[eval] load vposer", flush=True)
        self.vposer, _ = load_vposer(self.config.vposer_path, vp_model="snapshot")
        self.vposer = self.vposer.to(self.device)

        print("[eval] build smplx", flush=True)
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
            batch_size=config.output_seq_len + config.input_seq_len,
        ).to(self.device)

    def eval(self):
        print("[eval] build model", flush=True)
        model = SIF3D(self.config, self.vposer, self.body_model).to(self.device)
        assert self.config.load_model_dir is not None
        print(f"[eval] load checkpoint: {self.config.load_model_dir}", flush=True)
        state_dict = torch.load(
            self.config.load_model_dir,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        print("[eval] checkpoint loaded", flush=True)
        state_dict = {k: v for k, v in state_dict.items() if ("smplx" not in k and "vposer" not in k)}
        print("[eval] load state_dict into model", flush=True)
        model.load_state_dict(state_dict, strict=False)
        print("[eval] state_dict loaded into model", flush=True)

        with torch.no_grad():
            model.eval()
            test_metrics = MetricTracker("loss_total", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe")

            print("[eval] start loop", flush=True)
            for data in self.test_loader:
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
                del poses_label, seq, scene

                objects = objects.to(self.device)
                object_mask = object_mask.to(self.device)
                gt_object_point_prior = gt_object_point_prior.to(self.device)
                gt_object_point_prior_mask = gt_object_point_prior_mask.to(self.device)
                poses_input = poses_input.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)

                _, joints_predict = model(
                    poses_input,
                    joints_input[:, :, :23],
                    scene_points,
                    objects,
                    object_mask,
                    gt_object_point_prior,
                    gt_object_point_prior_mask,
                )

                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
                    joints_predict,
                    joints_label[:, :, :23],
                    joints_input[:, :, :23],
                )
                loss = (
                    loss_trans_gcn.mean()
                    + loss_des_trans_gcn
                    + mpjpe_gcn.mean()
                    + des_mpjpe_gcn
                )

                batch_size = objects.shape[0]
                test_metrics.update("loss_total", loss, batch_size)
                test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), batch_size)
                test_metrics.update("loss_des_trans", loss_des_trans_gcn, batch_size)
                test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), batch_size)
                test_metrics.update("des_mpjpe", des_mpjpe_gcn, batch_size)

            result = test_metrics.result()
            print(result)
            test_metrics.reset()

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
    evaluator = SMPLXEvaluatorMmap(config)
    evaluator.eval()
    print("eval done in", time.time() - start)
