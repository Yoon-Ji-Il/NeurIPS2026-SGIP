from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_dual import MotionFromGazeDualConfig  # noqa: E402
from dataset import gimo_dataset_dual  # noqa: E402
from model.sif3d_final import SIF3D  # noqa: E402
from utils.logger import MetricTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = MotionFromGazeDualConfig()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="best_*.pth",
        help="Glob pattern inside --run-dir.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Optional text file. Defaults to <run-dir>/best_eval_results.txt",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional json summary path. Defaults to <run-dir>/best_eval_summary.json",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


class DualCheckpointEvaluator:
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.device = choose_device(config.device)
        print(f"[eval] device={self.device}", flush=True)

        self.test_generator = torch.Generator()
        self.test_generator.manual_seed(config.seed + 1)
        print("[eval] building dataset", flush=True)
        dataset = gimo_dataset_dual.EgoEvalDataset(config, train=False)
        print(f"[eval] dataset ready: test_samples={len(dataset)}", flush=True)
        self.test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=self.test_generator,
        )
        print(f"[eval] dataloader ready: num_workers={config.num_workers}", flush=True)

        print("[eval] loading vposer", flush=True)
        self.vposer, _ = load_vposer(config.vposer_path, vp_model="snapshot")
        self.vposer = self.vposer.to(self.device)
        print("[eval] vposer ready", flush=True)
        print("[eval] creating smplx model", flush=True)
        self.body_model = smplx.create(
            config.smplx_path,
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
        print("[eval] smplx ready", flush=True)

    def _load_model(self, checkpoint_path: Path) -> SIF3D:
        print(f"[eval] loading checkpoint={checkpoint_path.name}", flush=True)
        model = SIF3D(self.config, self.vposer, self.body_model).to(self.device)
        print(f"[eval] model graph ready={checkpoint_path.name}", flush=True)
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        print(f"[eval] state_dict loaded on cpu={checkpoint_path.name}", flush=True)
        state_dict = {k: v for k, v in state_dict.items() if ("smplx" not in k and "vposer" not in k)}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"[eval] checkpoint loaded={checkpoint_path.name}", flush=True)
        return model

    def calc_loss_gcn(
        self,
        poses_predict: torch.Tensor,
        poses_label: torch.Tensor,
        poses_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)
        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()

    @torch.no_grad()
    def evaluate_checkpoint(self, checkpoint_path: Path) -> dict[str, float]:
        model = self._load_model(checkpoint_path)
        test_metrics = MetricTracker("loss_total", "loss_trans", "loss_des_trans", "mpjpe", "des_mpjpe")
        print(f"[eval] starting loop for {checkpoint_path.name}", flush=True)

        for batch_index, data in enumerate(self.test_loader):
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
            if batch_index == 0:
                print(f"[eval] first batch done for {checkpoint_path.name}", flush=True)

        result = test_metrics.result()
        print(f"[eval] finished loop for {checkpoint_path.name}: {result}", flush=True)
        return result


def main() -> None:
    config = parse_args()
    seed_everything(config.seed, config.deterministic)

    run_dir = config.run_dir.resolve()
    checkpoints = sorted(run_dir.glob(config.checkpoint_pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {config.checkpoint_pattern!r} in {run_dir}")

    results_file = config.results_file or (run_dir / "best_eval_results.txt")
    summary_json = config.summary_json or (run_dir / "best_eval_summary.json")

    evaluator = DualCheckpointEvaluator(config)
    rows: list[dict[str, float | str]] = []

    with results_file.open("w", encoding="utf-8") as out:
        for checkpoint_path in checkpoints:
            relative_checkpoint = checkpoint_path.relative_to(REPO_ROOT)
            header = f"===== Evaluating {relative_checkpoint} ====="
            print(header)
            out.write(header + "\n")
            metrics = evaluator.evaluate_checkpoint(checkpoint_path)
            print(metrics)
            out.write(str(metrics) + "\n")
            out.write(f"loss_total={metrics['loss_total']}\n")
            rows.append({"checkpoint": checkpoint_path.name, **metrics})

        best = min(rows, key=lambda row: float(row["loss_total"]))
        out.write("===== Best Overall =====\n")
        out.write(f"best_ckpt={best['checkpoint']}\n")
        out.write(f"best_total={best['loss_total']}\n")
        print("===== Best Overall =====")
        print(f"best_ckpt={best['checkpoint']}")
        print(f"best_total={best['loss_total']}")

    payload = {
        "results": rows,
        "best_overall": best,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
