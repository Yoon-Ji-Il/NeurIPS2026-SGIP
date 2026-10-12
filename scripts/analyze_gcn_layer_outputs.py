import argparse
import csv
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import smplx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_dual import MotionFromGazeDualConfig
from dataset import gimo_dataset_dual
from human_body_prior.tools.model_loader import load_vposer
from model.sif3d_final import SIF3D


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class LayerCaptureGCN(nn.Module):
    """Expose diagnostic GCN outputs after each residual block.

    Each intermediate hidden state is projected with the trained final gc7
    layer and then passed through the same IDCT path as the actual decoder.
    These are not independently trained heads; they are a depth diagnostic.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.last_outputs = OrderedDict()

    def _project_to_joints(self, hidden, residual_dct):
        y = self.decoder.gc7(hidden)
        y = y[:, : self.decoder.node] + residual_dct
        return self.decoder.idct(
            y,
            self.decoder.node // 3,
            3,
            dct_used=self.decoder.in_frames + self.decoder.out_frames,
        )

    def forward(self, x):
        self.last_outputs = OrderedDict()

        residual_dct = self.decoder.dct(
            x,
            dct_used=self.decoder.in_frames + self.decoder.out_frames,
        )

        y = self.decoder.gc1(residual_dct)
        bs, node, channels = y.shape
        y = self.decoder.bn1(y.view(bs, -1)).view(bs, node, channels)
        y = self.decoder.act_f(y)
        y = self.decoder.do(y)
        self.last_outputs["after_gc1"] = self._project_to_joints(y, residual_dct)

        for idx, block in enumerate(self.decoder.gcbs, start=1):
            y = block(y)
            self.last_outputs[f"after_block_{idx}"] = self._project_to_joints(
                y,
                residual_dct,
            )

        final = self._project_to_joints(y, residual_dct)
        self.last_outputs["final"] = final
        return final


class MetricAccumulator:
    def __init__(self):
        self.rows = OrderedDict()

    def update(self, name, loss_trans, loss_des_trans, mpjpe, des_mpjpe, batch_size):
        if name not in self.rows:
            self.rows[name] = {
                "n": 0,
                "loss_total": 0.0,
                "loss_trans": 0.0,
                "loss_des_trans": 0.0,
                "mpjpe": 0.0,
                "des_mpjpe": 0.0,
            }
        loss_total = (
            loss_trans.mean()
            + loss_des_trans
            + mpjpe.mean()
            + des_mpjpe
        )
        row = self.rows[name]
        row["n"] += batch_size
        row["loss_total"] += float(loss_total.detach().cpu()) * batch_size
        row["loss_trans"] += float(loss_trans[:, 6:].mean().detach().cpu()) * batch_size
        row["loss_des_trans"] += float(loss_des_trans.detach().cpu()) * batch_size
        row["mpjpe"] += float(mpjpe[:, 6:].mean().detach().cpu()) * batch_size
        row["des_mpjpe"] += float(des_mpjpe.detach().cpu()) * batch_size

    def result_rows(self):
        out = []
        for layer, row in self.rows.items():
            n = max(row["n"], 1)
            out.append(
                {
                    "layer": layer,
                    "n": row["n"],
                    "loss_total": row["loss_total"] / n,
                    "loss_trans": row["loss_trans"] / n,
                    "loss_des_trans": row["loss_des_trans"] / n,
                    "mpjpe": row["mpjpe"] / n,
                    "des_mpjpe": row["des_mpjpe"] / n,
                }
            )
        return out


def calc_loss_gcn(poses_predict, poses_label, poses_input):
    poses_label = torch.cat([poses_input, poses_label], dim=1)
    loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

    poses_label = poses_label - poses_label[:, :, [0]]
    poses_predict = poses_predict - poses_predict[:, :, [0]]
    mpjpe = torch.norm(poses_predict - poses_label, dim=-1)
    return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


def load_model(config, device):
    vposer, _ = load_vposer(config.vposer_path, vp_model="snapshot")
    vposer = vposer.to(device)

    body_model = smplx.create(
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
    ).to(device)

    model = SIF3D(config, vposer, body_model).to(device)
    if config.load_model_dir is None:
        raise ValueError("--load_model_dir is required")
    state_dict = torch.load(config.load_model_dir, map_location=device, weights_only=False)
    state_dict = {k: v for k, v in state_dict.items() if ("smplx" not in k and "vposer" not in k)}
    model.load_state_dict(state_dict, strict=False)
    model.motion_decoder = LayerCaptureGCN(model.motion_decoder)
    model.eval()
    return model


def evaluate_split(config, model, split, device):
    generator = torch.Generator()
    generator.manual_seed(config.seed + (0 if split == "train" else 1))
    loader = DataLoader(
        gimo_dataset_dual.EgoEvalDataset(config, train=(split == "train")),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    metrics = MetricAccumulator()

    with torch.no_grad():
        for data in loader:
            (
                objects,
                object_mask,
                gt_object_point_prior,
                gt_object_point_prior_mask,
                poses_input,
                _poses_label,
                joints_input,
                joints_label,
                scene_points,
                _seq,
                _scene,
            ) = data

            objects = objects.to(device)
            object_mask = object_mask.to(device)
            gt_object_point_prior = gt_object_point_prior.to(device)
            gt_object_point_prior_mask = gt_object_point_prior_mask.to(device)
            poses_input = poses_input.to(device)
            joints_input = joints_input.to(device)
            joints_label = joints_label.to(device)
            scene_points = scene_points.to(device).contiguous()

            _, _ = model(
                poses_input,
                joints_input[:, :, :23],
                scene_points,
                objects,
                object_mask,
                gt_object_point_prior,
                gt_object_point_prior_mask,
            )

            batch_size = objects.shape[0]
            layer_outputs = model.motion_decoder.last_outputs
            eval_outputs = OrderedDict()

            if config.only_mix:
                layer_names = []
            else:
                layer_names = list(layer_outputs.keys())
            for layer_name in layer_names:
                eval_outputs[layer_name] = layer_outputs[layer_name]

            for mix_pair in config.mix_pair:
                root_layer, pose_layer = mix_pair.split(":", maxsplit=1)
                if root_layer not in layer_outputs:
                    raise KeyError(f"Unknown root layer {root_layer!r}; available: {list(layer_outputs)}")
                if pose_layer not in layer_outputs:
                    raise KeyError(f"Unknown pose layer {pose_layer!r}; available: {list(layer_outputs)}")
                root_joints = layer_outputs[root_layer][:, :, [0]]
                pose_rel = layer_outputs[pose_layer] - layer_outputs[pose_layer][:, :, [0]]
                eval_outputs[f"root_{root_layer}__pose_{pose_layer}"] = root_joints + pose_rel

            for layer_name, pred_joints in eval_outputs.items():
                loss_trans, loss_des_trans, mpjpe, des_mpjpe = calc_loss_gcn(
                    pred_joints[:, :, :23],
                    joints_label[:, :, :23],
                    joints_input[:, :, :23],
                )
                metrics.update(
                    layer_name,
                    loss_trans,
                    loss_des_trans,
                    mpjpe,
                    des_mpjpe,
                    batch_size,
                )

    return metrics.result_rows()


def add_analysis_args(parser):
    parser.add_argument("--analysis_split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--output_csv", default="runs_dh/baseline12_gcn_layer_outputs.csv")
    parser.add_argument(
        "--mix_pair",
        action="append",
        default=[],
        help="Evaluate root trajectory from one captured layer and root-relative pose from another, e.g. after_block_7:after_block_9.",
    )
    parser.add_argument(
        "--only_mix",
        action="store_true",
        help="Print/save only --mix_pair outputs instead of all captured layers.",
    )
    return parser


def main():
    parser = MotionFromGazeDualConfig()
    add_analysis_args(parser)
    config = parser.parse_args()
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    splits = ["train", "test"] if config.analysis_split == "both" else [config.analysis_split]
    all_rows = []
    for split in splits:
        rows = evaluate_split(config, model, split, device)
        for row in rows:
            row = {"split": split, **row}
            all_rows.append(row)

    output_csv = Path(config.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "layer",
                "n",
                "loss_total",
                "loss_trans",
                "loss_des_trans",
                "mpjpe",
                "des_mpjpe",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"saved: {output_csv}")
    for row in all_rows:
        print(
            "{split:5s} {layer:15s} total={loss_total:.4f} "
            "trans={loss_trans:.4f} des_trans={loss_des_trans:.4f} "
            "mpjpe={mpjpe:.4f} des_mpjpe={des_mpjpe:.4f}".format(**row)
        )


if __name__ == "__main__":
    main()
