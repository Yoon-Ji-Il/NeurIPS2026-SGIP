#!/usr/bin/env python
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer
from matplotlib.animation import FuncAnimation, PillowWriter

from config.config_dual import MotionFromGazeDualConfig
from dataset import gimo_dataset_dual
from eval_dual import seed_everything
from model.sif3d_final import SIF3D


BODY23_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (15, 22),
]

SCENE_COLOR = "#c7ced6"
OBJECT_COLOR = "#ff9f43"
INPUT_COLOR = "#111111"
GT_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"


def add_vis_args(parser):
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        default=[0],
        help="Test-set sample indices to visualize.",
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default=None,
        help="Directory for saved visualizations. Defaults near the checkpoint.",
    )
    parser.add_argument(
        "--scene_plot_points",
        type=int,
        default=2500,
        help="How many scene points to draw per sample.",
    )
    parser.add_argument(
        "--object_plot_points",
        type=int,
        default=512,
        help="How many object points to draw per sample.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=4,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="3D camera elevation.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D camera azimuth.",
    )
    return parser


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def sample_points(points, max_points, seed):
    if points is None or len(points) == 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep]


def compute_metrics(pred_seq, gt_seq, input_seq_len):
    pred = torch.from_numpy(pred_seq)
    gt = torch.from_numpy(gt_seq)

    root_l2 = torch.norm(pred[:, 0] - gt[:, 0], dim=-1)

    pred_rel = pred - pred[:, [0]]
    gt_rel = gt - gt[:, [0]]
    mpjpe = torch.norm(pred_rel - gt_rel, dim=-1)

    metrics = {
        "root_mean_future": float(root_l2[input_seq_len:].mean().item()),
        "root_dest": float(root_l2[-1].item()),
        "mpjpe_mean_future": float(mpjpe[input_seq_len:].mean().item()),
        "des_mpjpe": float(mpjpe[-1].mean().item()),
    }
    return metrics


def build_lims(*arrays):
    points = [arr.reshape(-1, 3) for arr in arrays if arr is not None and len(arr) > 0]
    merged = np.concatenate(points, axis=0)
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.75)
    return [(center[i] - radius, center[i] + radius) for i in range(3)]


def plot_skeleton(ax, joints, color, alpha=1.0, linewidth=2.0):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=12, alpha=alpha, depthshade=False)
    for src, dst in BODY23_EDGES:
        segment = joints[[src, dst]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def style_3d_axis(ax, lims, title, elev, azim):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(
        (
            lims[0][1] - lims[0][0],
            lims[1][1] - lims[1][0],
            lims[2][1] - lims[2][0],
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


class DualPredictionVisualizer:
    def __init__(self, config):
        self.config = config
        self.dataset = gimo_dataset_dual.EgoEvalDataset(config, train=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vposer, _ = load_vposer(config.vposer_path, vp_model="snapshot")
        self.vposer = self.vposer.to(self.device)

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

        self.model = SIF3D(config, self.vposer, self.body_model).to(self.device)
        state_dict = torch.load(config.load_model_dir, map_location=self.device, weights_only=False)
        state_dict = {k: v for k, v in state_dict.items() if ("smplx" not in k and "vposer" not in k)}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict(self, sample_idx):
        raw = self.dataset[sample_idx]
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
        ) = raw

        objects_t = objects.unsqueeze(0).to(self.device)
        object_mask_t = object_mask.unsqueeze(0).to(self.device)
        gt_prior_t = gt_object_point_prior.unsqueeze(0).to(self.device)
        gt_prior_mask_t = gt_object_point_prior_mask.unsqueeze(0).to(self.device)
        poses_input_t = poses_input.unsqueeze(0).to(self.device)
        joints_input_t = joints_input.unsqueeze(0).to(self.device)
        scene_points_t = scene_points.unsqueeze(0).to(self.device).contiguous()

        with torch.no_grad():
            _, joints_predict = self.model(
                poses_input_t,
                joints_input_t[:, :, :23],
                scene_points_t,
                objects_t,
                object_mask_t,
                gt_prior_t,
                gt_prior_mask_t,
            )

        gt_seq = torch.cat([joints_input[:, :23], joints_label[:, :23]], dim=0).cpu().numpy()
        pred_seq = joints_predict.cpu().numpy()[0]
        scene_np = scene_points.cpu().numpy()
        object_np = objects.cpu().numpy() if bool(object_mask.item()) else np.zeros((0, 3), dtype=np.float32)

        return {
            "sample_idx": sample_idx,
            "scene": scene,
            "seq": seq,
            "input_seq_len": joints_input.shape[0],
            "gt_seq": gt_seq,
            "pred_seq": pred_seq,
            "scene_points": sample_points(scene_np, self.config.scene_plot_points, seed=sample_idx),
            "object_points": sample_points(object_np, self.config.object_plot_points, seed=sample_idx + 1234),
            "metrics": compute_metrics(pred_seq, gt_seq, joints_input.shape[0]),
        }


def draw_pose_panel(ax, scene_points, object_points, observed_seq, target_seq, frame_idx, input_seq_len, lims, title, color, elev, azim):
    if len(scene_points) > 0:
        ax.scatter(
            scene_points[:, 0],
            scene_points[:, 1],
            scene_points[:, 2],
            c=SCENE_COLOR,
            s=1.5,
            alpha=0.10,
            depthshade=False,
        )
    if len(object_points) > 0:
        ax.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            c=OBJECT_COLOR,
            s=4,
            alpha=0.25,
            depthshade=False,
        )

    observed_now = min(frame_idx, input_seq_len - 1)
    plot_skeleton(ax, observed_seq[observed_now], INPUT_COLOR, alpha=0.25, linewidth=1.2)

    root_obs = observed_seq[: min(frame_idx + 1, input_seq_len), 0]
    ax.plot(root_obs[:, 0], root_obs[:, 1], root_obs[:, 2], color=INPUT_COLOR, linewidth=2.0, alpha=0.8)

    if frame_idx >= input_seq_len:
        target_root = target_seq[input_seq_len: frame_idx + 1, 0]
        if len(target_root) > 0:
            ax.plot(
                target_root[:, 0],
                target_root[:, 1],
                target_root[:, 2],
                color=color,
                linewidth=2.6,
                alpha=0.95,
            )
        plot_skeleton(ax, target_seq[frame_idx], color, alpha=1.0, linewidth=2.2)
    else:
        plot_skeleton(ax, observed_seq[frame_idx], INPUT_COLOR, alpha=0.95, linewidth=2.2)

    style_3d_axis(ax, lims, title, elev, azim)


def save_overview(sample, output_dir, elev, azim, dpi):
    gt_seq = sample["gt_seq"]
    pred_seq = sample["pred_seq"]
    input_seq_len = sample["input_seq_len"]
    scene_points = sample["scene_points"]
    object_points = sample["object_points"]
    lims = build_lims(scene_points, object_points, gt_seq, pred_seq)

    fig = plt.figure(figsize=(15, 5))
    ax_gt = fig.add_subplot(1, 3, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 3, 2, projection="3d")
    ax_top = fig.add_subplot(1, 3, 3)

    draw_pose_panel(
        ax_gt,
        scene_points,
        object_points,
        gt_seq,
        gt_seq,
        len(gt_seq) - 1,
        input_seq_len,
        lims,
        "Ground Truth",
        GT_COLOR,
        elev,
        azim,
    )
    draw_pose_panel(
        ax_pred,
        scene_points,
        object_points,
        gt_seq,
        pred_seq,
        len(pred_seq) - 1,
        input_seq_len,
        lims,
        "Prediction",
        PRED_COLOR,
        elev,
        azim,
    )

    gt_root = gt_seq[:, 0]
    pred_root = pred_seq[:, 0]
    ax_top.scatter(scene_points[:, 0], scene_points[:, 2], c=SCENE_COLOR, s=1.0, alpha=0.10)
    if len(object_points) > 0:
        ax_top.scatter(object_points[:, 0], object_points[:, 2], c=OBJECT_COLOR, s=5, alpha=0.25)
    ax_top.plot(gt_root[:input_seq_len, 0], gt_root[:input_seq_len, 2], color=INPUT_COLOR, linewidth=2.2, label="Observed")
    ax_top.plot(gt_root[input_seq_len - 1:, 0], gt_root[input_seq_len - 1:, 2], color=GT_COLOR, linewidth=2.4, label="GT future")
    ax_top.plot(pred_root[input_seq_len - 1:, 0], pred_root[input_seq_len - 1:, 2], color=PRED_COLOR, linewidth=2.4, label="Pred future")
    ax_top.scatter(gt_root[-1, 0], gt_root[-1, 2], color=GT_COLOR, s=28)
    ax_top.scatter(pred_root[-1, 0], pred_root[-1, 2], color=PRED_COLOR, s=28)
    ax_top.set_title("Top-Down Root Trajectory", fontsize=11)
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("z")
    ax_top.grid(alpha=0.2)
    ax_top.legend(loc="best", fontsize=9)
    ax_top.set_aspect("equal", adjustable="box")

    metrics = sample["metrics"]
    fig.suptitle(
        (
            f"sample={sample['sample_idx']}  scene={sample['scene']}  seq={sample['seq']}\n"
            f"root_dest={metrics['root_dest']:.4f}  des_mpjpe={metrics['des_mpjpe']:.4f}  "
            f"root_mean_future={metrics['root_mean_future']:.4f}  mpjpe_mean_future={metrics['mpjpe_mean_future']:.4f}"
        ),
        fontsize=12,
    )
    fig.tight_layout()

    save_path = output_dir / "overview.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_gif(sample, output_dir, elev, azim, dpi, gif_fps):
    gt_seq = sample["gt_seq"]
    pred_seq = sample["pred_seq"]
    input_seq_len = sample["input_seq_len"]
    scene_points = sample["scene_points"]
    object_points = sample["object_points"]
    lims = build_lims(scene_points, object_points, gt_seq, pred_seq)

    fig = plt.figure(figsize=(10, 5))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")

    metrics = sample["metrics"]
    fig.suptitle(
        (
            f"sample={sample['sample_idx']}  scene={sample['scene']}  seq={sample['seq']}\n"
            f"root_dest={metrics['root_dest']:.4f}  des_mpjpe={metrics['des_mpjpe']:.4f}"
        ),
        fontsize=12,
    )

    def update(frame_idx):
        ax_gt.cla()
        ax_pred.cla()
        draw_pose_panel(
            ax_gt,
            scene_points,
            object_points,
            gt_seq,
            gt_seq,
            frame_idx,
            input_seq_len,
            lims,
            f"Ground Truth  frame={frame_idx}",
            GT_COLOR,
            elev,
            azim,
        )
        draw_pose_panel(
            ax_pred,
            scene_points,
            object_points,
            gt_seq,
            pred_seq,
            frame_idx,
            input_seq_len,
            lims,
            f"Prediction  frame={frame_idx}",
            PRED_COLOR,
            elev,
            azim,
        )
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(gt_seq),
        interval=max(int(1000 / max(gif_fps, 1)), 1),
        blit=False,
        repeat=True,
    )

    save_path = output_dir / "motion.gif"
    anim.save(save_path, writer=PillowWriter(fps=gif_fps), dpi=dpi)
    plt.close(fig)
    return save_path


def resolve_output_dir(config):
    if config.vis_output_dir:
        return Path(config.vis_output_dir).expanduser().resolve()
    ckpt_path = Path(config.load_model_dir).expanduser().resolve()
    return ckpt_path.parent / "visualizations" / sanitize_name(ckpt_path.stem)


def main():
    parser = MotionFromGazeDualConfig()
    add_vis_args(parser)
    config = parser.parse_args()

    if config.load_model_dir is None:
        raise ValueError("--load_model_dir is required")

    seed_everything(config.seed, config.deterministic)
    output_root = resolve_output_dir(config)
    output_root.mkdir(parents=True, exist_ok=True)

    visualizer = DualPredictionVisualizer(config)

    print(f"checkpoint={Path(config.load_model_dir).resolve()}")
    print(f"dataset_size={len(visualizer.dataset)}")
    print(f"output_root={output_root}")

    for sample_idx in config.sample_indices:
        if sample_idx < 0 or sample_idx >= len(visualizer.dataset):
            raise IndexError(f"sample_idx {sample_idx} is out of range for dataset size {len(visualizer.dataset)}")

        sample = visualizer.predict(sample_idx)
        sample_dir = output_root / (
            f"sample_{sample_idx:03d}_{sanitize_name(sample['scene'])}_{sanitize_name(sample['seq'])}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        overview_path = save_overview(sample, sample_dir, config.elev, config.azim, config.dpi)
        gif_path = save_gif(sample, sample_dir, config.elev, config.azim, config.dpi, config.gif_fps)

        metrics = sample["metrics"]
        print(
            f"saved sample={sample_idx} seq={sample['seq']} scene={sample['scene']} "
            f"root_dest={metrics['root_dest']:.4f} des_mpjpe={metrics['des_mpjpe']:.4f}"
        )
        print(f"overview={overview_path}")
        print(f"gif={gif_path}")


if __name__ == "__main__":
    main()
