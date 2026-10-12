#!/usr/bin/env python
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import smplx
import torch
import trimesh
from human_body_prior.tools.model_loader import load_vposer

from config.config_dual import MotionFromGazeDualConfig
from dataset import gimo_dataset_dual
from eval_dual import seed_everything
from model.sif3d_final import SIF3D


INPUT_BODY_COLOR = np.array([40, 40, 40, 255], dtype=np.uint8)
GT_BODY_COLOR = np.array([52, 152, 219, 255], dtype=np.uint8)
PRED_BODY_COLOR = np.array([231, 76, 60, 255], dtype=np.uint8)
INPUT_BODY_COLOR_LIGHT = np.array([150, 150, 150, 255], dtype=np.uint8)
GT_BODY_COLOR_LIGHT = np.array([172, 214, 250, 255], dtype=np.uint8)
PRED_BODY_COLOR_LIGHT = np.array([250, 180, 174, 255], dtype=np.uint8)
GT_BODY_COLOR_MID = np.array([110, 185, 236, 255], dtype=np.uint8)
PRED_BODY_COLOR_MID = np.array([243, 124, 113, 255], dtype=np.uint8)
SCENE_COLOR = np.array([190, 195, 199, 255], dtype=np.uint8)
BODY23_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (15, 22),
]


def add_export_args(parser):
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        default=[0],
        help="Test-set sample indices to export.",
    )
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="Export every sample in the test split.",
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default=None,
        help="Directory for saved mesh exports. Defaults near the checkpoint.",
    )
    parser.add_argument(
        "--export_full_sequence",
        action="store_true",
        help="Export all input+future frames. Default exports future frames only.",
    )
    parser.add_argument(
        "--scene_mesh_name",
        type=str,
        default="textured_output.obj",
        help="Preferred scene mesh file name inside <scene>/scene_obj/.",
    )
    parser.add_argument(
        "--scene_pointcloud_name",
        type=str,
        default="scene_downsampled.ply",
        help="Fallback scene asset if scene mesh is missing.",
    )
    parser.add_argument(
        "--export_scene_only",
        action="store_true",
        help="Also save a scene-only GLB per sample.",
    )
    parser.add_argument(
        "--diagnostic_only",
        action="store_true",
        help="Export only the diagnostic skeleton scene for each sample.",
    )
    parser.add_argument(
        "--export_pred_smplx_recon",
        action="store_true",
        help="Fit SMPL-X meshes to final predicted joints and export mesh visualizations.",
    )
    parser.add_argument(
        "--smplx_fit_iters",
        type=int,
        default=120,
        help="Optimization iterations per frame when fitting SMPL-X to final predicted joints.",
    )
    parser.add_argument(
        "--smplx_fit_lr",
        type=float,
        default=0.05,
        help="Learning rate for SMPL-X fitting to final predicted joints.",
    )
    return parser


def smplx_zero_args(batch_size, dtype, device, smplx_model):
    return {
        "betas": torch.zeros(batch_size, getattr(smplx_model, "num_betas", 10), dtype=dtype, device=device),
        "left_hand_pose": torch.zeros(batch_size, getattr(smplx_model, "num_pca_comps", 12), dtype=dtype, device=device),
        "right_hand_pose": torch.zeros(batch_size, getattr(smplx_model, "num_pca_comps", 12), dtype=dtype, device=device),
        "expression": torch.zeros(batch_size, getattr(smplx_model, "num_expression_coeffs", 10), dtype=dtype, device=device),
        "jaw_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
        "leye_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
        "reye_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
    }


def latent_to_vertices(pose_latent, vposer, smplx_model):
    bs, seq_len, _ = pose_latent.shape
    batch_size = bs * seq_len
    body_pose = vposer.decode(pose_latent[:, :, 6:], output_type="aa").view(-1, 63)
    dtype = pose_latent.dtype
    device = pose_latent.device
    smplx_model.batch_size = batch_size
    smplx_output = smplx_model(
        return_verts=True,
        body_pose=body_pose,
        global_orient=pose_latent[:, :, :3].view(-1, 3),
        transl=pose_latent[:, :, 3:6].view(-1, 3),
        pose_embedding=pose_latent[:, :, 6:].view(-1, 32),
        **smplx_zero_args(batch_size, dtype, device, smplx_model),
    )
    return smplx_output.vertices.reshape(bs, seq_len, -1, 3)


def smplx_forward_frames(global_orient, transl, pose_embedding, vposer, smplx_model):
    batch_size = global_orient.shape[0]
    dtype = global_orient.dtype
    device = global_orient.device
    body_pose = vposer.decode(pose_embedding, output_type="aa").view(batch_size, 63)
    smplx_model.batch_size = batch_size
    return smplx_model(
        return_verts=True,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        pose_embedding=pose_embedding,
        **smplx_zero_args(batch_size, dtype, device, smplx_model),
    )


def fit_smplx_frame_to_joints(target_joints, init_pose, vposer, smplx_model, num_iters, lr):
    init_orient = init_pose[:3].detach().clone()
    init_trans = init_pose[3:6].detach().clone()
    init_latent = init_pose[6:].detach().clone()

    global_orient = init_orient.clone().requires_grad_(True)
    transl = init_trans.clone().requires_grad_(True)
    pose_embedding = init_latent.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([global_orient, transl, pose_embedding], lr=lr)
    best_params = None
    best_joint_loss = None

    for _ in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        smplx_output = smplx_forward_frames(
            global_orient.unsqueeze(0),
            transl.unsqueeze(0),
            pose_embedding.unsqueeze(0),
            vposer,
            smplx_model,
        )
        pred_joints = smplx_output.joints[0, :23]
        joint_loss = (pred_joints - target_joints).pow(2).mean()
        reg_loss = (
            1e-4 * (pose_embedding - init_latent).pow(2).mean()
            + 1e-5 * (global_orient - init_orient).pow(2).mean()
            + 1e-5 * (transl - init_trans).pow(2).mean()
        )
        loss = joint_loss + reg_loss
        loss.backward()
        optimizer.step()

        joint_loss_value = float(joint_loss.detach().item())
        if best_joint_loss is None or joint_loss_value < best_joint_loss:
            best_joint_loss = joint_loss_value
            best_params = {
                "global_orient": global_orient.detach().clone(),
                "transl": transl.detach().clone(),
                "pose_embedding": pose_embedding.detach().clone(),
            }

    with torch.no_grad():
        smplx_output = smplx_forward_frames(
            best_params["global_orient"].unsqueeze(0),
            best_params["transl"].unsqueeze(0),
            best_params["pose_embedding"].unsqueeze(0),
            vposer,
            smplx_model,
        )
        pred_joints = smplx_output.joints[0, :23]
        fit_error = float(torch.norm(pred_joints - target_joints, dim=-1).mean().item())

    return (
        smplx_output.vertices[0].detach().cpu().numpy(),
        pred_joints.detach().cpu().numpy(),
        fit_error,
    )


def fit_smplx_sequence_to_joints(target_joints_seq, init_pose_seq, vposer, smplx_model, num_iters, lr):
    fitted_vertices = []
    fitted_joints = []
    fit_errors = []
    for frame_idx in range(target_joints_seq.shape[0]):
        vertices, joints, fit_error = fit_smplx_frame_to_joints(
            target_joints_seq[frame_idx],
            init_pose_seq[frame_idx],
            vposer,
            smplx_model,
            num_iters=num_iters,
            lr=lr,
        )
        fitted_vertices.append(vertices)
        fitted_joints.append(joints)
        fit_errors.append(fit_error)

    return (
        np.stack(fitted_vertices, axis=0),
        np.stack(fitted_joints, axis=0),
        fit_errors,
    )


def compute_metrics(pred_seq, gt_seq, input_seq_len):
    pred = torch.from_numpy(pred_seq)
    gt = torch.from_numpy(gt_seq)
    root_l2 = torch.norm(pred[:, 0] - gt[:, 0], dim=-1)
    pred_rel = pred - pred[:, [0]]
    gt_rel = gt - gt[:, [0]]
    mpjpe = torch.norm(pred_rel - gt_rel, dim=-1)
    input_last_root = gt[input_seq_len - 1, 0]
    gt_first_root = gt[input_seq_len, 0]
    pred_first_root = pred[input_seq_len, 0]
    return {
        "root_mean_future": float(root_l2[input_seq_len:].mean().item()),
        "root_dest": float(root_l2[-1].item()),
        "mpjpe_mean_future": float(mpjpe[input_seq_len:].mean().item()),
        "des_mpjpe": float(mpjpe[-1].mean().item()),
        "input_to_gt_first_root_gap": float(torch.norm(gt_first_root - input_last_root).item()),
        "input_to_pred_first_root_gap": float(torch.norm(pred_first_root - input_last_root).item()),
        "pred_first_to_gt_first_root_error": float(torch.norm(pred_first_root - gt_first_root).item()),
    }


def paint_mesh(mesh, rgba):
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile(rgba[None, :], (len(mesh.vertices), 1))
    return mesh


def color_ramp(start_rgba, end_rgba, n_steps):
    if n_steps <= 0:
        return []
    if n_steps == 1:
        return [end_rgba.astype(np.uint8)]
    start = start_rgba.astype(np.float32)
    end = end_rgba.astype(np.float32)
    ramp = []
    for idx in range(n_steps):
        alpha = idx / float(n_steps - 1)
        rgba = (1.0 - alpha) * start + alpha * end
        ramp.append(np.round(rgba).astype(np.uint8))
    return ramp


def load_scene_asset(scene_root, scene_mesh_name, scene_pointcloud_name, scale, transform_norm):
    mesh_candidates = []
    if scene_mesh_name:
        mesh_candidates.append(scene_root / "scene_obj" / scene_mesh_name)
    mesh_candidates.append(scene_root / "scene_obj" / "textured_output.obj")
    mesh_candidates.append(scene_root / "scene_obj" / "scene_obj.obj")

    checked_paths = []
    for mesh_path in mesh_candidates:
        if mesh_path in checked_paths:
            continue
        checked_paths.append(mesh_path)
        if mesh_path.exists():
            loaded = trimesh.load(mesh_path, process=False)
            loaded = transform_scene_asset(loaded, scale, transform_norm)
            if isinstance(loaded, trimesh.Scene):
                return loaded, mesh_path
            if isinstance(loaded, trimesh.Trimesh):
                if getattr(loaded.visual, "kind", None) in {"texture", "vertex"}:
                    return loaded, mesh_path
                return paint_mesh(loaded, SCENE_COLOR), mesh_path

    point_path = scene_root / "scene_obj" / scene_pointcloud_name
    if point_path.exists():
        loaded = trimesh.load(point_path, process=False)
        loaded = transform_scene_asset(loaded, scale, transform_norm)
        if isinstance(loaded, trimesh.PointCloud):
            points = loaded.vertices
        elif isinstance(loaded, trimesh.Trimesh):
            points = loaded.vertices
        else:
            raise ValueError(f"Unsupported fallback scene asset: {point_path}")
        pointcloud = trimesh.points.PointCloud(points, colors=np.tile(SCENE_COLOR[None, :], (len(points), 1)))
        return pointcloud, point_path

    raise FileNotFoundError(
        f"Could not find scene asset under {scene_root / 'scene_obj'} "
        f"(looked for {[str(p.name) for p in checked_paths]} and {scene_pointcloud_name})"
    )


def transform_scene_asset(asset, scale, transform_norm):
    asset = asset.copy()
    asset.apply_scale(1.0 / float(scale))
    asset.apply_transform(transform_norm)
    return asset


def resolve_output_dir(config):
    if config.vis_output_dir:
        return Path(config.vis_output_dir).expanduser().resolve()
    ckpt_path = Path(config.load_model_dir).expanduser().resolve()
    return ckpt_path.parent / "mesh_visualizations" / ckpt_path.stem


def sha1_file(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DualMeshExporter:
    def __init__(self, config):
        self.config = config
        self.dataset = gimo_dataset_dual.EgoEvalDataset(config, train=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gimo_root = Path(config.dataroot).expanduser().resolve()
        self.checkpoint_path = Path(config.load_model_dir).expanduser().resolve()
        self.checkpoint_sha1 = sha1_file(self.checkpoint_path)
        self.object_points_name = config.object_points_name
        self.object_mask_name = config.object_mask_name or config.object_points_name.replace(".pth", "_mask.pth")

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
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = {k: v for k, v in state_dict.items() if ("smplx" not in k and "vposer" not in k)}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.faces = np.asarray(self.body_model.faces, dtype=np.int64)

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

        del objects, object_mask, gt_object_point_prior, gt_object_point_prior_mask, scene_points

        poses_input_t = poses_input.unsqueeze(0).to(self.device)
        poses_label_t = poses_label.unsqueeze(0).to(self.device)
        joints_input_t = joints_input.unsqueeze(0).to(self.device)

        objects_t = raw[0].unsqueeze(0).to(self.device)
        object_mask_t = raw[1].unsqueeze(0).to(self.device)
        gt_prior_t = raw[2].unsqueeze(0).to(self.device)
        gt_prior_mask_t = raw[3].unsqueeze(0).to(self.device)
        scene_points_t = raw[8].unsqueeze(0).to(self.device).contiguous()

        with torch.no_grad():
            pred_latent, pred_joints = self.model(
                poses_input_t,
                joints_input_t[:, :, :23],
                scene_points_t,
                objects_t,
                object_mask_t,
                gt_prior_t,
                gt_prior_mask_t,
            )

            gt_pose_seq = torch.cat([poses_input_t, poses_label_t], dim=1)
            pred_pose_seq = torch.cat([poses_input_t, pred_latent[:, self.config.input_seq_len:]], dim=1)

            gt_vertices = latent_to_vertices(gt_pose_seq, self.vposer, self.body_model)[0].cpu().numpy()
            pred_vertices = latent_to_vertices(pred_pose_seq, self.vposer, self.body_model)[0].cpu().numpy()

        gt_joints = torch.cat([joints_input[:, :23], joints_label[:, :23]], dim=0).cpu().numpy()
        pred_joints = pred_joints.cpu().numpy()[0]
        metrics = compute_metrics(pred_joints, gt_joints, joints_input.shape[0])

        scene_root = self.gimo_root / scene
        transform_info_path = scene_root / seq / self.dataset.trans_path_list[sample_idx]
        with open(transform_info_path, "r", encoding="utf-8") as f:
            transform_info = json.load(f)
        scale = float(transform_info["scale"])
        transform_norm = np.loadtxt(scene_root / "scene_obj" / "transform_norm.txt").reshape(4, 4)
        transform_norm = transform_norm.astype(np.float64)
        transform_norm[:3, 3] /= scale
        scene_asset, scene_asset_path = load_scene_asset(
            scene_root,
            self.config.scene_mesh_name,
            self.config.scene_pointcloud_name,
            scale,
            transform_norm,
        )

        return {
            "sample_idx": sample_idx,
            "dataset_csv_row": self.dataset.csv_row_list[sample_idx],
            "scene": scene,
            "seq": seq,
            "input_seq_len": joints_input.shape[0],
            "input_vertices": gt_vertices[: joints_input.shape[0]].copy(),
            "input_joints": gt_joints[: joints_input.shape[0]].copy(),
            "gt_vertices": gt_vertices,
            "gt_joints": gt_joints,
            "pred_vertices": pred_vertices,
            "pred_joints": pred_joints,
            "gt_pose_seq": gt_pose_seq[0].detach().cpu().numpy(),
            "pred_pose_seq_init": pred_pose_seq[0].detach().cpu().numpy(),
            "scene_asset": scene_asset,
            "scene_asset_path": scene_asset_path,
            "scale": scale,
            "transform_info_path": transform_info_path,
            "transform_norm_path": scene_root / "scene_obj" / "transform_norm.txt",
            "metrics": metrics,
            "start_end": self.dataset.start_end_list[sample_idx],
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha1": self.checkpoint_sha1,
            "object_points_name": self.object_points_name,
            "object_mask_name": self.object_mask_name,
        }


def ensure_pred_smplx_fitted(sample, exporter):
    if "pred_fitted_vertices" in sample:
        return sample

    input_len = sample["input_seq_len"]
    target_joints = torch.from_numpy(sample["pred_joints"][input_len:]).float().to(exporter.device)
    init_pose_seq = torch.from_numpy(sample["pred_pose_seq_init"][input_len:]).float().to(exporter.device)

    fitted_vertices, fitted_joints, fit_errors = fit_smplx_sequence_to_joints(
        target_joints,
        init_pose_seq,
        exporter.vposer,
        exporter.body_model,
        num_iters=exporter.config.smplx_fit_iters,
        lr=exporter.config.smplx_fit_lr,
    )
    sample["pred_fitted_vertices"] = fitted_vertices
    sample["pred_fitted_joints"] = fitted_joints
    sample["pred_fitted_joint_errors"] = fit_errors
    sample["pred_fitted_joint_error_mean"] = float(np.mean(fit_errors))
    sample["pred_fitted_joint_error_max"] = float(np.max(fit_errors))
    sample["smplx_fit_iters"] = int(exporter.config.smplx_fit_iters)
    sample["smplx_fit_lr"] = float(exporter.config.smplx_fit_lr)
    return sample


def export_frame(scene_asset, body_vertices, faces, rgba, export_path):
    body_mesh = trimesh.Trimesh(vertices=body_vertices, faces=faces, process=False)
    body_mesh = paint_mesh(body_mesh, rgba)

    scene = make_export_scene(scene_asset)
    scene.add_geometry(body_mesh, node_name="body")
    scene.export(export_path)


def make_export_scene(scene_asset):
    if isinstance(scene_asset, trimesh.Scene):
        return scene_asset.copy()
    return trimesh.Scene([scene_asset.copy()])


def add_body_sequence(scene, vertices_seq, faces, color_list, prefix):
    for frame_idx, (vertices, rgba) in enumerate(zip(vertices_seq, color_list)):
        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        body_mesh = paint_mesh(body_mesh, rgba)
        scene.add_geometry(body_mesh, node_name=f"{prefix}_{frame_idx:03d}")


def add_joint_sequence(scene, joints_seq, color_list, prefix, sphere_radius=0.028, bone_radius=0.012):
    for frame_idx, (joints, rgba) in enumerate(zip(joints_seq, color_list)):
        color = np.asarray(rgba, dtype=np.uint8)
        for joint_idx, center in enumerate(joints):
            sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
            sphere.visual.vertex_colors = np.tile(color[None, :], (len(sphere.vertices), 1))
            sphere.apply_translation(center)
            scene.add_geometry(sphere, node_name=f"{prefix}_{frame_idx:03d}_joint_{joint_idx:02d}")
        for edge_idx, (src, dst) in enumerate(BODY23_EDGES):
            start = joints[src]
            end = joints[dst]
            if np.linalg.norm(end - start) < 1e-8:
                continue
            bone = trimesh.creation.cylinder(radius=bone_radius, segment=np.stack([start, end], axis=0))
            bone.visual.vertex_colors = np.tile(color[None, :], (len(bone.vertices), 1))
            scene.add_geometry(bone, node_name=f"{prefix}_{frame_idx:03d}_bone_{edge_idx:02d}")


def export_combined_scene(sample, faces, export_path, export_full_sequence=False):
    input_len = sample["input_seq_len"]
    gt_vertices = sample["gt_vertices"]
    pred_vertices = sample["pred_vertices"]

    if export_full_sequence:
        gt_seq = gt_vertices
        pred_seq = pred_vertices
    else:
        gt_seq = gt_vertices[input_len:]
        pred_seq = pred_vertices[input_len:]

    scene = make_export_scene(sample["scene_asset"])

    input_colors = color_ramp(INPUT_BODY_COLOR_LIGHT, INPUT_BODY_COLOR, len(sample["input_vertices"]))
    gt_colors = color_ramp(GT_BODY_COLOR_LIGHT, GT_BODY_COLOR, len(gt_seq))
    pred_colors = color_ramp(PRED_BODY_COLOR_LIGHT, PRED_BODY_COLOR, len(pred_seq))

    add_body_sequence(scene, sample["input_vertices"], faces, input_colors, "input")
    add_body_sequence(scene, gt_seq, faces, gt_colors, "gt")
    add_body_sequence(scene, pred_seq, faces, pred_colors, "pred")
    scene.export(export_path)


def export_pred_fitted_smplx_scene(sample, faces, export_path, export_full_sequence=False):
    scene = make_export_scene(sample["scene_asset"])
    input_colors = color_ramp(INPUT_BODY_COLOR_LIGHT, INPUT_BODY_COLOR, len(sample["input_vertices"]))
    input_len = sample["input_seq_len"]
    if export_full_sequence:
        gt_seq = sample["gt_vertices"]
        pred_seq = np.concatenate([sample["input_vertices"], sample["pred_fitted_vertices"]], axis=0)
        gt_colors = color_ramp(GT_BODY_COLOR_LIGHT, GT_BODY_COLOR, len(gt_seq))
        pred_input_colors = color_ramp(INPUT_BODY_COLOR_LIGHT, INPUT_BODY_COLOR, input_len)
        pred_future_colors = color_ramp(PRED_BODY_COLOR_LIGHT, PRED_BODY_COLOR, len(sample["pred_fitted_vertices"]))
        pred_colors = pred_input_colors + pred_future_colors
    else:
        gt_seq = sample["gt_vertices"][input_len:]
        pred_seq = sample["pred_fitted_vertices"]
        gt_colors = color_ramp(GT_BODY_COLOR_LIGHT, GT_BODY_COLOR, len(gt_seq))
        pred_colors = color_ramp(PRED_BODY_COLOR_LIGHT, PRED_BODY_COLOR, len(pred_seq))

    add_body_sequence(scene, sample["input_vertices"], faces, input_colors, "input")
    add_body_sequence(scene, gt_seq, faces, gt_colors, "gt")
    add_body_sequence(scene, pred_seq, faces, pred_colors, "pred_smplx_fit")
    scene.export(export_path)


def export_pred_fitted_smplx_transition_scene(sample, faces, export_path):
    scene = make_export_scene(sample["scene_asset"])
    input_last = sample["input_vertices"][-1:]
    gt_first = sample["gt_vertices"][sample["input_seq_len"]: sample["input_seq_len"] + 1]
    gt_final = sample["gt_vertices"][-1:]
    pred_first = sample["pred_fitted_vertices"][:1]
    pred_final = sample["pred_fitted_vertices"][-1:]

    add_body_sequence(scene, input_last, faces, [INPUT_BODY_COLOR], "input_last")
    add_body_sequence(scene, gt_first, faces, [GT_BODY_COLOR_LIGHT], "gt_first")
    add_body_sequence(scene, pred_first, faces, [PRED_BODY_COLOR_LIGHT], "pred_first_smplx_fit")
    add_body_sequence(scene, gt_final, faces, [GT_BODY_COLOR], "gt_final")
    add_body_sequence(scene, pred_final, faces, [PRED_BODY_COLOR], "pred_final_smplx_fit")
    scene.export(export_path)


def export_full_skeleton_scene(sample, export_path, export_full_sequence=False):
    input_len = sample["input_seq_len"]
    gt_seq = sample["gt_joints"] if export_full_sequence else sample["gt_joints"][input_len:]
    pred_seq = sample["pred_joints"] if export_full_sequence else sample["pred_joints"][input_len:]

    scene = make_export_scene(sample["scene_asset"])
    input_colors = color_ramp(INPUT_BODY_COLOR_LIGHT, INPUT_BODY_COLOR, len(sample["input_joints"]))
    gt_colors = color_ramp(GT_BODY_COLOR_LIGHT, GT_BODY_COLOR, len(gt_seq))
    pred_colors = color_ramp(PRED_BODY_COLOR_LIGHT, PRED_BODY_COLOR, len(pred_seq))

    add_joint_sequence(scene, sample["input_joints"], input_colors, "input")
    add_joint_sequence(scene, gt_seq, gt_colors, "gt")
    add_joint_sequence(scene, pred_seq, pred_colors, "pred")
    scene.export(export_path)


def export_final_skeleton_scene(sample, export_path):
    scene = make_export_scene(sample["scene_asset"])
    add_joint_sequence(scene, sample["input_joints"][-1:], [INPUT_BODY_COLOR], "input_last")
    add_joint_sequence(scene, sample["gt_joints"][-1:], [GT_BODY_COLOR], "gt_final")
    add_joint_sequence(scene, sample["pred_joints"][-1:], [PRED_BODY_COLOR], "pred_final")
    scene.export(export_path)


def export_transition_final_skeleton_scene(sample, export_path):
    scene = make_export_scene(sample["scene_asset"])
    input_len = sample["input_seq_len"]
    add_joint_sequence(scene, sample["input_joints"][input_len - 1: input_len], [INPUT_BODY_COLOR], "input_last")
    add_joint_sequence(scene, sample["gt_joints"][input_len: input_len + 1], [GT_BODY_COLOR_LIGHT], "gt_first")
    add_joint_sequence(scene, sample["pred_joints"][input_len: input_len + 1], [PRED_BODY_COLOR_LIGHT], "pred_first")
    add_joint_sequence(scene, sample["gt_joints"][-1:], [GT_BODY_COLOR], "gt_final")
    add_joint_sequence(scene, sample["pred_joints"][-1:], [PRED_BODY_COLOR], "pred_final")
    scene.export(export_path)


def write_summary(sample, sample_dir):
    metrics = sample["metrics"]
    start_frame, end_frame = sample["start_end"]
    extra_lines = []
    if "pred_fitted_joint_error_mean" in sample:
        extra_lines.extend(
            [
                f"pred_smplx_fit_joint_error_mean: {sample['pred_fitted_joint_error_mean']:.6f}",
                f"pred_smplx_fit_joint_error_max: {sample['pred_fitted_joint_error_max']:.6f}",
                f"pred_smplx_fit_iters: {sample['smplx_fit_iters']}",
                f"pred_smplx_fit_lr: {sample['smplx_fit_lr']}",
            ]
        )
    text = "\n".join(
        [
            f"sample_idx: {sample['sample_idx']}",
            f"dataset_csv_row: {sample['dataset_csv_row']}",
            f"scene: {sample['scene']}",
            f"sequence: {sample['seq']}",
            f"dataset_start_frame: {start_frame}",
            f"dataset_end_frame: {end_frame}",
            f"input_seq_len: {sample['input_seq_len']}",
            f"checkpoint: {sample['checkpoint_path']}",
            f"checkpoint_sha1: {sample['checkpoint_sha1']}",
            f"object_points_name: {sample['object_points_name']}",
            f"object_mask_name: {sample['object_mask_name']}",
            f"scene_asset: {sample['scene_asset_path']}",
            f"transform_info: {sample['transform_info_path']}",
            f"transform_norm: {sample['transform_norm_path']}",
            f"scale: {sample['scale']}",
            f"root_mean_future: {metrics['root_mean_future']:.6f}",
            f"root_dest: {metrics['root_dest']:.6f}",
            f"mpjpe_mean_future: {metrics['mpjpe_mean_future']:.6f}",
            f"des_mpjpe: {metrics['des_mpjpe']:.6f}",
            f"input_to_gt_first_root_gap: {metrics['input_to_gt_first_root_gap']:.6f}",
            f"input_to_pred_first_root_gap: {metrics['input_to_pred_first_root_gap']:.6f}",
            f"pred_first_to_gt_first_root_error: {metrics['pred_first_to_gt_first_root_error']:.6f}",
            *extra_lines,
            f"input_frames: 0..{sample['input_seq_len'] - 1}",
            f"future_frames: {sample['input_seq_len']}..{len(sample['gt_vertices']) - 1}",
            "prediction_source: pred_joints is the final model output; pred_vertices is only a latent-to-mesh reconstruction",
            "combined_glb_contents: scene + input(observed) + gt(future/full) + pred(future/full)",
            "pred_smplx_recon_glb_contents: scene + input(observed mesh) + gt(future mesh) + pred(future fitted SMPL-X mesh)",
            "pred_smplx_recon_full16_glb_contents: scene + input(observed mesh) + gt(full 16-step mesh) + pred(full 16-step mesh; observed prefix copied from input, future suffix fitted from final pred_joints)",
            "pred_smplx_recon_transition_glb_contents: scene + input_last + gt_first + pred_first(fitted) + gt_final + pred_final(fitted)",
            "full_skeleton_glb_contents: scene + input(observed skeletons) + gt(future/full skeletons) + pred(future/full skeletons)",
            "final_skeleton_glb_contents: scene + input_last + gt_final + pred_final",
            "transition_final_skeleton_glb_contents: scene + input_last + gt_first + pred_first + gt_final + pred_final",
            "body_colors: input=gray/black, gt=lightblue->blue, pred=lightred->red",
            "scene_transform_applied: scene *= 1/scale, then transform_norm",
            "scene_preference: textured_output.obj -> scene_obj.obj -> scene_downsampled.ply",
            "default_export: input observed + gt future + pred future unless --export_full_sequence is set",
        ]
    )
    summary_path = sample_dir / "summary.txt"
    summary_path.write_text(text + "\n", encoding="utf-8")
    return summary_path


def write_sample_metadata(sample, sample_dir):
    start_frame, end_frame = sample["start_end"]
    metadata = {
        "sample_idx": int(sample["sample_idx"]),
        "dataset_csv_row": int(sample["dataset_csv_row"]),
        "scene": sample["scene"],
        "sequence": sample["seq"],
        "dataset_start_frame": int(start_frame),
        "dataset_end_frame": int(end_frame),
        "input_seq_len": int(sample["input_seq_len"]),
        "future_seq_len": int(len(sample["gt_joints"]) - sample["input_seq_len"]),
        "checkpoint": str(sample["checkpoint_path"]),
        "checkpoint_sha1": sample["checkpoint_sha1"],
        "object_points_name": sample["object_points_name"],
        "object_mask_name": sample["object_mask_name"],
        "scene_asset": str(sample["scene_asset_path"]),
        "transform_info": str(sample["transform_info_path"]),
        "transform_norm": str(sample["transform_norm_path"]),
        "scale": float(sample["scale"]),
        "metrics": sample["metrics"],
    }
    if "pred_fitted_joint_error_mean" in sample:
        metadata["pred_smplx_fit"] = {
            "joint_error_mean": float(sample["pred_fitted_joint_error_mean"]),
            "joint_error_max": float(sample["pred_fitted_joint_error_max"]),
            "joint_errors_per_frame": [float(x) for x in sample["pred_fitted_joint_errors"]],
            "iters": int(sample["smplx_fit_iters"]),
            "lr": float(sample["smplx_fit_lr"]),
        }
    metadata_path = sample_dir / "sample_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def write_run_manifest(exporter, output_root, sample_indices):
    manifest = {
        "checkpoint": str(exporter.checkpoint_path),
        "checkpoint_sha1": exporter.checkpoint_sha1,
        "dataroot": str(exporter.gimo_root),
        "split": "test",
        "dataset_size": len(exporter.dataset),
        "object_points_name": exporter.object_points_name,
        "object_mask_name": exporter.object_mask_name,
        "scene_mesh_name": exporter.config.scene_mesh_name,
        "scene_pointcloud_name": exporter.config.scene_pointcloud_name,
        "sample_indices": list(sample_indices),
        "input_seq_len": int(exporter.config.input_seq_len),
        "output_seq_len": int(exporter.config.output_seq_len),
        "sample_points": int(exporter.config.sample_points),
        "export_pred_smplx_recon": bool(exporter.config.export_pred_smplx_recon),
        "smplx_fit_iters": int(exporter.config.smplx_fit_iters),
        "smplx_fit_lr": float(exporter.config.smplx_fit_lr),
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_dataset_index_map(exporter, output_root, sample_indices):
    entries = []
    for sample_idx in sample_indices:
        start_frame, end_frame = exporter.dataset.start_end_list[sample_idx]
        entries.append(
            {
                "sample_idx": int(sample_idx),
                "dataset_csv_row": int(exporter.dataset.csv_row_list[sample_idx]),
                "scene": exporter.dataset.scenes_path_list[sample_idx],
                "sequence": exporter.dataset.sequences_path_list[sample_idx],
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "transformation": exporter.dataset.trans_path_list[sample_idx],
            }
        )
    index_map_path = output_root / "dataset_index_map.json"
    index_map_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return index_map_path


def main():
    parser = MotionFromGazeDualConfig()
    add_export_args(parser)
    config = parser.parse_args()

    if config.load_model_dir is None:
        raise ValueError("--load_model_dir is required")

    seed_everything(config.seed, config.deterministic)
    output_root = resolve_output_dir(config)
    output_root.mkdir(parents=True, exist_ok=True)

    exporter = DualMeshExporter(config)
    print(f"checkpoint={Path(config.load_model_dir).resolve()}")
    print(f"checkpoint_sha1={exporter.checkpoint_sha1}")
    print(f"object_points_name={exporter.object_points_name}")
    print(f"dataset_size={len(exporter.dataset)}")
    print(f"output_root={output_root}")

    sample_indices = list(range(len(exporter.dataset))) if config.all_samples else list(config.sample_indices)
    total_samples = len(sample_indices)
    run_manifest_path = write_run_manifest(exporter, output_root, sample_indices)
    dataset_index_map_path = write_dataset_index_map(exporter, output_root, sample_indices)
    print(f"run_manifest={run_manifest_path}")
    print(f"dataset_index_map={dataset_index_map_path}")

    for order_idx, sample_idx in enumerate(sample_indices, start=1):
        if sample_idx < 0 or sample_idx >= len(exporter.dataset):
            raise IndexError(f"sample_idx {sample_idx} is out of range for dataset size {len(exporter.dataset)}")

        sample = exporter.predict(sample_idx)
        sample_dir = output_root / f"sample_{sample_idx:03d}_{sample['scene']}_{sample['seq']}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        pred_smplx_recon_path = None
        pred_smplx_recon_transition_path = None
        if config.export_pred_smplx_recon:
            ensure_pred_smplx_fitted(sample, exporter)

        summary_path = write_summary(sample, sample_dir)
        metadata_path = write_sample_metadata(sample, sample_dir)
        scene_copy_path = None
        if config.export_scene_only:
            scene_copy_path = sample_dir / "scene_only.glb"
            make_export_scene(sample["scene_asset"]).export(scene_copy_path)

        if config.export_pred_smplx_recon:
            pred_smplx_recon_path = sample_dir / "input_gt_pred_smplx_recon_with_scene.glb"
            export_pred_fitted_smplx_scene(sample, exporter.faces, pred_smplx_recon_path)
            pred_smplx_recon_full16_path = sample_dir / "input_gt_pred_full16_smplx_recon_with_scene.glb"
            export_pred_fitted_smplx_scene(sample, exporter.faces, pred_smplx_recon_full16_path, export_full_sequence=True)
            pred_smplx_recon_transition_path = sample_dir / "input_gt_pred_transition_smplx_recon_with_scene.glb"
            export_pred_fitted_smplx_transition_scene(sample, exporter.faces, pred_smplx_recon_transition_path)
        else:
            pred_smplx_recon_full16_path = None
        full_skeleton_path = sample_dir / "input_gt_pred_full_skeleton_with_scene.glb"
        export_full_skeleton_scene(sample, full_skeleton_path, export_full_sequence=config.export_full_sequence)
        final_skeleton_path = sample_dir / "input_gt_pred_final_skeleton_with_scene.glb"
        export_final_skeleton_scene(sample, final_skeleton_path)
        transition_final_path = sample_dir / "input_gt_pred_transition_final_skeleton_with_scene.glb"
        export_transition_final_skeleton_scene(sample, transition_final_path)
        combined_path = None
        if not config.diagnostic_only:
            combined_path = sample_dir / "input_gt_pred_with_scene.glb"
            export_combined_scene(
                sample,
                exporter.faces,
                combined_path,
                export_full_sequence=config.export_full_sequence,
            )

        metrics = sample["metrics"]
        print(
            f"saved [{order_idx}/{total_samples}] sample={sample_idx} seq={sample['seq']} scene={sample['scene']} "
            f"root_dest={metrics['root_dest']:.4f} des_mpjpe={metrics['des_mpjpe']:.4f}"
        )
        print(f"summary={summary_path}")
        print(f"sample_metadata={metadata_path}")
        if scene_copy_path is not None:
            print(f"scene_only={scene_copy_path}")
        if pred_smplx_recon_path is not None:
            print(f"pred_smplx_recon={pred_smplx_recon_path}")
        if pred_smplx_recon_full16_path is not None:
            print(f"pred_smplx_recon_full16={pred_smplx_recon_full16_path}")
        if pred_smplx_recon_transition_path is not None:
            print(f"pred_smplx_recon_transition={pred_smplx_recon_transition_path}")
        print(f"full_skeleton={full_skeleton_path}")
        print(f"final_skeleton={final_skeleton_path}")
        print(f"transition_final_skeleton={transition_final_path}")
        if combined_path is not None:
            print(f"combined={combined_path}")


if __name__ == "__main__":
    main()
