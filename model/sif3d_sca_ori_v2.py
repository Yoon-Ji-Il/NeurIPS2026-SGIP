import torch
import torch.nn as nn

from model.sif3d_final import (
    SCABlock,
    SIF3DObjectPointNoGazeBase,
    TIABlock,
    human_centered_scene,
)
from utils.vis_utils import latent_to_joints


class SIF3DScaOriResidualV2(SIF3DObjectPointNoGazeBase):
    """SIF3D student variant with orientation correction predicted after SCA.

    The baseline SIF3D predicts orientation and translation from TIA, then pose
    from SCA. This variant keeps translation from TIA, but lets SCA predict a
    small residual orientation correction. The residual head is zero-initialized,
    so loading a baseline SIF3D checkpoint starts from the same predictions.
    """

    def __init__(self, config, vposer, smplx_model):
        super().__init__(
            config,
            vposer,
            smplx_model,
            TIABlock,
            use_object_prior_in_sca=False,
            sca_block_cls=SCABlock,
            use_gt_point_prior_in_sca=False,
        )
        self.tia_blocks = nn.ModuleList([TIABlock(config.motion_latent_dim, num_head=4) for _ in range(3)])
        self.sca_blocks = nn.ModuleList([SCABlock(config.motion_latent_dim, num_head=4) for _ in range(3)])
        self.ori_residual_fc = nn.Linear(config.motion_latent_dim, 3)
        nn.init.zeros_(self.ori_residual_fc.weight)
        nn.init.zeros_(self.ori_residual_fc.bias)

        self.ori_residual_scale = float(getattr(config, "sca_ori_residual_scale", 0.1))
        self.detach_sca_trajectory = bool(getattr(config, "sca_spatial_detach_trajectory", True))
        self.detach_ori_base = bool(getattr(config, "sca_ori_detach_base", True))
        self.preserve_decoder_root_translation = bool(
            getattr(config, "preserve_decoder_root_translation", False)
        )

    def forward(
        self,
        motions,
        joints,
        scene_xyz,
        objects,
        object_mask,
        gt_object_point_prior=None,
        gt_object_point_prior_mask=None,
    ):
        bs, input_seq_len = motions.shape[:2]

        if objects.ndim != 3 or objects.shape[-1] != 3:
            raise ValueError(f"Expected object point cloud with shape [B, P, 3], got {objects.shape}")

        object_points = objects
        object_mask = object_mask.float().view(bs, 1, 1)

        scene_feats, _ = self.pointnet(scene_xyz.repeat(1, 1, 2))
        scene_feats = scene_feats.transpose(1, 2)

        motions = torch.cat([motions, joints.view(bs, input_seq_len, -1)], dim=-1)
        motions = self.motion_emb(motions)
        motions = self.motion_encoder(motions)
        motion_seq_len = motions.shape[1]

        object_embedding = self.object_encoder(object_points)

        out = motions.clone()
        for blk in self.tia_blocks:
            out = blk(out, scene_feats, object_embedding)
        out_tia = out.clone()
        ori_tia, trans_tia = torch.split(self.trajectory_fc(out_tia), [3, 3], dim=-1)

        sca_ori = ori_tia.detach() if self.detach_sca_trajectory else ori_tia
        sca_trans = trans_tia.detach() if self.detach_sca_trajectory else trans_tia
        scene_xyz_normalized = human_centered_scene(scene_xyz, sca_ori, sca_trans)
        dist_from_human = torch.norm(scene_xyz_normalized, dim=-1)
        dist_salience = -((dist_from_human * 1.5) ** 3)
        dir_salience = scene_xyz_normalized[:, :, :, 0] / torch.sqrt(
            (scene_xyz_normalized[:, :, :, [0, 2]] ** 2).sum(dim=-1).clamp_min(1e-6)
        )
        spatial_prior = (
            dist_salience * self.dist_factor
            + dir_salience * self.dir_factor
            + self.spatial_att(scene_xyz_normalized)
        )

        obj_dir_salience = self._build_object_dir_prior(scene_xyz, object_points, sca_trans)
        spatial_prior = spatial_prior + self.obj_dir_factor * obj_dir_salience * object_mask

        if self.use_object_prior_in_sca:
            object_prior = self.object_prior_factor * self.object_prior_scorer(object_embedding, scene_feats)
            spatial_prior = spatial_prior + object_prior * object_mask

        if self.use_gt_point_prior_in_sca:
            if gt_object_point_prior is None or gt_object_point_prior_mask is None:
                raise ValueError("gt_object_point_prior and gt_object_point_prior_mask are required for this model variant")
            gt_object_point_prior = gt_object_point_prior.float()
            gt_object_point_prior_mask = gt_object_point_prior_mask.float().view(bs, 1, 1)
            gt_prior = gt_object_point_prior.unsqueeze(1).expand(-1, motion_seq_len, -1)
            spatial_prior = spatial_prior + self.gt_prior_factor * gt_prior * gt_object_point_prior_mask

        out = motions.clone()
        for blk in self.sca_blocks:
            out = blk(out, scene_feats, spatial_prior)
        out_sca = out.clone()

        pose = self.pose_fc(out_sca)
        ori_base = ori_tia.detach() if self.detach_ori_base else ori_tia
        ori_residual = self.ori_residual_scale * self.ori_residual_fc(out_sca)
        ori = ori_base + ori_residual
        trans = trans_tia

        latent = torch.cat([ori, trans, pose], dim=-1)
        recons_joints = latent_to_joints(latent, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()

        decoded_joints = self.motion_decoder(recons_joints)[:, :, :23]
        if self.preserve_decoder_root_translation:
            decoded_rel = decoded_joints - decoded_joints[:, :, [0]]
            pred_joints_full = decoded_rel + recons_joints[:, :, [0]]
        else:
            pred_joints_full = decoded_joints
        return latent, pred_joints_full
