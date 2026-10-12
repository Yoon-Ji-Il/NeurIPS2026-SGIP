"""Single-source final no-gaze, no-object model for Teacher-final.

This is a stripped-down variant of the original ``sif3d_final.py`` where the
object-embedding pathway has been completely removed:

    * ``ObjectPointLatentEncoder`` and ``ObjectBiasScorer`` are gone.
    * ``TIABlock`` no longer takes / fuses an ``object_embedding`` tensor.
    * The SCA spatial prior no longer receives any object-derived bias
      (``obj_dir_factor``, ``object_prior_factor``, ``gt_prior_factor``).
    * The forward signature drops ``objects``, ``object_mask``,
      ``gt_object_point_prior``, ``gt_object_point_prior_mask``.
    * The ``return_features`` dict no longer contains ``object_emb``.

Only motion + scene point cloud are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from model.base_cross_model import PerceiveEncoder, SelfAttentionLayer
from model.pointnet_plus2 import PointNet2SemSegSSGShape
from utils.vis_utils import latent_to_joints


def human_centered_scene(xyz, ori, trans):
    bs, seq_len, _ = ori.shape
    sin_ori = torch.sin(ori[:, :, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, :, 1]).unsqueeze(-1)
    xyz = xyz[:, None, :, :].repeat(1, seq_len, 1, 1) - trans[:, :, None, :]

    matrix = torch.cat(
        [
            torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
            torch.cat(
                [torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)],
                dim=-1,
            ),
            torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1),
        ],
        dim=-1,
    )
    matrix = matrix.view(bs, seq_len, 3, 3)
    return torch.matmul(xyz, matrix)


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out


class IntentionAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats):
        bs, seq_len, _ = motion.shape
        n_points = scene_feats.shape[1]

        q = self.motion_q(motion) * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        q = q.view(bs, self.num_head, seq_len, -1)
        k = k.view(bs, n_points, self.num_head, -1).transpose(1, 2)
        v = v.view(bs, n_points, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v).view(bs, seq_len, -1)
        return self.out_emb(out)


class SemanticAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats, spatial_prior):
        bs, seq_len, _ = motion.shape
        n_points = scene_feats.shape[1]

        q = self.motion_q(motion) * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        q = q.view(bs, self.num_head, seq_len, -1)
        k = k.view(bs, n_points, self.num_head, -1).transpose(1, 2)
        v = v.view(bs, n_points, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = att + spatial_prior[:, None, :, :]
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_emb(out)


class SCABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.sa_attention = SemanticAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats, spatial_prior):
        motion_sca = self.motion_encoder(motion)
        sm_feature = self.sa_attention(motion_sca, scene_feats, spatial_prior)
        out = torch.cat([motion_sca, sm_feature], dim=-1)
        return self.mlp(out)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, xyz):
        out = F.relu(self.fc1(xyz))
        return self.fc2(out).squeeze(-1)


class TIABlock(nn.Module):
    """TIA block without object embedding.

    Original TIABlock concatenated [motion, scene_feature, object_embedding].
    Here we drop the object branch entirely, so the MLP input dim is
    ``2 * feat_dim`` instead of ``3 * feat_dim``.
    """

    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.ia_attention = IntentionAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats):
        motion_tia = self.motion_encoder(motion)
        sm_feature = self.ia_attention(motion_tia, scene_feats)
        out = torch.cat([motion_tia, sm_feature], dim=-1)
        return self.mlp(out)


class SIF3D(nn.Module):
    """SIF3D final, no-gaze + no-object variant.

    Inputs : motion (poses), joints, scene point cloud.
    Outputs: latent (ori|trans|pose) + reconstructed joint sequence.
    """

    def __init__(self, config, vposer, smplx_model):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.vposer = vposer
        self.smplx_model = smplx_model

        self.pointnet = PointNet2SemSegSSGShape({"feat_dim": config.scene_feats_dim})
        self.motion_emb = nn.Linear(69 + 38, config.motion_hidden_dim)
        self.motion_encoder = PerceiveEncoder(
            n_input_channels=config.motion_hidden_dim,
            n_latent=config.output_seq_len + config.input_seq_len,
            n_latent_channels=config.motion_latent_dim,
            n_self_att_heads=config.motion_n_heads,
            n_self_att_layers=config.motion_n_layers,
            dropout=config.dropout,
        )

        # Spatial prior: only distance- and direction-based saliency,
        # no object-derived bias terms.
        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))

        self.tia_blocks = nn.ModuleList(
            [TIABlock(config.motion_latent_dim, num_head=4) for _ in range(3)]
        )
        self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)

        self.sca_blocks = nn.ModuleList(
            [SCABlock(config.motion_latent_dim, num_head=4) for _ in range(3)]
        )
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

        self.joints_fc = nn.Linear(config.motion_latent_dim * 2, 69)
        self.motion_decoder = GCN(config, node_n=69)

    def forward(
        self,
        motions,
        joints,
        scene_xyz,
        return_features=False,
    ):
        bs, input_seq_len = motions.shape[:2]

        scene_feats, _ = self.pointnet(scene_xyz.repeat(1, 1, 2))
        scene_feats = scene_feats.transpose(1, 2)

        motions = torch.cat([motions, joints.view(bs, input_seq_len, -1)], dim=-1)
        motions = self.motion_emb(motions)
        motions = self.motion_encoder(motions)

        # ---- Trajectory branch (TIA) ----
        out = motions.clone()
        for blk in self.tia_blocks:
            out = blk(out, scene_feats)
        out_tia = out.clone()
        ori, trans = torch.split(self.trajectory_fc(out_tia), [3, 3], dim=-1)

        # ---- Spatial prior (no object terms) ----
        scene_xyz_normalized = human_centered_scene(scene_xyz, ori, trans)
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

        # ---- Pose branch (SCA) ----
        out = motions.clone()
        for blk in self.sca_blocks:
            out = blk(out, scene_feats, spatial_prior)
        out_sca = out.clone()
        pose = self.pose_fc(out_sca)

        latent = torch.cat([ori, trans, pose], dim=-1)
        recons_joints = latent_to_joints(latent, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()

        pred_joints_full = self.motion_decoder(recons_joints)[:, :, :23]
        if return_features:
            features = {
                "motion_enc": motions,
                "out_tia": out_tia,
                "out_sca": out_sca,
                "ori_trans": torch.cat([ori, trans], dim=-1),
                "pose_latent": pose,
            }
            return latent, pred_joints_full, features
        return latent, pred_joints_full
