"""SIF3D variant for native HIK: 25 observed + 50 future frames.

This module is separate from ``sif3d_final.py`` on purpose.  The original
6+10 GIMO model and its checkpoint shapes remain untouched.
"""

import torch
import torch.nn.functional as F

from .gcn import GCN
from .sif3d_final import SIF3D as SIF3DGIMO
from .sif3d_final import human_centered_scene
from utils.vis_utils import latent_to_joints


def _resize_tensor_along(
    tensor: torch.Tensor,
    dimension: int,
    size: int,
) -> torch.Tensor:
    """Linearly resize one temporal parameter dimension."""

    moved = tensor.movedim(dimension, -1)
    old_size = moved.shape[-1]
    flattened = moved.reshape(-1, 1, old_size)
    resized = F.interpolate(
        flattened,
        size=int(size),
        mode="linear",
        align_corners=True,
    )
    resized = resized.reshape(*moved.shape[:-1], int(size))
    return resized.movedim(-1, dimension)


def load_gimo_checkpoint_for_hik(
    model,
    checkpoint_path,
    map_location="cpu",
):
    """Warm-start H25F50 from the trained GIMO 6+10 checkpoint.

    Shape-compatible parameters are copied exactly.  The five parameters
    whose dimensions directly encode the 16-frame GIMO timeline are linearly
    resized to 75 frames.  This is an initialization for HIK fine-tuning, not
    a replacement for HIK training.
    """

    source = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if isinstance(source, dict) and "state_dict" in source:
        source = source["state_dict"]
    target = model.state_dict()
    temporal_dimensions = {
        "motion_encoder.latent": 0,
        "object_encoder.latent": 0,
        "motion_decoder.gc1.weight": 0,
        "motion_decoder.gc7.weight": 1,
        "motion_decoder.gc7.bias": 0,
    }

    loadable = {}
    exact = []
    interpolated = []
    skipped = []
    for key, value in source.items():
        if key.startswith("smplx_model."):
            skipped.append(key)
            continue
        if key not in target:
            skipped.append(key)
            continue
        if value.shape == target[key].shape:
            loadable[key] = value
            exact.append(key)
            continue
        if key in temporal_dimensions:
            dimension = temporal_dimensions[key]
            resized = _resize_tensor_along(
                value.float(),
                dimension,
                target[key].shape[dimension],
            ).to(dtype=target[key].dtype)
            if resized.shape == target[key].shape:
                loadable[key] = resized
                interpolated.append(
                    {
                        "key": key,
                        "source_shape": list(value.shape),
                        "target_shape": list(resized.shape),
                    }
                )
                continue
        skipped.append(key)

    incompatible = model.load_state_dict(loadable, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "exact_parameter_tensors": len(exact),
        "interpolated_parameter_tensors": interpolated,
        "skipped_source_tensors": len(skipped),
        "missing_target_tensors": len(incompatible.missing_keys),
        "unexpected_target_tensors": len(incompatible.unexpected_keys),
        "requires_hik_finetuning": True,
    }


class SIF3D(SIF3DGIMO):
    INPUT_FRAMES = 25
    FUTURE_FRAMES = 50

    def __init__(self, config, vposer, smplx_model):
        if (
            config.input_seq_len != self.INPUT_FRAMES
            or config.output_seq_len != self.FUTURE_FRAMES
        ):
            raise ValueError(
                "SIF3D HIK 1s2s requires input_seq_len=25 and "
                f"output_seq_len=50, got {config.input_seq_len} and "
                f"{config.output_seq_len}"
            )
        super().__init__(config, vposer, smplx_model)

        # The original decoder has 16 DCT coefficients (6+10).  This isolated
        # native-HIK variant needs 75 coefficients (25+50).
        total_frames = config.input_seq_len + config.output_seq_len
        self.motion_decoder = GCN(
            config,
            input_feature=total_frames,
            node_n=69,
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
        return_features=False,
    ):
        bs, input_seq_len = motions.shape[:2]
        if input_seq_len != self.INPUT_FRAMES:
            raise ValueError(
                f"Expected {self.INPUT_FRAMES} observed frames, got "
                f"{input_seq_len}"
            )
        if objects.ndim != 3 or objects.shape[-1] != 3:
            raise ValueError(
                "Expected object point cloud with shape [B, P, 3], "
                f"got {objects.shape}"
            )

        object_points = objects
        object_mask = object_mask.float().view(bs, 1, 1)

        scene_feats, _ = self.pointnet(scene_xyz.repeat(1, 1, 2))
        scene_feats = scene_feats.transpose(1, 2)

        motions = torch.cat(
            [motions, joints.view(bs, input_seq_len, -1)],
            dim=-1,
        )
        motions = self.motion_emb(motions)
        motions = self.motion_encoder(motions)
        motion_seq_len = motions.shape[1]

        if self.disable_object_conditioning:
            object_embedding = torch.zeros(
                bs,
                motion_seq_len,
                self.config.motion_latent_dim,
                dtype=motions.dtype,
                device=motions.device,
            )
        else:
            object_embedding = self.object_encoder(object_points)
            object_embedding = object_embedding * object_mask

        out = motions.clone()
        for block in self.tia_blocks:
            out = block(out, scene_feats, object_embedding)
        out_tia = out.clone()
        ori, trans = torch.split(
            self.trajectory_fc(out_tia),
            [3, 3],
            dim=-1,
        )

        scene_xyz_normalized = human_centered_scene(scene_xyz, ori, trans)
        dist_from_human = torch.norm(scene_xyz_normalized, dim=-1)
        dist_salience = -((dist_from_human * 1.5) ** 3)
        dir_salience = scene_xyz_normalized[:, :, :, 0] / torch.sqrt(
            (
                scene_xyz_normalized[:, :, :, [0, 2]] ** 2
            ).sum(dim=-1).clamp_min(1e-6)
        )
        spatial_prior = (
            dist_salience * self.dist_factor
            + dir_salience * self.dir_factor
            + self.spatial_att(scene_xyz_normalized)
        )

        if not self.disable_object_conditioning:
            obj_dir_salience = self._build_object_dir_prior(
                scene_xyz,
                object_points,
                trans,
            )
            spatial_prior = (
                spatial_prior
                + self.obj_dir_factor
                * obj_dir_salience
                * object_mask
            )

        if self.use_object_prior_in_sca and not self.disable_object_conditioning:
            object_prior = (
                self.object_prior_factor
                * self.object_prior_scorer(object_embedding, scene_feats)
            )
            spatial_prior = spatial_prior + object_prior * object_mask

        if self.use_gt_point_prior_in_sca:
            if (
                gt_object_point_prior is None
                or gt_object_point_prior_mask is None
            ):
                raise ValueError(
                    "gt_object_point_prior and gt_object_point_prior_mask "
                    "are required for this model variant"
                )
            gt_object_point_prior = gt_object_point_prior.float()
            gt_object_point_prior_mask = (
                gt_object_point_prior_mask.float().view(bs, 1, 1)
            )
            gt_prior = gt_object_point_prior.unsqueeze(1).expand(
                -1,
                motion_seq_len,
                -1,
            )
            spatial_prior = (
                spatial_prior
                + self.gt_prior_factor
                * gt_prior
                * gt_object_point_prior_mask
            )

        out = motions.clone()
        for block in self.sca_blocks:
            out = block(out, scene_feats, spatial_prior)
        out_sca = out.clone()
        pose = self.pose_fc(out_sca)

        latent = torch.cat([ori, trans, pose], dim=-1)
        reconstructed = latent_to_joints(
            latent,
            self.vposer,
            self.smplx_model,
        )[:, :, :23]

        # Preserve the 25 observed ground-truth frames.  The original model
        # slices at frame 6, so it cannot represent HIK's H25F50 boundary.
        reconstructed = torch.cat(
            [joints, reconstructed[:, input_seq_len:]],
            dim=1,
        ).clone()
        pred_joints_full = self.motion_decoder(reconstructed)[:, :, :23]

        if return_features:
            features = {
                "motion_enc": motions,
                "object_emb": object_embedding,
                "out_tia": out_tia,
                "out_sca": out_sca,
                "ori_trans": torch.cat([ori, trans], dim=-1),
                "pose_latent": pose,
            }
            return latent, pred_joints_full, features
        return latent, pred_joints_full
