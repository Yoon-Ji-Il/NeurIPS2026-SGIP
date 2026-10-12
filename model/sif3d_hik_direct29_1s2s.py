"""VPoser-free SIF3D variant for native HIK 29-joint H25F50 forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from .gcn import GCN
from .sif3d_final import SIF3D as SIF3DGIMO
from .sif3d_final import human_centered_scene
from .sif3d_hik_1s2s import _resize_tensor_along


def load_gimo_checkpoint_for_direct29(model, checkpoint_path, map_location="cpu"):
    """Warm-start the representation-independent GIMO backbone only.

    Direct29 heads and its 87-node GCN have different semantics from the
    VPoser/69-node checkpoint. Mixing only shape-compatible decoder tensors
    makes the residual initialization unstable, so the entire direct output
    path deliberately stays freshly initialized.
    """

    source = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if isinstance(source, dict) and "state_dict" in source:
        source = source["state_dict"]
    target = model.state_dict()
    direct_prefixes = (
        "motion_emb.",
        "trajectory_fc.",
        "pose_fc.",
        "motion_decoder.",
        "joints_fc.",
        "vposer.",
        "smplx_model.",
    )
    temporal_dimensions = {
        "motion_encoder.latent": 0,
        "object_encoder.latent": 0,
    }
    loadable = {}
    exact = []
    interpolated = []
    skipped = []
    for key, value in source.items():
        if key.startswith(direct_prefixes) or key not in target:
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
        "direct29_representation": True,
        "direct_output_path_fresh": True,
        "vposer_or_smplx_used": False,
    }


class SIF3DDirect29(SIF3DGIMO):
    """Reuse SIF3D scene reasoning but predict root plus 28 local joints."""

    INPUT_FRAMES = 25
    FUTURE_FRAMES = 50
    N_JOINTS = 29

    def __init__(self, config):
        if (
            config.input_seq_len != self.INPUT_FRAMES
            or config.output_seq_len != self.FUTURE_FRAMES
        ):
            raise ValueError(
                "SIF3DDirect29 requires 25 observed and 50 future frames"
            )
        super().__init__(config, vposer=None, smplx_model=None)
        total_frames = config.input_seq_len + config.output_seq_len
        self.motion_emb = nn.Linear(
            self.N_JOINTS * 3,
            config.motion_hidden_dim,
        )
        # Keep the six-dimensional GIMO-compatible head. Only orientation-y
        # and the three root coordinates are consumed by this representation.
        self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)
        self.pose_fc = nn.Linear(
            config.motion_latent_dim,
            (self.N_JOINTS - 1) * 3,
        )
        self.motion_decoder = GCN(
            config,
            input_feature=total_frames,
            node_n=self.N_JOINTS * 3,
        )
        del self.joints_fc
        # The representation starts from a strong "last observed skeleton"
        # baseline. Heads learn corrections rather than metre-scale absolute
        # coordinates, and the GCN starts as an exact residual identity.
        nn.init.zeros_(self.trajectory_fc.weight)
        nn.init.zeros_(self.trajectory_fc.bias)
        nn.init.zeros_(self.pose_fc.weight)
        nn.init.zeros_(self.pose_fc.bias)
        nn.init.zeros_(self.motion_decoder.gc7.weight)
        nn.init.zeros_(self.motion_decoder.gc7.att)
        nn.init.zeros_(self.motion_decoder.gc7.bias)

    def forward(
        self,
        joints,
        orientation_input,
        scene_xyz,
        objects,
        object_mask,
        *,
        return_features=False,
    ):
        bs, input_frames, n_joints, dims = joints.shape
        if (input_frames, n_joints, dims) != (
            self.INPUT_FRAMES,
            self.N_JOINTS,
            3,
        ):
            raise ValueError(
                "Expected direct joints [B,25,29,3], got "
                f"{tuple(joints.shape)}"
            )
        if tuple(orientation_input.shape) != (bs, self.INPUT_FRAMES, 3):
            raise ValueError(
                "Expected observed orientation [B,25,3], got "
                f"{tuple(orientation_input.shape)}"
            )
        if objects.ndim != 3 or objects.shape[-1] != 3:
            raise ValueError(
                f"Expected object points [B,P,3], got {tuple(objects.shape)}"
            )
        object_mask = object_mask.float().view(bs, 1, 1)

        scene_feats, _ = self.pointnet(scene_xyz.repeat(1, 1, 2))
        scene_feats = scene_feats.transpose(1, 2)
        motions = self.motion_emb(joints.reshape(bs, input_frames, -1))
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
            object_embedding = self.object_encoder(objects) * object_mask

        out = motions.clone()
        for block in self.tia_blocks:
            out = block(out, scene_feats, object_embedding)
        out_tia = out
        orientation_residual, root_residual = torch.split(
            self.trajectory_fc(out_tia),
            [3, 3],
            dim=-1,
        )
        future_frames = motion_seq_len - input_frames
        orientation_baseline = torch.cat(
            [
                orientation_input,
                orientation_input[:, [-1]].expand(-1, future_frames, -1),
            ],
            dim=1,
        )
        root_observed = joints[:, :, 0]
        root_baseline = torch.cat(
            [
                root_observed,
                root_observed[:, [-1]].expand(-1, future_frames, -1),
            ],
            dim=1,
        )
        orientation = orientation_baseline + orientation_residual
        root = root_baseline + root_residual

        scene_xyz_normalized = human_centered_scene(
            scene_xyz,
            orientation,
            root,
        )
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
            object_direction = self._build_object_dir_prior(
                scene_xyz,
                objects,
                root,
            )
            spatial_prior = (
                spatial_prior
                + self.obj_dir_factor * object_direction * object_mask
            )
        if self.use_object_prior_in_sca and not self.disable_object_conditioning:
            object_prior = self.object_prior_factor * self.object_prior_scorer(
                object_embedding,
                scene_feats,
            )
            spatial_prior = spatial_prior + object_prior * object_mask

        out = motions.clone()
        for block in self.sca_blocks:
            out = block(out, scene_feats, spatial_prior)
        out_sca = out
        local_residual = self.pose_fc(out_sca).reshape(
            bs,
            motion_seq_len,
            self.N_JOINTS - 1,
            3,
        )
        observed_local = joints - joints[:, :, [0]]
        local_baseline = torch.cat(
            [
                observed_local[:, :, 1:],
                observed_local[:, [-1], 1:].expand(
                    -1,
                    future_frames,
                    -1,
                    -1,
                ),
            ],
            dim=1,
        )
        local_non_root = local_baseline + local_residual
        local = torch.cat(
            [
                torch.zeros(
                    bs,
                    motion_seq_len,
                    1,
                    3,
                    dtype=local_non_root.dtype,
                    device=local_non_root.device,
                ),
                local_non_root,
            ],
            dim=2,
        )
        pre_gcn_joints = root.unsqueeze(2) + local
        decoder_input = torch.cat(
            [joints, pre_gcn_joints[:, input_frames:]],
            dim=1,
        ).clone()
        predicted_joints = self.motion_decoder(decoder_input)

        auxiliary = {
            "orientation": orientation,
            "root": root,
            "pre_gcn_joints": pre_gcn_joints,
        }
        if return_features:
            auxiliary.update(
                {
                    "motion_enc": motions,
                    "object_emb": object_embedding,
                    "out_tia": out_tia,
                    "out_sca": out_sca,
                }
            )
        return predicted_joints, auxiliary
