from collections import OrderedDict

import torch
import torch.nn as nn

from model.sif3d_final import SIF3D


class LayerCaptureGCN(nn.Module):
    """GCN wrapper that keeps diagnostic outputs after each residual block."""

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


class SIF3DDestinationRefiner(SIF3D):
    """Frozen SIF3D plus tiny final-frame residual heads.

    The root residual translates the entire destination skeleton, so it changes
    destination trajectory without altering root-relative pose. The pose
    residual changes only non-root joints at the destination frame.
    """

    def __init__(self, config, vposer, smplx_model):
        super().__init__(config, vposer, smplx_model)
        feature_dim = 38 + 23 * 3
        hidden_dim = int(getattr(config, "dest_refine_hidden_dim", 128))

        self.dest_root_refiner = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.dest_pose_refiner = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 22 * 3),
        )
        nn.init.zeros_(self.dest_root_refiner[-1].weight)
        nn.init.zeros_(self.dest_root_refiner[-1].bias)
        nn.init.zeros_(self.dest_pose_refiner[-1].weight)
        nn.init.zeros_(self.dest_pose_refiner[-1].bias)

        self.refine_variant = getattr(config, "dest_refine_variant", "root")
        self.root_scale = float(getattr(config, "dest_root_residual_scale", 0.25))
        self.pose_scale = float(getattr(config, "dest_pose_residual_scale", 0.10))
        self.layer_capture_enabled = False
        self.layer_mix_root = getattr(config, "layer_mix_root", "after_block_8")
        self.layer_mix_pose = getattr(config, "layer_mix_pose", "after_block_5")

        self.last_base_joints = None
        self.last_root_delta = None
        self.last_pose_delta = None
        self.last_mix_target = None

    def enable_layer_capture(self):
        if not isinstance(self.motion_decoder, LayerCaptureGCN):
            self.motion_decoder = LayerCaptureGCN(self.motion_decoder)
        self.layer_capture_enabled = True

    def freeze_base(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.dest_root_refiner.parameters():
            param.requires_grad = True
        for param in self.dest_pose_refiner.parameters():
            param.requires_grad = True

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
        if return_features:
            latent, base_joints, features = super().forward(
                motions,
                joints,
                scene_xyz,
                objects,
                object_mask,
                gt_object_point_prior,
                gt_object_point_prior_mask,
                return_features=True,
            )
        else:
            latent, base_joints = super().forward(
                motions,
                joints,
                scene_xyz,
                objects,
                object_mask,
                gt_object_point_prior,
                gt_object_point_prior_mask,
            )
        self.last_base_joints = base_joints.detach()
        self.last_root_delta = None
        self.last_pose_delta = None
        self.last_mix_target = None

        if self.layer_capture_enabled:
            layer_outputs = self.motion_decoder.last_outputs
            if self.layer_mix_root in layer_outputs and self.layer_mix_pose in layer_outputs:
                root_joints = layer_outputs[self.layer_mix_root][:, :, [0]]
                pose_rel = layer_outputs[self.layer_mix_pose] - layer_outputs[self.layer_mix_pose][:, :, [0]]
                self.last_mix_target = (root_joints + pose_rel).detach()

        refined = base_joints.clone()
        dest_feat = torch.cat(
            [
                latent[:, -1],
                base_joints[:, -1].reshape(base_joints.shape[0], -1),
            ],
            dim=-1,
        )

        if self.refine_variant in {"root", "both", "mix_b8_pose_b5"}:
            root_delta = self.root_scale * self.dest_root_refiner(dest_feat)
            refined[:, -1] = refined[:, -1] + root_delta[:, None, :]
            self.last_root_delta = root_delta

        if self.refine_variant in {"pose", "both", "mix_b8_pose_b5"}:
            pose_delta = self.pose_scale * self.dest_pose_refiner(dest_feat).view(-1, 22, 3)
            refined[:, -1, 1:] = refined[:, -1, 1:] + pose_delta
            self.last_pose_delta = pose_delta

        if return_features:
            return latent, refined, features
        return latent, refined
