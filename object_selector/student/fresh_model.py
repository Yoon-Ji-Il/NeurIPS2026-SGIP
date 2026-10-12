from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(~valid, -torch.inf)
    has_valid = valid.any(dim=1, keepdim=True)
    safe = torch.where(has_valid, masked, torch.zeros_like(masked))
    probs = torch.softmax(safe, dim=1)
    return torch.where(has_valid, probs, torch.zeros_like(probs))


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(hidden_dim, output_dim), nn.ReLU()])
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def make_head(input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class FreshStudentSelector(nn.Module):
    """Small standalone student selector.

    It receives only student-side candidate/object features and past-motion
    context. It predicts object-wise semantic/final distributions plus an
    auxiliary want label for distillation.
    """

    def __init__(
        self,
        *,
        num_labels: int,
        num_wants: int,
        candidate_feature_dim: int,
        context_feature_dim: int,
        label_dim: int = 32,
        hidden_dim: int = 192,
        condition_on_want: bool = True,
        want_conditioning_mode: str = "memory",
        want_object_bias_weight: float = 0.0,
        dropout: float = 0.0,
        component_target_dim: int = 0,
        candidate_attention_layers: int = 0,
        context_attention: bool = False,
        motion_token_dim: int = 0,
        candidate_motion_attention: bool = False,
    ) -> None:
        super().__init__()
        if want_conditioning_mode not in {"concat", "memory"}:
            raise ValueError(f"Unsupported want_conditioning_mode: {want_conditioning_mode}")
        self.num_wants = int(num_wants)
        self.condition_on_want = condition_on_want
        self.want_conditioning_mode = want_conditioning_mode
        self.want_object_bias_weight = float(want_object_bias_weight)
        self.component_target_dim = int(component_target_dim)
        self.candidate_attention_layers = int(candidate_attention_layers)
        self.context_attention_enabled = bool(context_attention)
        self.motion_token_dim = int(motion_token_dim)
        self.candidate_motion_attention_enabled = bool(candidate_motion_attention)
        self.label_embedding = nn.Embedding(num_labels, label_dim)
        self.context_encoder = make_mlp(context_feature_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.candidate_encoder = make_mlp(candidate_feature_dim + label_dim, hidden_dim, hidden_dim, dropout=dropout)
        if self.candidate_attention_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.candidate_attention = nn.TransformerEncoder(encoder_layer, num_layers=self.candidate_attention_layers)
        else:
            self.candidate_attention = None
        if self.context_attention_enabled:
            self.context_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.context_attention = None
        if self.candidate_motion_attention_enabled:
            if self.motion_token_dim <= 0:
                raise ValueError("motion_token_dim must be > 0 when candidate_motion_attention is enabled")
            self.motion_token_encoder = make_mlp(self.motion_token_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.candidate_motion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.motion_token_encoder = None
            self.candidate_motion_attention = None
        sample_fuser_dim = hidden_dim * (3 if self.context_attention_enabled else 2)
        self.sample_fuser = make_mlp(sample_fuser_dim, hidden_dim, hidden_dim, dropout=dropout)
        if self.condition_on_want and self.want_conditioning_mode == "concat":
            self.want_condition_encoder = make_mlp(num_wants, hidden_dim, hidden_dim, dropout=dropout)
            self.want_head = make_head(hidden_dim, hidden_dim, num_wants, dropout=dropout)
        elif self.condition_on_want and self.want_conditioning_mode == "memory":
            self.want_query = nn.Linear(hidden_dim, hidden_dim)
            self.want_keys = nn.Parameter(torch.empty(num_wants, hidden_dim))
            self.want_values = nn.Parameter(torch.empty(num_wants, hidden_dim))
            nn.init.xavier_uniform_(self.want_keys)
            nn.init.xavier_uniform_(self.want_values)
        else:
            self.want_head = make_head(hidden_dim, hidden_dim, num_wants, dropout=dropout)

        if self.condition_on_want and self.want_conditioning_mode == "memory":
            pair_dim = hidden_dim * 3 + 1
        else:
            pair_dim = hidden_dim * (3 if self.condition_on_want else 2)
        if self.candidate_motion_attention_enabled:
            pair_dim += hidden_dim
        self.semantic_head = make_head(pair_dim, hidden_dim, 1, dropout=dropout)
        self.final_head = make_head(pair_dim, hidden_dim, 1, dropout=dropout)
        self.component_head = (
            make_head(pair_dim, hidden_dim, self.component_target_dim, dropout=dropout)
            if self.component_target_dim > 0
            else None
        )

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        label_ids: torch.Tensor,
        context_features: torch.Tensor,
        valid_mask: torch.Tensor,
        motion_tokens: torch.Tensor | None = None,
        motion_token_mask: torch.Tensor | None = None,
        teacher_want_idx: torch.Tensor | None = None,
        want_teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError(f"candidate_features must be [B, N, D], got {tuple(candidate_features.shape)}")
        if label_ids.shape != candidate_features.shape[:2]:
            raise ValueError(f"label_ids must be [B, N], got {tuple(label_ids.shape)}")
        if valid_mask.shape != candidate_features.shape[:2]:
            raise ValueError(f"valid_mask must be [B, N], got {tuple(valid_mask.shape)}")
        label_features = self.label_embedding(label_ids.to(device=candidate_features.device))
        candidate_input = torch.cat([candidate_features, label_features], dim=-1)
        candidate_context = self.candidate_encoder(candidate_input)
        valid_bool = valid_mask.to(device=candidate_features.device, dtype=torch.bool)
        if self.candidate_attention is not None:
            candidate_context = self.candidate_attention(candidate_context, src_key_padding_mask=~valid_bool)
        valid = valid_mask.to(device=candidate_features.device, dtype=candidate_features.dtype)
        pooled_candidates = (candidate_context * valid[..., None]).sum(dim=1)
        pooled_candidates = pooled_candidates / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        motion_context = self.context_encoder(context_features)
        sample_parts = [motion_context, pooled_candidates]
        if self.context_attention is not None:
            attended_scene, _ = self.context_attention(
                motion_context[:, None, :],
                candidate_context,
                candidate_context,
                key_padding_mask=~valid_bool,
                need_weights=False,
            )
            sample_parts.append(attended_scene.squeeze(1))
        sample_context = self.sample_fuser(torch.cat(sample_parts, dim=-1))
        expanded_context = sample_context[:, None, :].expand(-1, candidate_features.shape[1], -1)
        pair_parts = [expanded_context, candidate_context]
        if self.candidate_motion_attention is not None:
            if motion_tokens is None:
                raise ValueError("motion_tokens are required when candidate_motion_attention is enabled")
            if motion_tokens.ndim != 3:
                raise ValueError(f"motion_tokens must be [B, T, D], got {tuple(motion_tokens.shape)}")
            if motion_token_mask is None:
                motion_token_mask = torch.ones(
                    motion_tokens.shape[:2],
                    dtype=torch.bool,
                    device=motion_tokens.device,
                )
            encoded_motion = self.motion_token_encoder(motion_tokens.to(device=candidate_features.device))
            motion_mask = motion_token_mask.to(device=candidate_features.device, dtype=torch.bool)
            attended_motion, _ = self.candidate_motion_attention(
                candidate_context,
                encoded_motion,
                encoded_motion,
                key_padding_mask=~motion_mask,
                need_weights=False,
            )
            pair_parts.append(attended_motion)
        want_object_score: torch.Tensor | None = None
        if self.condition_on_want and self.want_conditioning_mode == "memory":
            want_query = self.want_query(sample_context)
            want_logits = want_query @ self.want_keys.T / math.sqrt(float(want_query.shape[-1]))
            want_probs = torch.softmax(want_logits, dim=-1)
            want_probs = self._maybe_teacher_force_want_probs(
                want_probs=want_probs,
                teacher_want_idx=teacher_want_idx,
                ratio=want_teacher_forcing_ratio,
            )
            want_context = want_probs @ self.want_values
            if candidate_features.shape[-1] >= self.num_wants:
                object_want_prior = candidate_features[..., -self.num_wants :].to(device=candidate_features.device)
            else:
                # TODO: Make object-want prior an explicit dataset field if candidate features change.
                object_want_prior = torch.zeros(
                    (*candidate_features.shape[:2], self.num_wants),
                    dtype=candidate_features.dtype,
                    device=candidate_features.device,
                )
            want_object_score = (object_want_prior * want_probs[:, None, :]).sum(dim=-1, keepdim=True)
            pair_parts.extend(
                [
                    want_context[:, None, :].expand(-1, candidate_features.shape[1], -1),
                    want_object_score,
                ]
            )
        elif self.condition_on_want:
            want_logits = self.want_head(sample_context)
            want_probs = torch.softmax(want_logits, dim=-1)
            want_probs = self._maybe_teacher_force_want_probs(
                want_probs=want_probs,
                teacher_want_idx=teacher_want_idx,
                ratio=want_teacher_forcing_ratio,
            )
            want_context = self.want_condition_encoder(want_probs)
            pair_parts.append(want_context[:, None, :].expand(-1, candidate_features.shape[1], -1))
        else:
            want_logits = self.want_head(sample_context)
        pair = torch.cat(pair_parts, dim=-1)

        semantic_logits = self.semantic_head(pair).squeeze(-1)
        final_logits = self.final_head(pair).squeeze(-1)
        component_prediction = (
            self.component_head(pair)
            if self.component_head is not None
            else torch.zeros((*candidate_features.shape[:2], 0), dtype=candidate_features.dtype, device=candidate_features.device)
        )
        if want_object_score is not None and self.want_object_bias_weight != 0.0:
            final_logits = final_logits + self.want_object_bias_weight * want_object_score.squeeze(-1)
        semantic_distribution = masked_softmax(semantic_logits, valid_mask)
        final_distribution = masked_softmax(final_logits, valid_mask)
        return {
            "semantic_logits": semantic_logits,
            "final_logits": final_logits,
            "semantic_distribution": semantic_distribution,
            "final_distribution": final_distribution,
            "want_logits": want_logits,
            "component_prediction": component_prediction,
        }

    def _maybe_teacher_force_want_probs(
        self,
        *,
        want_probs: torch.Tensor,
        teacher_want_idx: torch.Tensor | None,
        ratio: float,
    ) -> torch.Tensor:
        if teacher_want_idx is None or ratio <= 0.0:
            return want_probs
        clamped_ratio = max(0.0, min(float(ratio), 1.0))
        target = F.one_hot(
            teacher_want_idx.to(device=want_probs.device).clamp_min(0).clamp_max(self.num_wants - 1),
            num_classes=self.num_wants,
        ).to(dtype=want_probs.dtype)
        return (1.0 - clamped_ratio) * want_probs + clamped_ratio * target
