import torch


BONE_PAIRS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (15, 22),
]


def calc_loss_bone(
    joints_predict,
    joints_label,
    joints_input,
    bone_pairs,
    kind="l1",
) -> torch.Tensor:
    joints_target = torch.cat([joints_input, joints_label], dim=1)
    if joints_predict.shape != joints_target.shape:
        raise ValueError(
            f"Pred/target shapes must match for bone loss, got {joints_predict.shape} vs {joints_target.shape}"
        )
    if joints_predict.ndim != 4 or joints_predict.shape[-1] != 3:
        raise ValueError(f"Expected joints shaped [B, T, J, 3], got {joints_predict.shape}")
    if bone_pairs.ndim != 2 or bone_pairs.shape[1] != 2:
        raise ValueError(f"bone_pairs must have shape [num_bones, 2], got {bone_pairs.shape}")
    if bone_pairs.numel() == 0:
        raise ValueError("bone_pairs must be non-empty")

    num_joints = joints_predict.shape[2]
    min_index = int(bone_pairs.min().item())
    max_index = int(bone_pairs.max().item())
    if min_index < 0 or max_index >= num_joints:
        raise ValueError(
            f"bone_pairs indices must be within [0, {num_joints - 1}], got min={min_index} max={max_index}"
        )

    parents = bone_pairs[:, 0]
    children = bone_pairs[:, 1]

    pred_bones = joints_predict[:, :, children] - joints_predict[:, :, parents]
    target_bones = joints_target[:, :, children] - joints_target[:, :, parents]

    pred_len = pred_bones.norm(dim=-1)
    target_len = target_bones.norm(dim=-1)
    diff = pred_len - target_len

    if kind == "l1":
        return diff.abs().mean()
    if kind == "l2":
        return diff.square().mean()

    raise ValueError(f"Unsupported bone loss type: {kind}")
