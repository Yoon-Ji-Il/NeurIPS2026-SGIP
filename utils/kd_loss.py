import torch
import torch.nn.functional as F


def cosine_distill(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity feature alignment.

    Args:
        student_feat: [..., D] tensor (any leading shape).
        teacher_feat: same shape as student_feat.

    Returns:
        Scalar: 1 - mean cosine similarity over all leading dims.
    """
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            f"Shape mismatch: student={student_feat.shape} teacher={teacher_feat.shape}"
        )
    s_norm = F.normalize(student_feat, dim=-1)
    t_norm = F.normalize(teacher_feat.detach(), dim=-1)
    cos_sim = (s_norm * t_norm).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def smooth_l1_distill(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 feature alignment."""
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            f"Shape mismatch: student={student_feat.shape} teacher={teacher_feat.shape}"
        )
    return F.smooth_l1_loss(student_feat, teacher_feat.detach())


def feature_distill(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    kind: str = "cosine",
) -> torch.Tensor:
    """Dispatcher for feature distillation."""
    if kind == "cosine":
        return cosine_distill(student_feat, teacher_feat)
    if kind == "smooth_l1":
        return smooth_l1_distill(student_feat, teacher_feat)
    raise ValueError(f"Unknown kd_feat_kind: {kind}")
