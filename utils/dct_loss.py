import math

import torch


_DCT_MATRIX_CACHE = {}


def get_dct_matrix(N: int, device, dtype) -> torch.Tensor:
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    cache_key = (N, str(device), dtype)
    if cache_key not in _DCT_MATRIX_CACHE:
        time_idx = torch.arange(N, device=device, dtype=dtype)
        freq_idx = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        dct_m = torch.cos((math.pi / N) * (time_idx + 0.5) * freq_idx)
        scale = torch.full((N, 1), math.sqrt(2.0 / N), device=device, dtype=dtype)
        scale[0, 0] = math.sqrt(1.0 / N)
        _DCT_MATRIX_CACHE[cache_key] = dct_m * scale

    return _DCT_MATRIX_CACHE[cache_key]


def calc_loss_dct(joints_predict, joints_label, joints_input, K: int, kind: str = "l1") -> torch.Tensor:
    joints_target = torch.cat([joints_input, joints_label], dim=1)
    if joints_predict.shape != joints_target.shape:
        raise ValueError(
            f"Pred/target shapes must match for DCT loss, got {joints_predict.shape} vs {joints_target.shape}"
        )

    seq_len = joints_predict.shape[1]
    if K <= 0 or K > seq_len:
        raise ValueError(f"K must be in [1, {seq_len}], got {K}")

    joints_target = joints_target - joints_target[:, :, [0]]
    joints_predict = joints_predict - joints_predict[:, :, [0]]

    dct_m = get_dct_matrix(seq_len, joints_predict.device, joints_predict.dtype)[:K]
    pred_coeffs = torch.einsum("kt,btjc->bkjc", dct_m, joints_predict)
    target_coeffs = torch.einsum("kt,btjc->bkjc", dct_m, joints_target)
    diff = pred_coeffs - target_coeffs

    if kind == "l1":
        return diff.abs().mean()
    if kind == "l2":
        return diff.square().mean()

    raise ValueError(f"Unsupported DCT loss type: {kind}")
