import torch

from utils.hik_metrics import compute_hik_horizon_metrics


def test_hik_metric_interpolates_exact_half_second_horizons():
    times = torch.arange(1, 51, dtype=torch.float32) / 25.0
    predicted = torch.zeros((1, 50, 2, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)
    predicted[0, :, 0, 0] = times
    predicted[0, :, 1, 0] = times * 2.0

    metrics = compute_hik_horizon_metrics(predicted, target)

    assert abs(metrics["trajectory"]["0.5s"] - 0.5) < 1e-6
    assert abs(metrics["trajectory"]["1.0s"] - 1.0) < 1e-6
    assert abs(metrics["trajectory"]["1.5s"] - 1.5) < 1e-6
    assert abs(metrics["trajectory"]["2.0s"] - 2.0) < 1e-6
    assert abs(metrics["trajectory"]["mean"] - 1.02) < 1e-6
    assert abs(metrics["mpjpe"]["1.5s"] - 0.75) < 1e-6


def test_hik_metric_accepts_full_h25f50_sequence():
    predicted = torch.zeros((2, 75, 23, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)
    metrics = compute_hik_horizon_metrics(predicted, target)

    assert metrics["trajectory"]["mean"] == 0.0
    assert metrics["mpjpe"]["2.0s"] == 0.0
