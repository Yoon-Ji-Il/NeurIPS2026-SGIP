import torch

from utils.hik_direct29_losses import direct29_losses


def make_inputs():
    predicted = torch.zeros((1, 75, 29, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)
    orientation = torch.zeros((1, 75, 3), dtype=torch.float32)
    return predicted, target, orientation


def test_root_translation_does_not_enter_pose_loss():
    predicted, target, orientation = make_inputs()
    predicted[:, 25:, :, 0] = 0.4
    losses = direct29_losses(
        predicted,
        target,
        orientation,
        orientation,
    )
    assert torch.isclose(losses["root"], torch.tensor(0.4))
    assert torch.isclose(losses["pose"], torch.tensor(0.0))


def test_non_root_articulation_does_not_enter_root_loss():
    predicted, target, orientation = make_inputs()
    predicted[:, 25:, 1:, 2] = 0.2
    losses = direct29_losses(
        predicted,
        target,
        orientation,
        orientation,
    )
    assert torch.isclose(losses["root"], torch.tensor(0.0))
    assert torch.isclose(losses["pose"], torch.tensor(0.2))


def test_wrapped_yaw_is_separate_from_joint_losses():
    predicted, target, orientation = make_inputs()
    predicted_orientation = orientation.clone()
    predicted_orientation[:, 25:, 1] = torch.pi
    losses = direct29_losses(
        predicted,
        target,
        predicted_orientation,
        orientation,
    )
    assert torch.isclose(losses["root"], torch.tensor(0.0))
    assert torch.isclose(losses["pose"], torch.tensor(0.0))
    assert torch.isclose(losses["yaw"], torch.tensor(2.0))


def test_disjoint_does_not_add_destination_again():
    predicted, target, orientation = make_inputs()
    predicted[:, 25:, :, 0] = 0.1
    predicted[:, -1, :, 0] = 0.5
    losses = direct29_losses(
        predicted,
        target,
        orientation,
        orientation,
        mode="disjoint",
    )
    assert torch.isclose(losses["root_objective"], losses["root"])
    assert torch.isclose(losses["total"], losses["root"])


def test_legacy_overlap_adds_destination_again():
    predicted, target, orientation = make_inputs()
    predicted[:, 25:, 1:, 2] = 0.1
    predicted[:, -1, 1:, 2] = 0.5
    losses = direct29_losses(
        predicted,
        target,
        orientation,
        orientation,
        mode="legacy_overlap",
        destination_weight=1.0,
    )
    expected = losses["pose"] + losses["destination_pose_diagnostic"]
    assert torch.isclose(losses["pose_objective"], expected)
    assert torch.isclose(losses["total"], expected)
