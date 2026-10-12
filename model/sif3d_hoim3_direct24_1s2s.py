"""VPoser-free SIF3D direct-24 model for HOI-M3 H30F60."""

from __future__ import annotations

from .sif3d_hik_direct29_1s2s import (
    SIF3DDirect29,
    load_gimo_checkpoint_for_direct29,
)


class SIF3DDirect24(SIF3DDirect29):
    INPUT_FRAMES = 30
    FUTURE_FRAMES = 60
    N_JOINTS = 24


def load_gimo_checkpoint_for_direct24(
    model,
    checkpoint_path,
    map_location="cpu",
):
    result = load_gimo_checkpoint_for_direct29(
        model,
        checkpoint_path,
        map_location=map_location,
    )
    result.pop("direct29_representation", None)
    result["direct24_representation"] = True
    result["dataset"] = "HOI-M3 Entity HIK-scale"
    return result
