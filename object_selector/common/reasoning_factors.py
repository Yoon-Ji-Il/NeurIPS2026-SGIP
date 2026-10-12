from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


DEFAULT_REASONING_FACTORS: tuple[str, ...] = (
    "function_match",
    "posture_match",
    "orientation_match",
    "distance_preference",
    "interaction_feasibility",
)


FACTOR_DEFINITIONS: dict[str, str] = {
    "function_match": "How well the object's semantic affordance satisfies the want.",
    "posture_match": "How well the object supports the implied body posture or manipulation mode.",
    "orientation_match": "How well face/root/torso orientation points toward the object.",
    "distance_preference": "How spatially natural the object is given root/head distance.",
    "interaction_feasibility": "How feasible the interaction is given object size, location, and accessibility.",
}


@dataclass(frozen=True)
class ReasoningFactorSpec:
    names: tuple[str, ...] = DEFAULT_REASONING_FACTORS
    definitions: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.definitions is None:
            object.__setattr__(self, "definitions", FACTOR_DEFINITIONS)
        missing = [name for name in self.names if name not in self.definitions]
        if missing:
            raise ValueError(f"Missing reasoning factor definitions: {missing}")

    def validate_scores(self, scores: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in self.names if name not in scores]
        extra = [name for name in scores if name not in self.names]
        if missing:
            raise ValueError(f"Missing reasoning factor scores: {missing}")
        if extra:
            raise ValueError(f"Unknown reasoning factor scores: {extra}")
        validated: dict[str, float] = {}
        for name in self.names:
            value = float(scores[name])
            if not math.isfinite(value):
                raise ValueError(f"Reasoning factor {name!r} is not finite: {value}")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Reasoning factor {name!r} must be in [0, 1], got {value}")
            validated[name] = value
        return validated

    def prompt_payload(self) -> list[dict[str, str]]:
        return [
            {
                "name": name,
                "definition": self.definitions[name],
                "score_anchor": "0.0=no evidence, 0.5=plausible but ambiguous, 1.0=strong evidence",
            }
            for name in self.names
        ]
