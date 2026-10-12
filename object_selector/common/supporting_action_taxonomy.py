from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from object_selector.common.want_taxonomy import CANONICAL_WANTS, CANONICAL_WANT_SET, UNKNOWN_WANT


WANT_TO_SUPPORTING_ACTIONS: Final[dict[str, tuple[str, ...]]] = {
    "sit": ("approach", "turn", "sit"),
    "sleep": ("approach", "lie_down", "rest"),
    "drink": ("reach", "grasp", "hold", "drink"),
    "eat": ("reach", "grasp", "bring_close", "eat"),
    "read": ("pick", "hold", "read"),
    "use": ("reach", "hold", "operate"),
    "take": ("open", "reach", "grasp", "take"),
    "open": ("reach", "grasp", "pull_or_push", "open"),
    "light": ("reach", "press", "switch_on"),
    "clean": ("reach", "wipe", "clean"),
    "dress": ("pick", "wear", "adjust"),
    "throw": ("pick", "move", "throw_away"),
    "cook": ("reach", "hold", "prepare", "cook"),
    "write": ("grasp", "write"),
    "fix": ("grasp", "apply", "attach_or_fix"),
    "unknown": ("interact",),
}


def get_supporting_actions_for_want(want: str) -> list[str]:
    canonical_want = normalize_and_validate_want(want)
    return list(WANT_TO_SUPPORTING_ACTIONS[canonical_want])


def normalize_and_validate_want(want: str) -> str:
    canonical_want = str(want).strip().lower()
    if canonical_want not in CANONICAL_WANT_SET:
        allowed = ", ".join(CANONICAL_WANTS)
        raise ValueError(f"Unknown canonical want {want!r}; allowed: {allowed}")
    return canonical_want


def validate_supporting_action_taxonomy(
    mapping: Mapping[str, Sequence[str]] = WANT_TO_SUPPORTING_ACTIONS,
) -> None:
    missing = sorted(set(CANONICAL_WANTS) - set(mapping))
    extra = sorted(set(mapping) - set(CANONICAL_WANTS))
    if missing:
        raise ValueError(f"Missing supporting action sets for wants: {missing}")
    if extra:
        raise ValueError(f"Unknown wants in supporting action taxonomy: {extra}")

    for want in CANONICAL_WANTS:
        actions = tuple(str(action).strip() for action in mapping[want])
        if any(not action for action in actions):
            raise ValueError(f"Supporting actions for {want!r} must be non-empty strings")
        duplicates = sorted({action for action in actions if actions.count(action) > 1})
        if duplicates:
            raise ValueError(f"Supporting actions for {want!r} contain duplicates: {duplicates}")
        if want == UNKNOWN_WANT:
            if len(actions) != 1:
                raise ValueError("The unknown want should keep exactly one generic action")
        elif not 2 <= len(actions) <= 5:
            raise ValueError(f"Supporting actions for {want!r} must contain 2 to 5 actions")


validate_supporting_action_taxonomy()
