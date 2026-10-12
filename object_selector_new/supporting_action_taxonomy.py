"""Supporting action sequences for each canonical want."""

from __future__ import annotations

from typing import Final

from object_selector_new.common.want_taxonomy import (
    CANONICAL_WANTS,
    CANONICAL_WANT_SET,
    UNKNOWN_WANT,
    validate_canonical_want,
)


WANT_TO_SUPPORTING_ACTIONS: Final[dict[str, tuple[str, ...]]] = {
    "sit":     ("approach", "turn", "sit"),
    "sleep":   ("approach", "lie_down", "rest"),
    "drink":   ("reach", "grasp", "hold", "drink"),
    "eat":     ("reach", "grasp", "bring_close", "eat"),
    "read":    ("pick", "hold", "read"),
    "use":     ("reach", "hold", "operate"),
    "take":    ("open", "reach", "grasp", "take"),
    "open":    ("reach", "grasp", "pull_or_push", "open"),
    "light":   ("reach", "press", "switch_on"),
    "clean":   ("reach", "wipe", "clean"),
    "dress":   ("pick", "wear", "adjust"),
    "throw":   ("pick", "move", "throw_away"),
    "cook":    ("reach", "hold", "prepare", "cook"),
    "write":   ("grasp", "write"),
    "fix":     ("grasp", "apply", "attach_or_fix"),
    "unknown": ("interact",),
}


def get_supporting_actions_for_want(want: str) -> list[str]:
    canonical = validate_canonical_want(want)
    return list(WANT_TO_SUPPORTING_ACTIONS[canonical])


__all__ = ["WANT_TO_SUPPORTING_ACTIONS", "get_supporting_actions_for_want"]
