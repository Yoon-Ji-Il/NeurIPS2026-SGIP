"""Canonical want taxonomy and object → want mapping."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final


CANONICAL_WANTS: Final[tuple[str, ...]] = (
    "sit", "sleep", "drink", "eat", "read", "use", "take", "open",
    "light", "clean", "dress", "throw", "cook", "write", "fix", "unknown",
)
CANONICAL_WANT_SET: Final[frozenset[str]] = frozenset(CANONICAL_WANTS)
UNKNOWN_WANT: Final[str] = "unknown"
FALLBACK_WANT: Final[str] = "use"


OBJECT_ALIASES: Final[dict[str, str]] = {
    "air_conditionor": "air_conditioner",
    "bannana": "banana",
    "cloth": "clothes",
    "extra_things": "extra_thing",
    "fitness_equipment": "health_equipment",
    "mahcine": "kettle",
    "machine": "kettle",
    "shoes": "shoe",
    "tv_monitor": "tv_screen",
    "water_machine": "water_dispenser",
    "windows": "window",
}


SPECIFIC_OBJECT_TO_WANT: Final[dict[str, str]] = {
    "banana": "eat", "bed": "sleep", "bench": "sit", "bin": "throw",
    "book": "read", "bottle": "drink", "bowl": "eat", "box": "take",
    "button": "light", "cabinet": "take", "chair": "sit", "clothes": "dress",
    "cup": "drink", "cushion": "sit", "cutting_board": "cook", "door": "open",
    "drawer": "take", "eraser": "clean", "glue": "fix", "kettle": "cook",
    "lamp": "light", "paper": "write", "pen": "write", "plate": "eat",
    "pot": "cook", "refrigerator": "take", "shelf": "take", "shoe": "dress",
    "sink": "clean", "snack": "eat", "sofa": "sit", "spoon": "eat",
    "tissue": "clean", "trash": "throw", "water_dispenser": "drink",
    "window": "open",
}

FALLBACK_USE_OBJECTS: Final[frozenset[str]] = frozenset({
    "air_conditioner", "bag", "carrier", "dishwasher", "flower",
    "game_machine", "health_equipment", "ipad", "pipe", "plug",
    "weighing_scale",
})

OBJECT_TO_WANT: Final[dict[str, str]] = {
    **SPECIFIC_OBJECT_TO_WANT,
    **{name: FALLBACK_WANT for name in FALLBACK_USE_OBJECTS},
}


def _slugify(name: str) -> str:
    text = str(name).strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _has_token(text: str, token: str) -> bool:
    return token in text.split("_")


def normalize_object_name(name: str) -> str:
    text = _slugify(name)
    if not text:
        return ""
    if text in OBJECT_ALIASES:
        return OBJECT_ALIASES[text]
    if "shelf" in text:
        return "shelf"
    if "flower" in text:
        return "flower"
    if _has_token(text, "bag"):
        return "bag"
    if "trophy" in text or "tropy" in text:
        return "trophy"
    return text


def map_object_to_want(name: str) -> str:
    normalized = normalize_object_name(name)
    if not normalized or normalized == UNKNOWN_WANT:
        return UNKNOWN_WANT
    return OBJECT_TO_WANT.get(normalized, UNKNOWN_WANT)


def is_canonical_want(want: str) -> bool:
    return str(want).strip() in CANONICAL_WANT_SET


def validate_canonical_want(want: str) -> str:
    normalized = str(want).strip()
    if normalized not in CANONICAL_WANT_SET:
        raise ValueError(f"Unknown canonical want {want!r}; allowed: {CANONICAL_WANTS}")
    return normalized


__all__ = [
    "CANONICAL_WANTS", "CANONICAL_WANT_SET", "FALLBACK_USE_OBJECTS",
    "FALLBACK_WANT", "OBJECT_ALIASES", "OBJECT_TO_WANT",
    "SPECIFIC_OBJECT_TO_WANT", "UNKNOWN_WANT",
    "is_canonical_want", "map_object_to_want", "normalize_object_name",
    "validate_canonical_want",
]
