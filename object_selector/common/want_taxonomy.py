from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final


CANONICAL_WANTS: Final[tuple[str, ...]] = (
    "sit",
    "sleep",
    "drink",
    "eat",
    "read",
    "use",
    "take",
    "open",
    "light",
    "clean",
    "dress",
    "throw",
    "cook",
    "write",
    "fix",
    "unknown",
)

CANONICAL_WANT_SET: Final[frozenset[str]] = frozenset(CANONICAL_WANTS)
UNKNOWN_WANT: Final[str] = "unknown"
FALLBACK_WANT: Final[str] = "use"


OBJECT_ALIASES: Final[dict[str, str]] = {
    "air_conditioner": "air_conditioner",
    "air_conditionor": "air_conditioner",
    "bannana": "banana",
    "cloth": "clothes",
    "cutting_board": "cutting_board",
    "extra_things": "extra_thing",
    "fitness_equipment": "health_equipment",
    "game_machine": "game_machine",
    "health_equipment": "health_equipment",
    "mahcine": "kettle",
    # TODO: remove this alias if the raw "machine" label is later split into
    # multiple machine-like categories. In the current scene graphs it is the
    # same typo family as "mahcine".
    "machine": "kettle",
    "shoes": "shoe",
    "tv_screen": "tv_screen",
    "tv_monitor": "tv_screen",
    "water_machine": "water_dispenser",
    "weighing_scale": "weighing_scale",
    "windows": "window",
}


SPECIFIC_OBJECT_TO_WANT: Final[dict[str, str]] = {
    "banana": "eat",
    "bed": "sleep",
    "bench": "sit",
    "bin": "throw",
    "book": "read",
    "bottle": "drink",
    "bowl": "eat",
    "box": "take",
    "button": "light",
    "cabinet": "take",
    "chair": "sit",
    "clothes": "dress",
    "cup": "drink",
    "cushion": "sit",
    "cutting_board": "cook",
    "door": "open",
    "drawer": "take",
    "eraser": "clean",
    "glue": "fix",
    "kettle": "cook",
    "lamp": "light",
    "paper": "write",
    "pen": "write",
    "plate": "eat",
    "pot": "cook",
    "refrigerator": "take",
    "shelf": "take",
    "shoe": "dress",
    "sink": "clean",
    "snack": "eat",
    "sofa": "sit",
    "spoon": "eat",
    "tissue": "clean",
    "trash": "throw",
    "water_dispenser": "drink",
    "window": "open",
}

FALLBACK_USE_OBJECTS: Final[frozenset[str]] = frozenset(
    {
        "air_conditioner",
        "bag",
        "carrier",
        "dishwasher",
        "flower",
        "game_machine",
        "health_equipment",
        "ipad",
        "pipe",
        "plug",
        "weighing_scale",
    }
)

KNOWN_FALLBACK_USE_OBJECTS: Final[frozenset[str]] = FALLBACK_USE_OBJECTS
OBJECT_TO_WANT: Final[dict[str, str]] = {
    **SPECIFIC_OBJECT_TO_WANT,
    **{object_name: FALLBACK_WANT for object_name in FALLBACK_USE_OBJECTS},
}


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
    if "tropy" in text or "trophy" in text:
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
        allowed = ", ".join(CANONICAL_WANTS)
        raise ValueError(f"Unknown canonical want {want!r}; allowed: {allowed}")
    return normalized


def is_known_object_name(name: str) -> bool:
    normalized = normalize_object_name(name)
    return normalized in OBJECT_TO_WANT


def summarize_object_wants(names: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for original in names:
        raw = str(original).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        normalized = normalize_object_name(raw)
        rows.append(
            {
                "object": raw,
                "normalized_object": normalized,
                "want": map_object_to_want(raw),
                "mapping_source": mapping_source_for_object(raw),
            }
        )
    return sorted(rows, key=lambda row: (row["normalized_object"], row["object"]))


def unknown_or_fallback_objects(names: Iterable[str]) -> list[dict[str, str]]:
    return [
        row
        for row in summarize_object_wants(names)
        if row["want"] == UNKNOWN_WANT
    ]


def mapping_source_for_object(name: str) -> str:
    normalized = normalize_object_name(name)
    if not normalized or normalized == UNKNOWN_WANT:
        return "unknown"
    if normalized in SPECIFIC_OBJECT_TO_WANT:
        return "specific"
    if normalized in FALLBACK_USE_OBJECTS:
        return "fallback_use"
    return "unknown"


def validate_object_to_want_mapping(mapping: Mapping[str, str] = OBJECT_TO_WANT) -> None:
    overlap = sorted(set(SPECIFIC_OBJECT_TO_WANT) & set(FALLBACK_USE_OBJECTS))
    if overlap:
        raise ValueError(f"Objects cannot be both specific and fallback-use mapped: {overlap}")
    bad_wants = sorted({want for want in mapping.values() if want not in CANONICAL_WANT_SET})
    if bad_wants:
        raise ValueError(f"Object mapping contains non-canonical wants: {bad_wants}")
    bad_fallbacks = sorted(
        name
        for name, want in mapping.items()
        if want == FALLBACK_WANT and name not in FALLBACK_USE_OBJECTS
    )
    if bad_fallbacks:
        raise ValueError(f"{FALLBACK_WANT!r} is fallback-only; unexpected explicit use mappings: {bad_fallbacks}")


def _slugify(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _has_token(text: str, token: str) -> bool:
    return token in text.split("_")


validate_object_to_want_mapping()
