from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


MOTIVATION_FIELDS: tuple[str, ...] = (
    "need_state",
    "posture_mode",
    "support_type",
    "duration_type",
)


DEFAULT_VOCAB: dict[str, dict[str, str]] = {
    "need_state": {
        "rest": "body support or recovery",
        "hydration": "drink or liquid intake",
        "nutrition": "eat or food intake",
        "access_storage": "retrieve, place, or access stored objects",
        "access_outside": "open or inspect an outside boundary",
        "device_operation": "operate switch, appliance, screen, or machine",
        "cleaning_maintenance": "clean, repair, wipe, water, or maintain",
        "information_leisure": "read, inspect, play, or information use",
        "object_transfer": "pick, carry, place, or move an object",
        "self_care": "wear, adjust, or care for the body",
        "exercise": "exercise or body measurement",
        "other_unknown": "mixed or unclear need",
    },
    "posture_mode": {
        "upright_supported": "sit or lean upright on support",
        "reclined_supported": "recline on support",
        "full_body_supported": "whole body support",
        "reach_manipulate": "reach forward and manipulate by hand",
        "inspect_operate": "look at or operate without large body support",
        "open_access": "open or close a boundary/container",
        "lift_carry": "lift, carry, or transfer object",
        "stand_balance": "stand on or balance against object",
        "locomote": "move through or around environment",
        "unknown": "unclear posture",
    },
    "support_type": {
        "single_seat_support": "chair or one-person seat",
        "wide_lounge_support": "sofa or lounge support",
        "full_body_surface": "bed or whole-body surface",
        "openable_storage": "cabinet, drawer, refrigerator, shelf, or storage",
        "openable_outside_boundary": "door, window, curtain, or outside boundary",
        "direct_drink_container": "cup or bottle used for drinking",
        "food_surface_container": "plate, bowl, pot, or eating surface",
        "portable_item": "small portable object",
        "hand_tool_or_switch": "tool, switch, button, screen, or appliance control",
        "bodywear_or_selfcare": "shoes, clothes, tissue, or body-care object",
        "exercise_or_measure_device": "exercise or measurement device",
        "other_unknown": "mixed or unclear affordance",
    },
    "duration_type": {
        "instant": "momentary action",
        "brief": "short interaction",
        "sustained": "multi-second interaction",
        "extended": "long-duration interaction",
        "unknown": "unclear duration",
    },
}


@dataclass(frozen=True)
class MotivationVocab:
    values: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_combined_csv(cls, path: Path) -> "MotivationVocab":
        values: dict[str, dict[str, str]] = {field: {} for field in MOTIVATION_FIELDS}
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    source = str(row.get("source", "")).strip()
                    field = str(row.get("field", "")).strip()
                    label = str(row.get("class", "")).strip()
                    definition = str(row.get("definition", "")).strip()
                    if source == "vocab" and field in values and label:
                        values[field][label] = definition or label

            # If the file has no explicit vocab rows, use data rows as an extensible fallback.
            if not any(values[field] for field in MOTIVATION_FIELDS):
                with path.open("r", encoding="utf-8-sig", newline="") as file:
                    for row in csv.DictReader(file):
                        if str(row.get("source", "data")).strip() not in {"", "data"}:
                            continue
                        for field in MOTIVATION_FIELDS:
                            label = str(row.get(field, "")).strip()
                            if label:
                                values[field].setdefault(label, label)

        for field in MOTIVATION_FIELDS:
            if not values[field]:
                values[field].update(DEFAULT_VOCAB[field])
        return cls(values)

    def allowed(self, field: str) -> tuple[str, ...]:
        if field not in self.values:
            raise KeyError(f"Unknown motivation field: {field}")
        return tuple(self.values[field].keys())

    def validate_code(self, code: Mapping[str, str]) -> dict[str, str]:
        missing = [field for field in MOTIVATION_FIELDS if field not in code]
        extra = [field for field in code if field not in MOTIVATION_FIELDS]
        if missing:
            raise ValueError(f"Missing motivation fields: {missing}")
        if extra:
            raise ValueError(f"Unknown motivation fields: {extra}")
        validated: dict[str, str] = {}
        for field in MOTIVATION_FIELDS:
            value = str(code[field]).strip()
            if value not in self.values[field]:
                allowed = ", ".join(self.allowed(field))
                raise ValueError(f"Invalid {field}={value!r}; allowed: {allowed}")
            validated[field] = value
        return validated

    def prompt_payload(self) -> dict[str, list[dict[str, str]]]:
        return {
            field: [
                {"class": label, "definition": definition}
                for label, definition in self.values[field].items()
            ]
            for field in MOTIVATION_FIELDS
        }
