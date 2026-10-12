#!/usr/bin/env python
"""Induce a compact PROX-S intent taxonomy from normalized action labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_CSV = Path(
    "data/PROX/PROX_COINS_GIMO_EVENT_ANCHORED_GIMO_SPLIT/event_segments.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "./analysis/prox_action_intent_qwen3_14b"
)
ACTION_ALIASES = {"wrong_sit on": "sit on"}


def load_action_objects(csv_path: Path) -> dict[str, list[str]]:
    action_objects: dict[str, set[str]] = defaultdict(set)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            action = row["target_action_prefix"].strip().lower()
            action = ACTION_ALIASES.get(action, action)
            object_name = row["target_object_name"].strip().lower()
            action_objects[action].add(object_name)
    return {
        action: sorted(object_names)
        for action, object_names in sorted(action_objects.items())
    }


def build_open_prompt(action_objects: dict[str, list[str]]) -> str:
    payload = [
        {"action": action, "observed_objects": object_names}
        for action, object_names in action_objects.items()
    ]
    return (
        "You are constructing a high-level human-intent taxonomy for motion forecasting "
        "on a processed PROX-S subset.\n\n"
        "The input contains 13 normalized action labels. observed_objects lists object "
        "types paired with each action in the data; use them only to disambiguate meaning.\n\n"
        "Task:\n"
        "1. Group the actions into a compact set of semantically coherent high-level intents.\n"
        "2. Choose the number of intent categories yourself. No target number is given.\n"
        "3. An intent should describe the intended interaction outcome, not merely copy an "
        "action label or describe upward/downward motion.\n"
        "4. Assign every action to exactly one intent and do not omit or invent actions.\n"
        "5. Merge actions that express the same underlying goal, including different phases "
        "of that goal, but keep genuinely different goals separate.\n"
        "6. Avoid singleton categories unless semantically necessary.\n"
        "7. Use short lowercase snake_case intent names suitable as dataset labels.\n"
        "8. Report ambiguities caused by having only action-object labels and no RGB context.\n\n"
        "Return strict JSON only, with this schema:\n"
        "{\n"
        '  "category_count": <integer>,\n'
        '  "categories": [\n'
        "    {\n"
        '      "intent": "<lowercase_snake_case>",\n'
        '      "definition": "<one sentence>",\n'
        '      "actions": ["<action>", "..."],\n'
        '      "rationale": "<one concise sentence>"\n'
        "    }\n"
        "  ],\n"
        '  "ambiguities": ["<concise ambiguity>", "..."],\n'
        '  "summary": "<one concise sentence explaining why this category count is appropriate>"\n'
        "}\n\n"
        "INPUT_ACTIONS:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_constrained7_prompt(action_objects: dict[str, list[str]]) -> str:
    payload = [
        {"action": action, "observed_objects": object_names}
        for action, object_names in action_objects.items()
    ]
    return (
        "You are assigning normalized PROX-S action labels to a fixed high-level "
        "intent taxonomy for motion forecasting.\n\n"
        "Use exactly these seven intent labels and do not rename, add, or remove them:\n"
        "- sit_up\n"
        "- sit_down\n"
        "- lie\n"
        "- walk\n"
        "- touch\n"
        "- jump\n"
        "- unknown\n\n"
        "The following anchor assignments are fixed by the dataset designer and must "
        "not be changed:\n"
        "- sit_up: sit up, stand up\n"
        "- sit_down: sit down\n"
        "- lie: lie on, lie down\n"
        "- walk: step up, move leg\n"
        "- touch: touch\n"
        "- jump: jump on\n\n"
        "The designer originally wrote 'stand down' under sit_down, but no such input "
        "action exists; it is normalized here to the actual action 'sit down'.\n\n"
        "Assign the remaining actions (move on, sit on, stand on, step down) to one of "
        "the seven intents. Assign an action to a semantic intent only when the "
        "action-object label provides sufficient evidence. If its high-level intent is "
        "ambiguous without RGB or temporal context, assign it to unknown. Prefer "
        "unknown over an unsupported semantic guess.\n\n"
        "Requirements:\n"
        "1. Return exactly seven categories in the order listed above.\n"
        "2. Assign every input action exactly once; do not omit, duplicate, or invent actions.\n"
        "3. Preserve every fixed anchor assignment exactly.\n"
        "4. Keep unknown as an explicit non-empty category for ambiguous actions.\n"
        "5. Use observed_objects only as weak disambiguating evidence.\n"
        "6. Explain each non-anchor assignment concisely.\n\n"
        "Return strict JSON only, with this schema:\n"
        "{\n"
        '  "category_count": 7,\n'
        '  "categories": [\n'
        "    {\n"
        '      "intent": "<one fixed label>",\n'
        '      "definition": "<one sentence>",\n'
        '      "actions": ["<action>", "..."],\n'
        '      "rationale": "<one concise sentence>"\n'
        "    }\n"
        "  ],\n"
        '  "ambiguous_assignments": [\n'
        "    {\n"
        '      "action": "<action assigned to unknown>",\n'
        '      "reason": "<why action-object labels are insufficient>"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "<one concise sentence>"\n'
        "}\n\n"
        "INPUT_ACTIONS:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model output does not contain a JSON object")
        result = json.loads(stripped[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Model output JSON must be an object")
    return result


def validate_result(result: dict[str, Any], expected_actions: list[str]) -> None:
    categories = result.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories must be a non-empty list")
    if result.get("category_count") != len(categories):
        raise ValueError("category_count does not equal len(categories)")

    assigned: list[str] = []
    intents: list[str] = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("Each category must be an object")
        intent = category.get("intent")
        actions = category.get("actions")
        if not isinstance(intent, str) or not intent:
            raise ValueError("Each category must have a non-empty intent")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Intent {intent!r} must have a non-empty actions list")
        intents.append(intent)
        assigned.extend(actions)

    if len(intents) != len(set(intents)):
        raise ValueError("Intent names must be unique")
    if sorted(assigned) != sorted(expected_actions):
        missing = sorted(set(expected_actions) - set(assigned))
        extra = sorted(set(assigned) - set(expected_actions))
        duplicates = sorted({action for action in assigned if assigned.count(action) > 1})
        raise ValueError(
            f"Invalid action assignment: missing={missing}, extra={extra}, duplicates={duplicates}"
        )


def validate_constrained7_result(result: dict[str, Any]) -> None:
    expected_intents = [
        "sit_up",
        "sit_down",
        "lie",
        "walk",
        "touch",
        "jump",
        "unknown",
    ]
    categories = result["categories"]
    actual_intents = [category["intent"] for category in categories]
    if actual_intents != expected_intents:
        raise ValueError(
            f"Expected ordered intents {expected_intents}, got {actual_intents}"
        )
    action_sets = {
        category["intent"]: set(category["actions"])
        for category in categories
    }
    fixed_anchors = {
        "sit_up": {"sit up", "stand up"},
        "sit_down": {"sit down"},
        "lie": {"lie on", "lie down"},
        "walk": {"step up", "move leg"},
        "touch": {"touch"},
        "jump": {"jump on"},
    }
    violations = {
        intent: sorted(actions - action_sets[intent])
        for intent, actions in fixed_anchors.items()
        if not actions.issubset(action_sets[intent])
    }
    if violations:
        raise ValueError(f"Fixed anchor assignments were not preserved: {violations}")
    if not action_sets["unknown"]:
        raise ValueError("unknown must be non-empty in constrained7 mode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--mode",
        choices=("open", "constrained7"),
        default="open",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_objects = load_action_objects(args.csv)
    expected_actions = sorted(action_objects)
    if len(expected_actions) != 13:
        raise ValueError(f"Expected 13 normalized actions, found {len(expected_actions)}")

    prompt = (
        build_constrained7_prompt(action_objects)
        if args.mode == "constrained7"
        else build_open_prompt(action_objects)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(prompt)
    (args.output_dir / "action_objects.json").write_text(
        json.dumps(action_objects, ensure_ascii=False, indent=2) + "\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0},
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer([chat_text], return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = generated[0][encoded["input_ids"].shape[-1] :]
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    (args.output_dir / "raw_output.txt").write_text(raw_output + "\n")

    result = extract_json(raw_output)
    validate_result(result, expected_actions)
    if args.mode == "constrained7":
        validate_constrained7_result(result)
    (args.output_dir / "taxonomy.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )

    metadata = {
        "schema_version": "prox-s-action-to-intent-qwen3-14b-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "quantization": "nf4_4bit_double_quant",
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
        },
        "source_csv": str(args.csv),
        "source_csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "normalized_action_count": len(expected_actions),
        "normalized_actions": expected_actions,
        "taxonomy_mode": args.mode,
        "category_count": result["category_count"],
        "validation": "passed_all_actions_exactly_once",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
