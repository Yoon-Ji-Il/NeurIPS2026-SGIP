#!/usr/bin/env python3
"""Run the exact saved PROX fixed-7 taxonomy prompt with a local HF VLM."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_prox_action_intent_taxonomy_qwen3_14b import (  # noqa: E402
    extract_json,
    validate_constrained7_result,
    validate_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("qwen3_vl", "idefics3"),
    )
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--action-objects", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16"),
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "eager"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--constrained-output",
        action="store_true",
        help=(
            "Restrict decoding to valid full-taxonomy JSON candidates. "
            "The semantic prompt is unchanged."
        ),
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_inputs(
    processor: Any,
    prompt: str,
    device: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    try:
        rendered = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        rendered = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    encoded = processor(text=rendered, return_tensors="pt")
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


INTENTS = ("sit_up", "sit_down", "lie", "walk", "touch", "jump", "unknown")
FIXED_ANCHORS = {
    "sit_up": ("sit up", "stand up"),
    "sit_down": ("sit down",),
    "lie": ("lie down", "lie on"),
    "walk": ("move leg", "step up"),
    "touch": ("touch",),
    "jump": ("jump on",),
    "unknown": (),
}
REMAINING_ACTIONS = ("move on", "sit on", "stand on", "step down")


def compact_taxonomy(assignments: dict[str, str]) -> str:
    categories = []
    for intent in INTENTS:
        nonanchors = [
            action
            for action in REMAINING_ACTIONS
            if assignments[action] == intent
        ]
        categories.append(
            {
                "intent": intent,
                "definition": "Fixed PROX-S intent category.",
                "actions": [*FIXED_ANCHORS[intent], *nonanchors],
                "rationale": "Assigned from the supplied action-object evidence.",
            }
        )
    unknown_actions = [
        action for action in REMAINING_ACTIONS if assignments[action] == "unknown"
    ]
    return json.dumps(
        {
            "category_count": 7,
            "categories": categories,
            "ambiguous_assignments": [
                {
                    "action": action,
                    "reason": "The supplied action-object evidence is ambiguous.",
                }
                for action in unknown_actions
            ],
            "summary": "Every input action is assigned exactly once.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def valid_taxonomy_candidates() -> list[str]:
    candidates = []
    for intents in itertools.product(INTENTS, repeat=len(REMAINING_ACTIONS)):
        assignments = dict(zip(REMAINING_ACTIONS, intents))
        if "unknown" not in assignments.values():
            continue
        candidates.append(compact_taxonomy(assignments))
    if len(candidates) != 1105:
        raise AssertionError(f"Expected 1105 valid candidates, got {len(candidates)}")
    return candidates


class CandidateTrie:
    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        eos = tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        if eos is None:
            raise ValueError("tokenizer has no eos_token_id")
        self.prompt_length = prompt_length
        self.eos = int(eos)
        self.paths = [
            tuple(
                tokenizer.encode(candidate, add_special_tokens=False)
                + [self.eos]
            )
            for candidate in valid_taxonomy_candidates()
        ]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Two valid taxonomy candidates tokenize identically")

    def __call__(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        del batch_id
        generated = tuple(
            int(value) for value in input_ids[self.prompt_length :]
        )
        next_tokens = {
            path[len(generated)]
            for path in self.paths
            if len(generated) < len(path)
            and path[: len(generated)] == generated
        }
        return sorted(next_tokens) if next_tokens else [self.eos]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    prompt = args.prompt.read_text()
    action_objects = json.loads(args.action_objects.read_text())
    expected_actions = sorted(action_objects)
    if len(expected_actions) != 13:
        raise ValueError(
            f"Expected 13 normalized actions, found {len(expected_actions)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.prompt, args.output_dir / "prompt.txt")
    shutil.copyfile(
        args.action_objects,
        args.output_dir / "action_objects.json",
    )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
    )
    model_class = {
        "qwen3_vl": Qwen3VLForConditionalGeneration,
        "idefics3": Idefics3ForConditionalGeneration,
    }[args.architecture]
    model = model_class.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.device)
    model.eval()

    inputs = render_inputs(processor, prompt, args.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generation_kwargs: dict[str, Any] = {}
    if args.constrained_output:
        generation_kwargs["prefix_allowed_tokens_fn"] = CandidateTrie(
            tokenizer,
            prompt_tokens,
        )

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=pad_token_id,
            **generation_kwargs,
        )
    new_tokens = generated[0, prompt_tokens:]
    raw_output = processor.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()
    (args.output_dir / "raw_output.txt").write_text(raw_output + "\n")

    validation = "passed_all_actions_exactly_once_and_fixed7"
    try:
        result = extract_json(raw_output)
        validate_result(result, expected_actions)
        validate_constrained7_result(result)
    except Exception as exc:
        validation = f"failed: {type(exc).__name__}: {exc}"
        (args.output_dir / "validation_error.txt").write_text(validation + "\n")
        raise

    (args.output_dir / "taxonomy.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "validation_error.txt").unlink(missing_ok=True)
    metadata = {
        "schema_version": "prox-s-fixed7-taxonomy-hf-v1",
        "created_at_utc": utc_now(),
        "model": args.model_name,
        "model_reference": str(Path(args.model).resolve()),
        "architecture": args.architecture,
        "backend": "huggingface_transformers",
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "decoding": {
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "constrained_output": args.constrained_output,
            "valid_output_candidates": (
                1105 if args.constrained_output else None
            ),
        },
        "input_modality": "text_only_action_and_observed_object_lists",
        "source_csv": str(args.source_csv.resolve()),
        "source_csv_sha256": sha256_path(args.source_csv),
        "prompt_source": str(args.prompt.resolve()),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(new_tokens.shape[0]),
        "normalized_action_count": len(expected_actions),
        "normalized_actions": expected_actions,
        "taxonomy_mode": "constrained7",
        "category_count": result["category_count"],
        "validation": validation,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
