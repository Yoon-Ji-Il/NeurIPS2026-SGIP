#!/usr/bin/env python3
"""Generate same-prompt GIMO intent predictions with a Hugging Face VLM.

The semantic prompt and dataset preparation functions are imported directly
from run_remote_intent_category_classifier.py. Generation is constrained to
one of 16 valid JSON objects, so formatting failures cannot change the intent
agreement metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
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

from object_selector.common.want_taxonomy import CANONICAL_WANTS  # noqa: E402
from scripts.run_remote_intent_category_classifier import (  # noqa: E402
    DEFAULT_DATASET_CSV,
    PROMPT_VERSION,
    action_object_text,
    append_jsonl,
    build_prompt,
    load_checkpoint,
    parse_response,
    read_csv,
    response_schema,
    sha256,
    utc_now,
    validate_dataset,
    validate_or_write_manifest,
)


DEFAULT_MODEL = "pretrained_models/Idefics3-8B-Llama3"
MODEL_DISPLAY_NAME = "HuggingFaceM4/Idefics3-8B-Llama3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default=MODEL_DISPLAY_NAME)
    parser.add_argument(
        "--architecture",
        choices=("idefics3", "qwen3_vl"),
        default="idefics3",
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def compact_candidate(intent: str) -> str:
    return json.dumps(
        {
            "selected_intent": intent,
            "confidence": "medium",
            "reason": "Selected from the fixed taxonomy.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class CandidateTrie:
    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer has no eos_token_id")
        self.prompt_length = prompt_length
        self.paths = [
            tuple(
                tokenizer.encode(
                    compact_candidate(intent),
                    add_special_tokens=False,
                )
                + [int(eos)]
            )
            for intent in CANONICAL_WANTS
        ]
        if len(set(self.paths)) != len(CANONICAL_WANTS):
            raise ValueError("two constrained JSON candidates tokenize identically")
        self.eos = int(eos)

    def __call__(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        del batch_id
        generated = tuple(int(value) for value in input_ids[self.prompt_length :])
        next_tokens = {
            path[len(generated)]
            for path in self.paths
            if len(generated) < len(path)
            and path[: len(generated)] == generated
        }
        return sorted(next_tokens) if next_tokens else [self.eos]


def make_inputs(
    processor: Any, prompt: str, device: str
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=rendered, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def generate_prediction(
    model: Any,
    processor: Any,
    action: str,
    object_name: str,
    device: str,
    max_new_tokens: int,
) -> tuple[dict[str, str], str]:
    prompt = build_prompt(action, object_name)
    inputs = make_inputs(processor, prompt, device)
    prompt_length = int(inputs["input_ids"].shape[1])
    trie = CandidateTrie(processor.tokenizer, prompt_length)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=trie,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_tokens = generated[0, prompt_length:]
    raw_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_response(raw_text), raw_text


def main() -> None:
    args = parse_args()
    if args.max_pairs == 0 or args.max_pairs < -1:
        raise ValueError("--max-pairs must be -1 or a positive integer")
    torch.manual_seed(args.seed)

    raw_rows = read_csv(args.dataset_csv)
    prepared_rows, pair_map = validate_dataset(raw_rows)
    del prepared_rows
    all_pairs = list(pair_map.values())
    selected_pairs = (
        all_pairs if args.max_pairs < 0 else all_pairs[: args.max_pairs]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now(),
        "prompt_version": PROMPT_VERSION,
        "model_name": args.model_name,
        "backend": "huggingface_transformers",
        "architecture": args.architecture,
        "model_reference": str(Path(args.model).resolve()),
        "dataset_csv": str(args.dataset_csv.resolve()),
        "dataset_csv_sha256": sha256(args.dataset_csv),
        "dataset_rows": len(raw_rows),
        "all_unique_pairs": len(all_pairs),
        "selected_unique_pairs": len(selected_pairs),
        "canonical_intents": list(CANONICAL_WANTS),
        "response_schema": response_schema(),
        "temperature": 0,
        "seed": args.seed,
        "think": False,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "constrained_output": True,
        "constrained_candidates": {
            intent: compact_candidate(intent) for intent in CANONICAL_WANTS
        },
        "semantic_prompt_is_imported_from_ollama_classifier": True,
    }
    validate_or_write_manifest(args.output_dir / "manifest.json", manifest)

    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} exists; pass --resume or use a new output directory"
        )
    predictions = load_checkpoint(predictions_path) if args.resume else {}
    pending = [
        pair for pair in selected_pairs if pair["pair_key"] not in predictions
    ]
    print(
        json.dumps(
            {
                "model": args.model_name,
                "model_reference": args.model,
                "dataset_rows": len(raw_rows),
                "unique_pairs": len(selected_pairs),
                "completed": len(selected_pairs) - len(pending),
                "pending": len(pending),
                "device": args.device,
                "dtype": args.dtype,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not pending:
        return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model_class = {
        "idefics3": Idefics3ForConditionalGeneration,
        "qwen3_vl": Qwen3VLForConditionalGeneration,
    }[args.architecture]
    model = model_class.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.device)
    model.eval()

    try:
        already_complete = len(selected_pairs) - len(pending)
        for index, pair in enumerate(pending, start=1):
            ordinal = already_complete + index
            parsed, raw_text = generate_prediction(
                model,
                processor,
                pair["action"],
                pair["object"],
                args.device,
                args.max_new_tokens,
            )
            record = {
                "completed_at": utc_now(),
                "prompt_version": PROMPT_VERSION,
                "model_name": args.model_name,
                "backend": "huggingface_transformers",
                "pair_key": pair["pair_key"],
                "action": pair["action"],
                "object": pair["object"],
                "action_object_input": action_object_text(
                    pair["action"], pair["object"]
                ),
                "selected_intent": parsed["selected_intent"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "raw_response": raw_text,
                "sample_attempt": 1,
            }
            append_jsonl(predictions_path, record)
            predictions[pair["pair_key"]] = record
            print(
                f"[{ordinal}/{len(selected_pairs)}] "
                f"{pair['action']} + {pair['object']} -> "
                f"{parsed['selected_intent']}",
                flush=True,
            )
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
