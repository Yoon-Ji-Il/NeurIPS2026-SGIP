#!/usr/bin/env python3
"""Fast, resumable HIK semantic scoring with compact batched Qwen output.

This keeps the same per-sample match/ease rubric as
``run_hik_qwen3_14b_candidate_scores.py``.  It removes the unused generated
reasoning sentence, batches independent prompts, and can bootstrap successful
rows from an earlier sharded run.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.hik_action_geometry import iter_jsonl, stable_seed  # noqa: E402
from object_selector.teacher.qwen_client import LocalQwenSemanticClient  # noqa: E402
from run_hik_qwen3_14b_candidate_scores import (  # noqa: E402
    DEFAULT_ALIGNMENT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    existing_ids,
    extract_json_object,
    load_alignment_ids,
    selected_splits,
    stable_shuffle,
    validate_response,
)


INSTRUCTIONS = """You are the semantic scoring teacher for HIK object selection.
Score EVERY candidate object for the target person's event_action.
For each object_id, assign match and ease as integers from 1 to 5.
match is functional compatibility with the event/action.
ease is how natural that object type is for the observed actions.
Do not use geometry and do not select only one object.
Return strict compact JSON only. Do not explain or add reasoning text.
The semantic_scores object must contain every input object_id exactly once."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-graph-root",
        type=Path,
        default=REPO_ROOT / "runs_hik" / "scene_graphs_h25f50_action_geometry_v1",
    )
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT.with_name(DEFAULT_OUTPUT_ROOT.name + "_fast"),
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit-per-split", type=int, default=-1)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--model-name", default="Qwen/Qwen3-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--bootstrap-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Earlier shard roots whose successful rows are copied into this shard.",
    )
    return parser.parse_args()


def build_compact_prompt(
    graph: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if candidates is None:
        candidates = stable_shuffle(
            str(graph["sample_id"]),
            list(graph.get("objects", [])),
        )
    if not candidates:
        raise ValueError(f"No candidates for {graph['sample_id']}")
    payload = {
        "sample_id": graph["sample_id"],
        "event_action": graph.get("event_action", ""),
        "active_actions_last_observed": graph.get(
            "active_actions_last_observed",
            [],
        ),
        "active_actions_observed": graph.get("active_actions_observed", []),
        "candidate_objects": [
            {
                "object_id": candidate.get("object_id", ""),
                "label": candidate.get("label", ""),
            }
            for candidate in candidates
        ],
        "output_example": {
            "sample_id": graph["sample_id"],
            "semantic_scores": {
                "candidate object_id": {"match": 1, "ease": 1},
            },
        },
    }
    prompt = (
        f"{INSTRUCTIONS}\nINPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return prompt, candidates


class BatchedLocalQwenSemanticClient(LocalQwenSemanticClient):
    def complete_many(self, prompts: list[str]) -> list[str]:
        import torch

        if not prompts:
            return []
        chat_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            chat_texts.append(chat_text)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_width = int(encoded["input_ids"].shape[-1])
        return [
            self.tokenizer.decode(
                output_ids[input_width:],
                skip_special_tokens=True,
            ).strip()
            for output_ids in generated
        ]


def bootstrap_rows(
    *,
    roots: list[Path],
    split: str,
    output_path: Path,
    seen: set[str],
    num_shards: int,
    shard_index: int,
) -> int:
    copied = 0
    if not roots:
        return copied
    with output_path.open("a", encoding="utf-8") as output:
        for root in roots:
            source = root / split / "semantic_scores.jsonl"
            if not source.is_file():
                continue
            for row in iter_jsonl(source):
                sample_id = str(row.get("sample_id", ""))
                if not sample_id or sample_id in seen:
                    continue
                if stable_seed("hik_semantic_shard", sample_id) % num_shards != shard_index:
                    continue
                if str(row.get("event_action", "")) == "walking":
                    raise ValueError(f"Walking bootstrap row found: {sample_id}")
                output.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                seen.add(sample_id)
                copied += 1
        output.flush()
    return copied


def validate_raw(
    raw: str,
    graph: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = extract_json_object(raw)
    raw_scores = payload.get("semantic_scores")
    expected = {
        str(candidate.get("object_id", ""))
        for candidate in candidates
    }
    if isinstance(raw_scores, dict):
        returned = {str(object_id) for object_id in raw_scores}
        missing = sorted(expected - returned)
        extra = sorted(returned - expected)
        remap: dict[str, str] = {}
        for extra_id in extra:
            ranked = sorted(
                (
                    SequenceMatcher(None, extra_id, missing_id).ratio(),
                    missing_id,
                )
                for missing_id in missing
                if missing_id not in remap.values()
            )
            if ranked and ranked[-1][0] >= 0.92:
                remap[extra_id] = ranked[-1][1]
        if remap:
            payload["semantic_scores"] = {
                remap.get(str(object_id), str(object_id)): value
                for object_id, value in raw_scores.items()
            }
    return validate_response(payload, graph, candidates)


def recover_one(
    client: BatchedLocalQwenSemanticClient,
    graph: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_retries: int,
) -> dict[str, Any]:
    # Long candidate lists are the only observed source of omissions.  Score
    # their short chunks as one batch instead of retrying the same long output.
    # This retains Qwen as the scorer while avoiding repeated token-limit hits.
    chunks = [
        candidates[start : start + 6]
        for start in range(0, len(candidates), 6)
    ]
    chunk_prompts = [build_compact_prompt(graph, chunk)[0] for chunk in chunks]
    try:
        chunk_outputs = client.complete_many(chunk_prompts)
    except Exception:  # noqa: BLE001
        chunk_outputs = [""] * len(chunks)

    last_error: Exception | None = None
    recovered: list[dict[str, Any]] = []
    for chunk, initial_raw in zip(chunks, chunk_outputs, strict=True):
        chunk_prompt, _ = build_compact_prompt(graph, chunk)
        chunk_result = None
        raw_attempts = [initial_raw]
        for attempt in range(max(max_retries, 1)):
            try:
                raw = (
                    raw_attempts[attempt]
                    if attempt < len(raw_attempts)
                    else client.complete(chunk_prompt)
                )
                chunk_result = validate_raw(
                    raw,
                    graph,
                    chunk,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if chunk_result is None:
            raise RuntimeError(
                f"Qwen chunk recovery failed for {graph['sample_id']}: {last_error}"
            ) from last_error
        recovered.extend(chunk_result["semantic_scores"])
    combined = {
        "sample_id": graph["sample_id"],
        "semantic_scores": {
            row["object_id"]: {
                "match": row["match"],
                "ease": row["ease"],
            }
            for row in recovered
        },
    }
    return validate_response(combined, graph, candidates)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    client: BatchedLocalQwenSemanticClient | None = None
    processed = failed = bootstrapped = selected_for_shard = 0
    for split in selected_splits(args.split):
        alignment_ids = load_alignment_ids(args.alignment_root, split)
        if args.sample_id:
            alignment_ids = [
                sample_id
                for sample_id in alignment_ids
                if sample_id == args.sample_id
            ]
        if args.limit_per_split >= 0:
            alignment_ids = alignment_ids[: args.limit_per_split]

        graph_lookup = {
            str(graph["sample_id"]): graph
            for graph in iter_jsonl(args.scene_graph_root / split / "samples.jsonl")
        }
        missing_graphs = [
            sample_id
            for sample_id in alignment_ids
            if sample_id not in graph_lookup
        ]
        if missing_graphs:
            raise KeyError(f"Missing HIK scene graphs: {missing_graphs[:5]}")

        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "semantic_scores.jsonl"
        failure_path = output_dir / "failures.jsonl"
        if args.no_resume:
            output_path.write_text("", encoding="utf-8")
            failure_path.write_text("", encoding="utf-8")
        seen = existing_ids(output_path)
        if not args.no_resume:
            bootstrapped += bootstrap_rows(
                roots=args.bootstrap_roots,
                split=split,
                output_path=output_path,
                seen=seen,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )

        pending = [
            sample_id
            for sample_id in alignment_ids
            if stable_seed("hik_semantic_shard", sample_id) % args.num_shards
            == args.shard_index
            and sample_id not in seen
        ]
        selected_for_shard += len(pending) + sum(
            1
            for sample_id in seen
            if sample_id in graph_lookup
            and stable_seed("hik_semantic_shard", sample_id) % args.num_shards
            == args.shard_index
        )

        with (
            output_path.open("a", encoding="utf-8") as output,
            failure_path.open("a", encoding="utf-8") as failures,
        ):
            for batch_start in range(0, len(pending), args.batch_size):
                batch_ids = pending[batch_start : batch_start + args.batch_size]
                graphs = [graph_lookup[sample_id] for sample_id in batch_ids]
                built = [build_compact_prompt(graph) for graph in graphs]
                prompts = [item[0] for item in built]
                candidates_per_sample = [item[1] for item in built]

                if client is None:
                    client = BatchedLocalQwenSemanticClient(
                        model_name=args.model_name,
                        device=args.device,
                        torch_dtype=args.torch_dtype,
                        quantization=args.quantization,
                        max_new_tokens=args.max_new_tokens,
                    )
                # HIK test graphs commonly contain 24 candidates.  A single
                # full response frequently reaches the generation limit, so
                # route long lists directly to compact six-candidate batches.
                if any(len(candidates) > 12 for candidates in candidates_per_sample):
                    raw_outputs = [""] * len(prompts)
                else:
                    try:
                        raw_outputs = client.complete_many(prompts)
                    except Exception:  # noqa: BLE001
                        raw_outputs = [""] * len(prompts)

                for graph, candidates, raw in zip(
                    graphs,
                    candidates_per_sample,
                    raw_outputs,
                    strict=True,
                ):
                    result = None
                    last_error: Exception | None = None
                    try:
                        result = validate_raw(raw, graph, candidates)
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        try:
                            result = recover_one(
                                client,
                                graph,
                                candidates,
                                args.max_retries,
                            )
                        except Exception as recovery_exc:  # noqa: BLE001
                            last_error = recovery_exc

                    if result is None:
                        failed += 1
                        failures.write(
                            json.dumps(
                                {
                                    "sample_id": graph["sample_id"],
                                    "split": split,
                                    "error_type": type(last_error).__name__,
                                    "error": str(last_error),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        failures.flush()
                        print(
                            f"[failed] {graph['sample_id']}: {last_error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    output.write(
                        json.dumps(result, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
                    output.flush()
                    processed += 1
                    print(
                        f"[{split}] semantic scores: {graph['sample_id']}",
                        flush=True,
                    )

    summary = {
        "backend": "qwen_compact_batch",
        "model_name": args.model_name,
        "scene_graph_root": str(args.scene_graph_root.resolve()),
        "alignment_root": str(args.alignment_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "walking_excluded_by_alignment_manifest": True,
        "selected_for_shard": selected_for_shard,
        "bootstrapped_from_previous_run": bootstrapped,
        "processed_this_run": processed,
        "failed_this_run": failed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume": not args.no_resume,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failed:
        raise RuntimeError(f"Qwen semantic scoring failed for {failed} samples")


if __name__ == "__main__":
    main()
