#!/usr/bin/env python3
"""Classify GIMO Action+Object pairs into the fixed 16-intent taxonomy.

One deterministic Ollama request is made per unique Action+Object pair. The
prediction is then mapped back to every row in ``dataset_3_with_want.csv`` and
evaluated against its ``want`` label. The model can only emit one of the
canonical GIMO intent labels because the Ollama JSON schema uses an enum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.common.want_taxonomy import (  # noqa: E402
    CANONICAL_WANTS,
    CANONICAL_WANT_SET,
)
from object_selector.teacher.ollama_client import (  # noqa: E402
    RemoteOllamaStructuredClient,
)


DEFAULT_DATASET_CSV = REPO_ROOT / "dataset" / "dataset_3_with_want.csv"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
PROMPT_VERSION = "gimo_fixed_16_intent_zero_shot_v1"

INTENT_DEFINITIONS: dict[str, str] = {
    "clean": "clean, wipe, wash, or otherwise make something clean",
    "cook": "prepare food or handle cookware specifically for food preparation",
    "dress": "dress, change clothing, or handle footwear/clothing for wearing",
    "drink": "obtain, consume, or handle a beverage for drinking",
    "eat": "obtain, consume, or handle food or eating utensils for eating",
    "fix": "repair, attach, stitch, or restore something",
    "light": "turn on, control, or use a light source for illumination",
    "open": "open or close an access barrier such as a door or window",
    "read": "read, study, or obtain readable material for reading",
    "sit": "sit, settle, or use seating/support for a seated posture",
    "sleep": "lie down, sleep, wake, or use a bed for sleeping/rest",
    "take": "retrieve, pick up, carry, or access an ordinary stored item",
    "throw": "discard waste, put something in trash, or handle a waste container",
    "unknown": "the goal cannot be reliably assigned to any other allowed intent",
    "use": "general device, tool, exercise, transport, or object use with no more specific intent",
    "write": "write, draw, take notes, or obtain writing material for writing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify GIMO Action+Object pairs into the fixed 16 intent labels "
            "using a remote Ollama model."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=-1,
        help="Classify only the first N unique pairs for a smoke test; -1 uses all.",
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the initial Ollama /api/tags connectivity and model check.",
    )
    parser.add_argument(
        "--unload-after-run",
        action="store_true",
        help="Unload the model from VRAM at the end without deleting model files.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully even when some pair predictions are missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print the first prompt without API calls.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key).lstrip("\ufeff").strip(): (
            "" if value is None else str(value).strip()
        )
        for key, value in row.items()
        if key is not None and str(key).lstrip("\ufeff").strip()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def pair_key(action: str, object_name: str) -> str:
    return json.dumps(
        [normalize_label(action), normalize_label(object_name)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def action_object_text(action: str, object_name: str) -> str:
    return " ".join(
        part for part in (action.strip(), object_name.strip()) if part
    )


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_intent": {
                "type": "string",
                "enum": list(CANONICAL_WANTS),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "reason": {
                "type": "string",
                "description": "One short sentence justifying the selected category.",
            },
        },
        "required": ["selected_intent", "confidence", "reason"],
    }


def build_prompt(action: str, object_name: str) -> str:
    taxonomy_lines = "\n".join(
        f"- {intent}: {INTENT_DEFINITIONS[intent]}"
        for intent in CANONICAL_WANTS
    )
    return (
        "You are a zero-shot classifier for the fixed GIMO human-intent taxonomy.\n"
        "Given only an Action and an Object, select exactly one allowed intent.\n\n"
        "Allowed intent labels and definitions:\n"
        f"{taxonomy_lines}\n\n"
        "Decision rules:\n"
        "- Use both Action and Object; infer the likely human goal, not merely the verb.\n"
        "- Prefer a specific category over take or use when the object clearly implies it.\n"
        "- Use take for generic retrieval/access/carrying when no specific purpose is clear.\n"
        "- Use use for general operation/activity when no more specific category applies.\n"
        "- Use unknown only when no other category can be justified reliably.\n"
        "- selected_intent must be exactly one allowed lowercase label.\n"
        "- Do not invent a new category.\n"
        "- Return only JSON matching the supplied schema.\n\n"
        f"Action: {json.dumps(action, ensure_ascii=False)}\n"
        f"Object: {json.dumps(object_name, ensure_ascii=False)}"
    )


def validate_dataset(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not rows:
        raise ValueError("dataset contains no rows")
    required = {
        "sequence_path",
        "scene",
        "start_frame",
        "end_frame",
        "training",
        "action",
        "object",
        "want",
    }
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise ValueError(
            "dataset is missing columns: " + ", ".join(missing_columns)
        )

    prepared_rows: list[dict[str, Any]] = []
    pair_map: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        action = row["action"].strip()
        object_name = row["object"].strip()
        gt_intent = row["want"].strip()
        if not action or not object_name:
            raise ValueError(
                f"dataset row {index + 2} has empty action or object"
            )
        if gt_intent not in CANONICAL_WANT_SET:
            raise ValueError(
                f"dataset row {index + 2} has non-canonical want {gt_intent!r}"
            )
        key = pair_key(action, object_name)
        current = pair_map.get(key)
        if current is None:
            pair_map[key] = {
                "pair_key": key,
                "action": action,
                "object": object_name,
                "gt_intent": gt_intent,
                "count": 1,
                "first_row_index": index,
            }
        else:
            if current["gt_intent"] != gt_intent:
                raise ValueError(
                    "the same normalized Action+Object pair has conflicting GT "
                    f"intents: {action!r} + {object_name!r}: "
                    f"{current['gt_intent']!r} vs {gt_intent!r}"
                )
            current["count"] += 1

        scene = row["scene"].strip()
        sequence = row["sequence_path"].strip()
        start = row["start_frame"].strip()
        end = row["end_frame"].strip()
        sample_id = f"{scene}|{sequence}|{start}|{end}"
        prepared_rows.append(
            {
                "row_index": index,
                "row_id": index + 1,
                "split": "train"
                if normalize_label(row["training"]) in {"1", "1.0", "true"}
                else "test",
                "sample_id": sample_id,
                "sample_uid": f"row:{index:04d}:{sample_id}",
                "video_name": f"{scene}/{sequence}",
                "scene": scene,
                "sequence_path": sequence,
                "start_frame": start,
                "end_frame": end,
                "action": action,
                "object": object_name,
                "action_object_input": action_object_text(action, object_name),
                "gt_intent": gt_intent,
                "pair_key": key,
            }
        )
    return prepared_rows, pair_map


def get_available_models(base_url: str, timeout: float) -> list[str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/tags", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 20.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url.rstrip('/')}: {exc}"
        ) from exc
    return [
        str(item.get("name", "")).strip()
        for item in payload.get("models", [])
        if str(item.get("name", "")).strip()
    ]


def unload_model(base_url: str, model_name: str, timeout: float) -> None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=min(timeout, 60.0)) as response:
        response.read()


def parse_response(text: str) -> dict[str, str]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"response is not valid JSON: {candidate[:300]!r}")
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    selected_intent = str(payload.get("selected_intent", "")).strip().lower()
    confidence = str(payload.get("confidence", "")).strip().lower()
    reason = re.sub(r"\s+", " ", str(payload.get("reason", "")).strip())
    if selected_intent not in CANONICAL_WANT_SET:
        raise ValueError(
            f"selected_intent {selected_intent!r} is outside the GIMO taxonomy"
        )
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence {confidence!r}")
    if not reason:
        raise ValueError("reason is empty")
    if len(reason) > 500:
        reason = reason[:497].rstrip() + "..."
    return {
        "selected_intent": selected_intent,
        "confidence": confidence,
        "reason": reason,
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return predictions
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid checkpoint JSON at {path}:{line_number}"
                ) from exc
            key = str(payload.get("pair_key", "")).strip()
            selected = str(payload.get("selected_intent", "")).strip()
            if key and selected in CANONICAL_WANT_SET:
                predictions[key] = payload
    return predictions


def validate_or_write_manifest(
    path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        identity_fields = [
            "prompt_version",
            "model_name",
            "dataset_csv_sha256",
            "canonical_intents",
            "seed",
            "temperature",
            "think",
            "context_length",
            "max_new_tokens",
        ]
        mismatches = [
            field
            for field in identity_fields
            if existing.get(field) != expected.get(field)
        ]
        if mismatches:
            raise ValueError(
                "output directory belongs to a different run; mismatched "
                f"manifest fields: {', '.join(mismatches)}. "
                "Use a new --output-dir."
            )
        existing["selected_unique_pairs"] = expected["selected_unique_pairs"]
        existing["last_resumed_at"] = utc_now()
        write_json(path, existing)
        return existing
    write_json(path, expected)
    return expected


def call_model(
    client: RemoteOllamaStructuredClient,
    action: str,
    object_name: str,
    sample_retries: int,
) -> tuple[dict[str, str], str, int]:
    last_error: Exception | None = None
    prompt = build_prompt(action, object_name)
    for attempt in range(1, sample_retries + 1):
        try:
            response_text = client.complete_json(
                instructions="Return only the requested structured JSON.",
                prompt=prompt,
                json_schema=response_schema(),
                schema_name="gimo_intent_classification",
            )
            return parse_response(response_text), response_text, attempt
        except Exception as exc:
            last_error = exc
            if attempt < sample_retries:
                time.sleep(min(attempt, 3))
    raise RuntimeError(
        f"classification failed for Action={action!r}, Object={object_name!r}: "
        f"{last_error}"
    ) from last_error


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    rows: list[dict[str, Any]],
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    confusion: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in CANONICAL_WANTS} for gt in CANONICAL_WANTS
    }
    for row in rows:
        confusion[row["gt_intent"]][row["selected_intent"]] += 1

    per_intent: list[dict[str, Any]] = []
    supported_f1: list[float] = []
    all_f1: list[float] = []
    for intent in CANONICAL_WANTS:
        tp = confusion[intent][intent]
        support = sum(confusion[intent].values())
        predicted_count = sum(confusion[gt][intent] for gt in CANONICAL_WANTS)
        precision = safe_div(tp, predicted_count)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        all_f1.append(f1)
        if support:
            supported_f1.append(f1)
        per_intent.append(
            {
                "intent": intent,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    confusion_rows = [
        {
            "gt_intent": gt,
            **{f"pred_{pred}": confusion[gt][pred] for pred in CANONICAL_WANTS},
        }
        for gt in CANONICAL_WANTS
    ]
    correct = sum(bool(row["is_correct"]) for row in rows)
    split_metrics: dict[str, Any] = {}
    for split in ("train", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_correct = sum(bool(row["is_correct"]) for row in split_rows)
        split_metrics[split] = {
            "rows": len(split_rows),
            "correct": split_correct,
            "accuracy": safe_div(split_correct, len(split_rows)),
        }
    metrics = {
        "model_name": model_name,
        "rows": len(rows),
        "correct": correct,
        "accuracy": safe_div(correct, len(rows)),
        "macro_f1_all_16": safe_div(sum(all_f1), len(all_f1)),
        "macro_f1_supported_intents": safe_div(
            sum(supported_f1), len(supported_f1)
        ),
        "split_metrics": split_metrics,
    }
    return metrics, per_intent, confusion_rows


def main() -> None:
    args = parse_args()
    if args.context_length < 512:
        raise ValueError("--context-length must be >= 512")
    if args.max_new_tokens < 32:
        raise ValueError("--max-new-tokens must be >= 32")
    if args.http_retries < 1 or args.sample_retries < 1:
        raise ValueError("retry counts must be >= 1")
    if args.max_pairs == 0 or args.max_pairs < -1:
        raise ValueError("--max-pairs must be -1 or a positive integer")
    if set(INTENT_DEFINITIONS) != set(CANONICAL_WANTS):
        raise RuntimeError("intent definitions do not match CANONICAL_WANTS")

    raw_rows = read_csv(args.dataset_csv)
    prepared_rows, pair_map = validate_dataset(raw_rows)
    all_pairs = list(pair_map.values())
    selected_pairs = (
        all_pairs if args.max_pairs < 0 else all_pairs[: args.max_pairs]
    )
    selected_keys = {pair["pair_key"] for pair in selected_pairs}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "created_at": utc_now(),
        "prompt_version": PROMPT_VERSION,
        "model_name": args.model_name,
        "ollama_base_url": args.ollama_base_url.rstrip("/"),
        "dataset_csv": str(args.dataset_csv.resolve()),
        "dataset_csv_sha256": sha256(args.dataset_csv),
        "dataset_rows": len(prepared_rows),
        "all_unique_pairs": len(all_pairs),
        "selected_unique_pairs": len(selected_pairs),
        "canonical_intents": list(CANONICAL_WANTS),
        "intent_definitions": INTENT_DEFINITIONS,
        "temperature": 0,
        "seed": args.seed,
        "think": False,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
    }
    validate_or_write_manifest(manifest_path, manifest)

    print(
        json.dumps(
            {
                "model": args.model_name,
                "dataset_rows": len(prepared_rows),
                "all_unique_pairs": len(all_pairs),
                "selected_unique_pairs": len(selected_pairs),
                "canonical_intents": list(CANONICAL_WANTS),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        if selected_pairs:
            first = selected_pairs[0]
            print(
                "\nFIRST PROMPT\n"
                + build_prompt(first["action"], first["object"])
            )
            print(
                "\nJSON SCHEMA\n"
                + json.dumps(response_schema(), ensure_ascii=False, indent=2)
            )
        return

    if not args.skip_server_check:
        available_models = get_available_models(args.ollama_base_url, args.timeout)
        if args.model_name not in available_models:
            raise RuntimeError(
                f"model {args.model_name!r} is not available at "
                f"{args.ollama_base_url.rstrip('/')}. Available: "
                + ", ".join(available_models)
            )
        print(
            f"Ollama reachable; model available: {args.model_name}",
            flush=True,
        )

    client = RemoteOllamaStructuredClient(
        base_url=args.ollama_base_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        timeout=args.timeout,
        keep_alive=args.keep_alive,
        retry_attempts=args.http_retries,
        seed=args.seed,
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    predictions = load_checkpoint(predictions_path)
    missing_pairs = [
        pair for pair in selected_pairs if pair["pair_key"] not in predictions
    ]
    print(
        f"Checkpoint: {len(selected_pairs) - len(missing_pairs)}/"
        f"{len(selected_pairs)} unique pairs complete",
        flush=True,
    )

    try:
        for missing_index, pair in enumerate(missing_pairs, start=1):
            complete_before = len(selected_pairs) - len(missing_pairs)
            ordinal = complete_before + missing_index
            print(
                f"[{ordinal}/{len(selected_pairs)}] {args.model_name}: "
                f"{pair['action']} + {pair['object']}",
                flush=True,
            )
            try:
                parsed, response_text, attempt = call_model(
                    client,
                    pair["action"],
                    pair["object"],
                    args.sample_retries,
                )
                record = {
                    "completed_at": utc_now(),
                    "prompt_version": PROMPT_VERSION,
                    "model_name": args.model_name,
                    "pair_key": pair["pair_key"],
                    "action": pair["action"],
                    "object": pair["object"],
                    "action_object_input": action_object_text(
                        pair["action"], pair["object"]
                    ),
                    "selected_intent": parsed["selected_intent"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "raw_response": response_text,
                    "sample_attempt": attempt,
                }
                append_jsonl(predictions_path, record)
                predictions[pair["pair_key"]] = record
            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "failed_at": utc_now(),
                        "model_name": args.model_name,
                        "pair_key": pair["pair_key"],
                        "action": pair["action"],
                        "object": pair["object"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(
                    f"ERROR for {pair['action']!r} + {pair['object']!r}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if "Ollama request failed" in str(exc):
                    print(
                        "Stopping after persistent connection failure; "
                        "checkpoints are preserved.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
            if args.request_delay > 0:
                time.sleep(args.request_delay)
    finally:
        if args.unload_after_run:
            try:
                unload_model(
                    args.ollama_base_url, args.model_name, args.timeout
                )
                print(
                    f"Unloaded {args.model_name} from VRAM (model files kept).",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"WARNING: failed to unload {args.model_name}: {exc}",
                    file=sys.stderr,
                )

    completed_keys = selected_keys & set(predictions)
    remaining_keys = selected_keys - set(predictions)

    unique_rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        prediction = predictions.get(pair["pair_key"])
        if prediction is None:
            continue
        selected_intent = prediction["selected_intent"]
        unique_rows.append(
            {
                "pair_index": len(unique_rows),
                "action": pair["action"],
                "object": pair["object"],
                "action_object_input": action_object_text(
                    pair["action"], pair["object"]
                ),
                "row_count": pair["count"],
                "gt_intent": pair["gt_intent"],
                "selected_intent": selected_intent,
                "is_correct": selected_intent == pair["gt_intent"],
                "confidence": prediction["confidence"],
                "reason": prediction["reason"],
                "model_name": args.model_name,
            }
        )
    unique_fields = [
        "pair_index",
        "action",
        "object",
        "action_object_input",
        "row_count",
        "gt_intent",
        "selected_intent",
        "is_correct",
        "confidence",
        "reason",
        "model_name",
    ]
    write_csv(
        args.output_dir / "unique_pair_predictions.csv",
        unique_rows,
        unique_fields,
    )

    row_predictions: list[dict[str, Any]] = []
    loader_rows: list[dict[str, Any]] = []
    for row in prepared_rows:
        if row["pair_key"] not in selected_keys:
            continue
        prediction = predictions.get(row["pair_key"])
        if prediction is None:
            continue
        selected_intent = prediction["selected_intent"]
        output_row = {
            **{key: value for key, value in row.items() if key != "pair_key"},
            "gt_intent": row["gt_intent"],
            "selected_intent": selected_intent,
            "is_correct": selected_intent == row["gt_intent"],
            "confidence": prediction["confidence"],
            "reason": prediction["reason"],
            "model_name": args.model_name,
        }
        row_predictions.append(output_row)
        loader_rows.append(
            {
                "video_name": row["video_name"],
                "high_level_intent": selected_intent,
                "gt_intent": row["gt_intent"],
                "is_correct": selected_intent == row["gt_intent"],
                "action": row["action"],
                "object": row["object"],
                "model_name": args.model_name,
            }
        )
    row_fields = [
        "row_index",
        "row_id",
        "split",
        "sample_id",
        "sample_uid",
        "video_name",
        "scene",
        "sequence_path",
        "start_frame",
        "end_frame",
        "action",
        "object",
        "action_object_input",
        "gt_intent",
        "selected_intent",
        "is_correct",
        "confidence",
        "reason",
        "model_name",
    ]
    write_csv(
        args.output_dir / "intent_predictions.csv",
        row_predictions,
        row_fields,
    )
    write_csv(
        args.output_dir / "video_name_intent_category.csv",
        loader_rows,
        [
            "video_name",
            "high_level_intent",
            "gt_intent",
            "is_correct",
            "action",
            "object",
            "model_name",
        ],
    )

    metrics, per_intent, confusion_rows = evaluate(
        row_predictions, args.model_name
    )
    unique_correct = sum(bool(row["is_correct"]) for row in unique_rows)
    metrics.update(
        {
            "status": "complete" if not remaining_keys else "partial",
            "completed_at": utc_now(),
            "prompt_version": PROMPT_VERSION,
            "dataset_rows": len(prepared_rows),
            "all_unique_pairs": len(all_pairs),
            "selected_unique_pairs": len(selected_pairs),
            "completed_unique_pairs": len(completed_keys),
            "missing_unique_pairs": len(remaining_keys),
            "output_rows": len(row_predictions),
            "unique_pair_correct": unique_correct,
            "unique_pair_accuracy": safe_div(
                unique_correct, len(unique_rows)
            ),
            "canonical_intents": list(CANONICAL_WANTS),
            "gt_is_dataset_want": True,
            "files": {
                "row_predictions_csv": str(
                    args.output_dir / "intent_predictions.csv"
                ),
                "unique_pair_predictions_csv": str(
                    args.output_dir / "unique_pair_predictions.csv"
                ),
                "loader_csv": str(
                    args.output_dir / "video_name_intent_category.csv"
                ),
                "per_intent_metrics_csv": str(
                    args.output_dir / "per_intent_metrics.csv"
                ),
                "confusion_matrix_csv": str(
                    args.output_dir / "confusion_matrix.csv"
                ),
                "checkpoint_jsonl": str(predictions_path),
                "errors_jsonl": str(errors_path),
                "manifest_json": str(manifest_path),
            },
        }
    )
    write_csv(
        args.output_dir / "per_intent_metrics.csv",
        per_intent,
        [
            "intent",
            "support",
            "predicted_count",
            "true_positive",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(
        args.output_dir / "confusion_matrix.csv",
        confusion_rows,
        ["gt_intent"] + [f"pred_{intent}" for intent in CANONICAL_WANTS],
    )
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    if remaining_keys and not args.allow_partial:
        raise RuntimeError(
            f"{len(remaining_keys)} unique pairs are incomplete. "
            f"Rerun the same command to resume; see {errors_path}."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
