from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

from object_selector.common.motivation_vocab import MOTIVATION_FIELDS, MotivationVocab
from object_selector.common.reasoning_factors import ReasoningFactorSpec
from object_selector.common.schemas import normalize_gt_instance, teacher_output_json_schema
from object_selector.teacher.llm_reasoning_adapter import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEACHER_MODEL,
    OpenAIReasoningAdapter,
)
from object_selector.teacher.prompt_builder import TEACHER_INSTRUCTIONS, build_teacher_prompt
from object_selector.teacher.teacher_output_builder import TeacherOutputBuilder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_COMBINED_CSV = REPO_ROOT / "dataset" / "combined_dataset.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "object_selector" / "teacher_targets" / "gpt-5.4_structured_v1"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
TARGET_SCHEMA_VERSION = "teacher_structured_target_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured LLM Teacher targets from scene graph jsonl files."
    )
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--combined-csv", type=Path, default=DEFAULT_COMBINED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--backend", choices=("openai", "mock"), default="openai")
    parser.add_argument("--model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--reasoning-effort", type=str, default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sample-id", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--motivation-source",
        choices=("metadata", "llm"),
        default="metadata",
        help="Use combined_dataset motivation labels by default for stable distillation targets.",
    )
    parser.add_argument(
        "--include-gt-in-prompt",
        action="store_true",
        help="Oracle/debug mode only. Defaults off so the LLM reasons over candidates instead of seeing the answer id.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompt/schema debug files for the first selected sample, without calling the LLM.",
    )
    return parser.parse_args()


def normalize_number_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def sample_id_from_row(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("scene", "")).strip(),
            str(row.get("sequence_path", "")).strip(),
            normalize_number_text(row.get("start_frame", "")),
            normalize_number_text(row.get("end_frame", "")),
        ]
    )


def load_combined_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing combined dataset csv: {path}")
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("source", "data")).strip() not in {"", "data"}:
                continue
            if not str(row.get("sequence_path", "")).strip():
                continue
            sample_id = sample_id_from_row(row)
            cleaned = {key: str(value).strip() for key, value in row.items() if key is not None}
            cleaned["sample_id"] = sample_id
            cleaned["gt_object_id"] = normalize_gt_instance(cleaned.get("GT_instance", ""))
            # The source dataset has one duplicated row with identical labels; keep the first copy.
            metadata.setdefault(sample_id, cleaned)
    return metadata


def iter_scene_graphs(scene_graph_root: Path, split: str) -> list[dict[str, Any]]:
    path = scene_graph_root / split / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene graph jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(payload.get("sample_id", ""))
            if sample_id:
                ids.add(sample_id)
    return ids


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def selected_candidate_by_heuristic(scene_graph: Mapping[str, Any], metadata: Mapping[str, str]) -> str:
    gt_object_id = metadata.get("gt_object_id", "")
    candidate_ids = {str(obj.get("object_id", "")) for obj in scene_graph.get("objects", [])}
    if gt_object_id in candidate_ids:
        return gt_object_id

    target_name = (metadata.get("object_norm") or metadata.get("object") or "").strip().lower()
    best_object = ""
    best_score = -1.0e9
    for obj in scene_graph.get("objects", []):
        label = str(obj.get("label", "")).lower()
        metrics = obj.get("metrics", {})
        function_score = 1.0 if target_name and target_name in label else 0.0
        distance = float(metrics.get("distance") or 10.0)
        orientation = float(metrics.get("root_torso_alignment") or 0.0)
        score = 3.0 * function_score + orientation - 0.1 * distance
        if score > best_score:
            best_score = score
            best_object = str(obj.get("object_id", ""))
    if not best_object:
        raise ValueError("Cannot select a heuristic object from an empty candidate list")
    return best_object


def mock_llm_output(scene_graph: Mapping[str, Any], metadata: Mapping[str, str]) -> dict[str, Any]:
    selected_object = selected_candidate_by_heuristic(scene_graph, metadata)
    candidate_ids = [str(obj["object_id"]) for obj in scene_graph.get("objects", [])]
    weights: dict[str, float] = {}
    for obj in scene_graph.get("objects", []):
        metrics = obj.get("metrics", {})
        object_id = str(obj["object_id"])
        if object_id == selected_object:
            weights[object_id] = 8.0
            continue
        distance = float(metrics.get("distance") or 10.0)
        orient = max(float(metrics.get("root_torso_alignment") or 0.0), 0.0)
        weights[object_id] = max(0.05, math.exp(-0.35 * distance) * (0.5 + orient))
    total = sum(weights.values())
    posterior = {candidate_id: weights[candidate_id] / total for candidate_id in candidate_ids}
    selected = next(obj for obj in scene_graph.get("objects", []) if str(obj["object_id"]) == selected_object)
    selected_metrics = selected.get("metrics", {})
    label = str(selected.get("label", "object"))
    target_name = (metadata.get("object_norm") or metadata.get("object") or "").strip().lower()
    function_match = 1.0 if target_name and target_name in label.lower() else 0.65
    return {
        "selected_object": selected_object,
        "object_posterior": posterior,
        "motivation_code": {field: metadata.get(field, "") for field in MOTIVATION_FIELDS},
        "reasoning_factors": {
            "function_match": min(max(function_match, 0.0), 1.0),
            "posture_match": 0.75,
            "orientation_match": min(max((float(selected_metrics.get("root_torso_alignment") or 0.0) + 1.0) / 2.0, 0.0), 1.0),
            "distance_preference": min(max(math.exp(-0.3 * float(selected_metrics.get("distance") or 10.0)), 0.0), 1.0),
            "interaction_feasibility": 0.8,
        },
        "rationale_short": f"{label} best matches the privileged want and available geometry.",
    }


def build_target_record(
    *,
    scene_graph: Mapping[str, Any],
    metadata: Mapping[str, str],
    raw_output: Mapping[str, Any],
    builder: TeacherOutputBuilder,
    model_info: Mapping[str, Any],
) -> dict[str, Any]:
    teacher_output, warnings = builder.build(
        raw_output=raw_output,
        scene_graph=scene_graph,
        metadata=metadata,
    )
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "split": scene_graph.get("split"),
        "sample_index": scene_graph.get("sample_index"),
        "sample_id": scene_graph.get("sample_id"),
        "scene_name": scene_graph.get("scene_name"),
        "sequence_name": scene_graph.get("sequence_name"),
        "start_frame": scene_graph.get("start_frame"),
        "end_frame": scene_graph.get("end_frame"),
        "teacher_model": dict(model_info),
        "metadata": {
            "action": metadata.get("action", ""),
            "object": metadata.get("object", ""),
            "action_norm": metadata.get("action_norm", ""),
            "object_norm": metadata.get("object_norm", ""),
            "want_text": metadata.get("want_text", ""),
            "want_key": metadata.get("want_key", ""),
            "mapping_confidence": metadata.get("mapping_confidence", ""),
            "gt_object_id": metadata.get("gt_object_id", ""),
        },
        "candidate_ids": [str(obj["object_id"]) for obj in scene_graph.get("objects", [])],
        "teacher_output": teacher_output.to_dict(),
        "validation_warnings": warnings,
    }


def process_split(
    *,
    split: str,
    args: argparse.Namespace,
    metadata_by_sample: Mapping[str, dict[str, str]],
    vocab: MotivationVocab,
    factor_spec: ReasoningFactorSpec,
) -> dict[str, Any]:
    output_split_dir = args.output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_split_dir / "targets.jsonl"
    error_path = output_split_dir / "errors.jsonl"
    debug_dir = output_split_dir / "debug"

    skip_ids = existing_sample_ids(target_path) if args.resume else set()
    scene_graphs = iter_scene_graphs(args.scene_graph_root, split)
    builder = TeacherOutputBuilder(
        vocab=vocab,
        factor_spec=factor_spec,
        motivation_source=args.motivation_source,
    )
    adapter = None
    if args.backend == "openai" and not args.dry_run:
        adapter = OpenAIReasoningAdapter(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            env_file=args.env_file,
            max_output_tokens=args.max_output_tokens,
        )

    written = 0
    skipped = 0
    errors = 0
    selected_sample_ids: list[str] = []

    def write_progress_summary() -> None:
        write_json(
            output_split_dir / "summary.json",
            {
                "schema_version": TARGET_SCHEMA_VERSION,
                "split": split,
                "backend": args.backend,
                "model": args.model if args.backend == "openai" else "heuristic_mock",
                "reasoning_effort": args.reasoning_effort if args.backend == "openai" else "none",
                "target_path": str(target_path),
                "error_path": str(error_path),
                "num_written": written,
                "num_skipped": skipped,
                "num_errors": errors,
                "selected_sample_ids": selected_sample_ids,
                "dry_run": bool(args.dry_run),
            },
        )

    target_mode = "a" if args.resume else "w"
    with target_path.open(target_mode, encoding="utf-8") as target_file, error_path.open(
        "a" if args.resume else "w",
        encoding="utf-8",
    ) as error_file:
        for scene_graph in scene_graphs:
            sample_id = str(scene_graph.get("sample_id", ""))
            if args.sample_id and sample_id != args.sample_id:
                continue
            if sample_id in skip_ids:
                skipped += 1
                if skipped % 10 == 0:
                    print(f"[{split}] skipped existing {skipped} samples", flush=True)
                continue
            if args.limit >= 0 and written >= args.limit:
                break
            metadata = metadata_by_sample.get(sample_id)
            if metadata is None:
                errors += 1
                error_file.write(json.dumps({"sample_id": sample_id, "error": "missing metadata"}) + "\n")
                write_progress_summary()
                if args.stop_on_error:
                    raise KeyError(f"Missing metadata for sample_id={sample_id}")
                continue

            candidate_ids = [str(obj["object_id"]) for obj in scene_graph.get("objects", [])]
            schema = teacher_output_json_schema(
                candidate_ids=candidate_ids,
                vocab=vocab,
                factor_spec=factor_spec,
            )
            prompt = build_teacher_prompt(
                scene_graph=scene_graph,
                metadata=metadata,
                vocab=vocab,
                factor_spec=factor_spec,
                include_gt_in_prompt=args.include_gt_in_prompt,
            )

            if args.dry_run:
                debug_dir.mkdir(parents=True, exist_ok=True)
                write_json(debug_dir / f"{sample_id.replace('|', '__')}_prompt.json", json.loads(prompt))
                write_json(debug_dir / f"{sample_id.replace('|', '__')}_schema.json", schema)
                written += 1
                selected_sample_ids.append(sample_id)
                print(f"[{split}] dry-run wrote prompt/schema for {sample_id}", flush=True)
                write_progress_summary()
                break

            try:
                print(
                    f"[{split}] processing sample_index={scene_graph.get('sample_index')} "
                    f"sample_id={sample_id}",
                    flush=True,
                )
                if args.backend == "mock":
                    raw_output = mock_llm_output(scene_graph, metadata)
                    model_info = {
                        "backend": "mock",
                        "model": "heuristic_mock",
                        "reasoning_effort": "none",
                    }
                else:
                    if adapter is None:
                        raise RuntimeError("OpenAI adapter was not initialized")
                    raw_output, model_info = adapter.complete_json(
                        instructions=TEACHER_INSTRUCTIONS,
                        prompt=prompt,
                        json_schema=schema,
                    )
                    model_info = {"backend": "openai", **dict(model_info)}

                record = build_target_record(
                    scene_graph=scene_graph,
                    metadata=metadata,
                    raw_output=raw_output,
                    builder=builder,
                    model_info=model_info,
                )
                target_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                target_file.write("\n")
                target_file.flush()
                written += 1
                selected_sample_ids.append(sample_id)
                print(
                    f"[{split}] wrote {written} target(s); selected="
                    f"{record['teacher_output']['selected_object']} sample_id={sample_id}",
                    flush=True,
                )
                write_progress_summary()
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as exc:  # noqa: BLE001 - keep long offline jobs running by default.
                errors += 1
                error_file.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_file.flush()
                print(
                    f"[{split}] error {errors} on sample_id={sample_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                write_progress_summary()
                if args.stop_on_error:
                    raise

    write_progress_summary()
    summary = json.loads((output_split_dir / "summary.json").read_text(encoding="utf-8"))
    return summary


def main() -> None:
    args = parse_args()
    splits = ["train", "test"] if args.split == "all" else [args.split]
    vocab = MotivationVocab.from_combined_csv(args.combined_csv)
    factor_spec = ReasoningFactorSpec()
    metadata_by_sample = load_combined_metadata(args.combined_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        process_split(
            split=split,
            args=args,
            metadata_by_sample=metadata_by_sample,
            vocab=vocab,
            factor_spec=factor_spec,
        )
        for split in splits
    ]
    write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": TARGET_SCHEMA_VERSION,
            "scene_graph_root": str(args.scene_graph_root),
            "combined_csv": str(args.combined_csv),
            "output_dir": str(args.output_dir),
            "motivation_source": args.motivation_source,
            "include_gt_in_prompt": bool(args.include_gt_in_prompt),
            "splits": summaries,
        },
    )
    for summary in summaries:
        print(
            f"[{summary['split']}] wrote={summary['num_written']} "
            f"skipped={summary['num_skipped']} errors={summary['num_errors']} -> {summary['target_path']}"
        )
    print(f"summary -> {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
