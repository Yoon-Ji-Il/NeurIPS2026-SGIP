#!/usr/bin/env python3
"""Compare PROX fixed-7 intent taxonomies at action and segment levels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ACTION_ALIASES = {"wrong_sit on": "sit on"}
INTENTS = ("sit_up", "sit_down", "lie", "walk", "touch", "jump", "unknown")
UNFIXED_ACTIONS = ("move on", "sit on", "stand on", "step down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-name", default="Qwen3-VL-8B_assumed_baseline")
    parser.add_argument(
        "--candidate",
        nargs=2,
        metavar=("NAME", "TAXONOMY_JSON"),
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_taxonomy(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    taxonomy = json.loads(path.read_text())
    mapping: dict[str, str] = {}
    categories = taxonomy.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"{path}: categories must be a list")
    for category in categories:
        intent = category["intent"]
        if intent not in INTENTS:
            raise ValueError(f"{path}: unexpected intent {intent!r}")
        for action in category["actions"]:
            if action in mapping:
                raise ValueError(f"{path}: duplicate action {action!r}")
            mapping[action] = intent
    return mapping, taxonomy


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_map, baseline_taxonomy = load_taxonomy(args.baseline)
    candidate_maps: dict[str, dict[str, str]] = {}
    candidate_taxonomies: dict[str, dict[str, Any]] = {}
    for name, path_text in args.candidate:
        path = Path(path_text)
        candidate_maps[name], candidate_taxonomies[name] = load_taxonomy(path)
    models = [args.baseline_name, *candidate_maps]

    with args.source_csv.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    normalized_actions = sorted(
        {
            ACTION_ALIASES.get(
                row["target_action_prefix"].strip().lower(),
                row["target_action_prefix"].strip().lower(),
            )
            for row in source_rows
        }
    )
    expected = set(normalized_actions)
    for name, mapping in [
        (args.baseline_name, baseline_map),
        *candidate_maps.items(),
    ]:
        if set(mapping) != expected:
            raise ValueError(
                f"{name}: action mismatch; "
                f"missing={sorted(expected - set(mapping))}, "
                f"extra={sorted(set(mapping) - expected)}"
            )

    action_counts = Counter(
        ACTION_ALIASES.get(
            row["target_action_prefix"].strip().lower(),
            row["target_action_prefix"].strip().lower(),
        )
        for row in source_rows
    )
    action_rows: list[dict[str, Any]] = []
    for action in normalized_actions:
        row: dict[str, Any] = {
            "action": action,
            "segment_count": action_counts[action],
            args.baseline_name: baseline_map[action],
        }
        for name, mapping in candidate_maps.items():
            row[name] = mapping[action]
            row[f"{name}_matches_baseline"] = mapping[action] == baseline_map[action]
        action_rows.append(row)
    action_fields = ["action", "segment_count", args.baseline_name]
    for name in candidate_maps:
        action_fields.extend([name, f"{name}_matches_baseline"])
    write_csv(args.output_dir / "action_comparison.csv", action_rows, action_fields)

    segment_rows: list[dict[str, Any]] = []
    for source in source_rows:
        action_raw = source["target_action_prefix"].strip().lower()
        action = ACTION_ALIASES.get(action_raw, action_raw)
        row = {
            "split": source["split"],
            "sample_index": source["sample_index"],
            "sample_id": source["sample_id"],
            "action": action,
            "object": source["target_object_name"].strip().lower(),
            args.baseline_name: baseline_map[action],
        }
        for name, mapping in candidate_maps.items():
            row[name] = mapping[action]
            row[f"{name}_matches_baseline"] = mapping[action] == baseline_map[action]
        segment_rows.append(row)
    segment_fields = [
        "split",
        "sample_index",
        "sample_id",
        "action",
        "object",
        args.baseline_name,
    ]
    for name in candidate_maps:
        segment_fields.extend([name, f"{name}_matches_baseline"])
    write_csv(
        args.output_dir / "segment_comparison.csv",
        segment_rows,
        segment_fields,
    )

    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in segment_rows:
        pair_groups.setdefault((row["action"], row["object"]), []).append(row)
    pair_rows: list[dict[str, Any]] = []
    for (action, object_name), rows in sorted(pair_groups.items()):
        row = {
            "action": action,
            "object": object_name,
            "segment_count": len(rows),
            args.baseline_name: baseline_map[action],
        }
        for name, mapping in candidate_maps.items():
            row[name] = mapping[action]
            row[f"{name}_matches_baseline"] = mapping[action] == baseline_map[action]
        pair_rows.append(row)
    pair_fields = ["action", "object", "segment_count", args.baseline_name]
    for name in candidate_maps:
        pair_fields.extend([name, f"{name}_matches_baseline"])
    write_csv(
        args.output_dir / "action_object_pair_comparison.csv",
        pair_rows,
        pair_fields,
    )

    summary_rows: list[dict[str, Any]] = []
    scopes = [("all", segment_rows)]
    scopes.extend(
        (split, [row for row in segment_rows if row["split"] == split])
        for split in ("train", "val", "test")
    )
    for name, mapping in candidate_maps.items():
        action_matches = sum(
            mapping[action] == baseline_map[action] for action in normalized_actions
        )
        summary_rows.append(
            {
                "model": name,
                "metric": "action_macro_agreement",
                "scope": "all_13_actions",
                "matches": action_matches,
                "total": len(normalized_actions),
                "agreement_percent": pct(action_matches, len(normalized_actions)),
            }
        )
        unfixed_matches = sum(
            mapping[action] == baseline_map[action] for action in UNFIXED_ACTIONS
        )
        summary_rows.append(
            {
                "model": name,
                "metric": "unfixed_action_macro_agreement",
                "scope": "all_4_model_decided_actions",
                "matches": unfixed_matches,
                "total": len(UNFIXED_ACTIONS),
                "agreement_percent": pct(
                    unfixed_matches,
                    len(UNFIXED_ACTIONS),
                ),
            }
        )
        pair_matches = sum(
            mapping[action] == baseline_map[action]
            for action, _ in pair_groups
        )
        summary_rows.append(
            {
                "model": name,
                "metric": "unique_action_object_pair_agreement",
                "scope": "all_unique_pairs",
                "matches": pair_matches,
                "total": len(pair_groups),
                "agreement_percent": pct(pair_matches, len(pair_groups)),
            }
        )
        for scope, rows in scopes:
            matches = sum(
                row[f"{name}_matches_baseline"] is True for row in rows
            )
            summary_rows.append(
                {
                    "model": name,
                    "metric": "segment_weighted_agreement",
                    "scope": scope,
                    "matches": matches,
                    "total": len(rows),
                    "agreement_percent": pct(matches, len(rows)),
                }
            )
            unfixed_rows = [
                row for row in rows if row["action"] in UNFIXED_ACTIONS
            ]
            unfixed_segment_matches = sum(
                row[f"{name}_matches_baseline"] is True
                for row in unfixed_rows
            )
            summary_rows.append(
                {
                    "model": name,
                    "metric": "unfixed_segment_weighted_agreement",
                    "scope": scope,
                    "matches": unfixed_segment_matches,
                    "total": len(unfixed_rows),
                    "agreement_percent": pct(
                        unfixed_segment_matches,
                        len(unfixed_rows),
                    ),
                }
            )
    summary_fields = [
        "model",
        "metric",
        "scope",
        "matches",
        "total",
        "agreement_percent",
    ]
    write_csv(args.output_dir / "agreement_summary.csv", summary_rows, summary_fields)

    distribution_rows: list[dict[str, Any]] = []
    all_maps = {args.baseline_name: baseline_map, **candidate_maps}
    for split in ("all", "train", "val", "test"):
        rows = (
            segment_rows
            if split == "all"
            else [row for row in segment_rows if row["split"] == split]
        )
        for model_name, mapping in all_maps.items():
            counts = Counter(mapping[row["action"]] for row in rows)
            for intent in INTENTS:
                distribution_rows.append(
                    {
                        "split": split,
                        "model": model_name,
                        "intent": intent,
                        "count": counts[intent],
                        "total": len(rows),
                        "percent": pct(counts[intent], len(rows)),
                    }
                )
    distribution_fields = ["split", "model", "intent", "count", "total", "percent"]
    write_csv(
        args.output_dir / "intent_distribution.csv",
        distribution_rows,
        distribution_fields,
    )

    summary = {
        "baseline": {
            "report_name": args.baseline_name,
            "taxonomy_path": str(args.baseline.resolve()),
            "provenance_note": (
                "This saved taxonomy was originally generated by Qwen/Qwen3-14B "
                "NF4; it is treated as the Qwen3-VL-8B reference only at the "
                "user's explicit request."
            ),
        },
        "source_csv": str(args.source_csv.resolve()),
        "segments": len(segment_rows),
        "splits": dict(Counter(row["split"] for row in segment_rows)),
        "actions": len(normalized_actions),
        "fixed_anchor_actions": len(normalized_actions) - len(UNFIXED_ACTIONS),
        "model_decided_actions": list(UNFIXED_ACTIONS),
        "unique_action_object_pairs": len(pair_groups),
        "metrics": summary_rows,
        "baseline_taxonomy": baseline_taxonomy,
        "candidate_taxonomies": candidate_taxonomies,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# PROX fixed-7 intent agreement",
        "",
        (
            f"Reference: `{args.baseline_name}` (saved taxonomy originally "
            "generated by Qwen/Qwen3-14B NF4; relabeled for this requested comparison)."
        ),
        "",
        "| Model | Metric | Scope | Match | Agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['scope']} | "
            f"{row['matches']}/{row['total']} | {row['agreement_percent']:.2f}% |"
        )
    lines.extend(["", "## Action assignments", ""])
    for row in action_rows:
        assignments = ", ".join(
            f"{model}={row[model]}" for model in models
        )
        lines.append(
            f"- `{row['action']}` (n={row['segment_count']}): {assignments}"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
