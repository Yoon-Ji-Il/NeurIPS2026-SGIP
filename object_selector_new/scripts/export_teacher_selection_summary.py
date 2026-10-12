"""Convert a long-format teacher CSV into one selected-vs-GT row per sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "sample_index",
    "split",
    "sample_id",
    "scene",
    "sequence_path",
    "start_frame",
    "end_frame",
    "action",
    "want",
    "num_candidates",
    "selected_object_id",
    "selected_name",
    "selected_category",
    "selected_rubric",
    "selected_rubric_posterior",
    "selected_final_score",
    "selected_fusion_posterior",
    "gt_object_id",
    "gt_name",
    "gt_category",
    "exact_match",
    "class_match",
    "winner_stage",
    "tie_set_size",
    "tie_set_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write one selected-object vs GT row per teacher input sample."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    return parser.parse_args()


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_stat() -> dict[str, int]:
    return {"evaluated": 0, "exact_correct": 0, "class_correct": 0}


def add_stat(stat: dict[str, int], *, exact: bool, class_match: bool) -> None:
    stat["evaluated"] += 1
    stat["exact_correct"] += int(exact)
    stat["class_correct"] += int(class_match)


def finalize_stat(stat: dict[str, int]) -> dict[str, int | float]:
    denominator = max(stat["evaluated"], 1)
    return {
        **stat,
        "exact_accuracy": stat["exact_correct"] / denominator,
        "class_accuracy": stat["class_correct"] / denominator,
    }


def export_summary(
    *,
    input_jsonl: Path,
    teacher_csv: Path,
    output_csv: Path,
    metrics_json: Path,
) -> dict[str, Any]:
    samples = read_jsonl(input_jsonl)
    candidate_rows = read_csv(teacher_csv)
    expected_candidate_rows = sum(len(sample.get("object_candidates", [])) for sample in samples)
    if len(candidate_rows) != expected_candidate_rows:
        raise ValueError(
            f"teacher candidate row count mismatch: expected {expected_candidate_rows}, "
            f"found {len(candidate_rows)}"
        )

    summary_rows: list[dict[str, Any]] = []
    stats = {"all": make_stat(), "train": make_stat(), "test": make_stat()}
    offset = 0
    for sample_index, sample in enumerate(samples):
        raw_candidates = list(sample.get("object_candidates", []))
        count = len(raw_candidates)
        group = candidate_rows[offset : offset + count]
        offset += count

        expected_ids = [str(candidate.get("object_id", "")) for candidate in raw_candidates]
        actual_ids = [str(row.get("object_id", "")) for row in group]
        if actual_ids != expected_ids:
            raise ValueError(
                f"candidate order mismatch at sample_index={sample_index} "
                f"sample_id={sample.get('sample_id')!r}"
            )
        if any(row.get("sample_id") != str(sample.get("sample_id", "")) for row in group):
            raise ValueError(f"sample_id mismatch at sample_index={sample_index}")

        selected = [row for row in group if is_true(row.get("is_teacher_selected"))]
        gt_rows = [row for row in group if is_true(row.get("is_gt"))]
        if len(selected) != 1:
            raise ValueError(
                f"expected exactly one selected row at sample_index={sample_index}, found {len(selected)}"
            )
        if len(gt_rows) != 1:
            raise ValueError(
                f"expected exactly one GT row at sample_index={sample_index}, found {len(gt_rows)}"
            )

        chosen = selected[0]
        gt = gt_rows[0]
        exact = chosen["object_id"] == gt["object_id"]
        class_match = chosen["category"] == gt["category"]
        split = str(sample.get("split", ""))
        add_stat(stats["all"], exact=exact, class_match=class_match)
        if split in stats:
            add_stat(stats[split], exact=exact, class_match=class_match)

        metadata = dict(sample.get("metadata", {}))
        tie_ids = [row["object_id"] for row in group if is_true(row.get("is_in_tie_set"))]
        summary_rows.append(
            {
                "sample_index": sample_index,
                "split": split,
                "sample_id": sample.get("sample_id", ""),
                "scene": metadata.get("scene", ""),
                "sequence_path": metadata.get("sequence_path", ""),
                "start_frame": metadata.get("start_frame", ""),
                "end_frame": metadata.get("end_frame", ""),
                "action": metadata.get("action", ""),
                "want": sample.get("want", ""),
                "num_candidates": count,
                "selected_object_id": chosen.get("object_id", ""),
                "selected_name": chosen.get("name", ""),
                "selected_category": chosen.get("category", ""),
                "selected_rubric": chosen.get("rubric", ""),
                "selected_rubric_posterior": chosen.get("rubric_posterior", ""),
                "selected_final_score": chosen.get("final_score", ""),
                "selected_fusion_posterior": chosen.get("fusion_posterior", ""),
                "gt_object_id": gt.get("object_id", ""),
                "gt_name": gt.get("name", ""),
                "gt_category": gt.get("category", ""),
                "exact_match": int(exact),
                "class_match": int(class_match),
                "winner_stage": chosen.get("winner_stage", ""),
                "tie_set_size": chosen.get("tie_set_size", ""),
                "tie_set_ids": "|".join(tie_ids),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    metrics: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "teacher_csv": str(teacher_csv),
        "selection_summary_csv": str(output_csv),
        "num_samples": len(samples),
        "num_candidate_rows": len(candidate_rows),
        "metrics": {split: finalize_stat(stat) for split, stat in stats.items()},
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    metrics = export_summary(
        input_jsonl=args.input_jsonl,
        teacher_csv=args.teacher_csv,
        output_csv=args.output_csv,
        metrics_json=args.metrics_json,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
