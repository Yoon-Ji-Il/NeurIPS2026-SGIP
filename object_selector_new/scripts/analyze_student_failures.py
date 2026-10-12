#!/usr/bin/env python3
"""Analyze per-sample failure modes for a trained student export."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze student failure modes from candidate_scores.csv")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [
            {
                (key.lstrip("\ufeff").strip() if key else key): (value.strip() if isinstance(value, str) else str(value))
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(file)
        ]


def read_sample_meta(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            split = str(row.get("split", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            metadata = row.get("metadata", {}) or {}
            out[(split, sample_id)] = {
                "want": str(row.get("want") or metadata.get("action") or "").strip(),
                "gt_object_id": str(metadata.get("gt_object_id") or "").strip(),
                "gt_label": str(metadata.get("gt_label") or "").strip(),
                "scene_name": str(metadata.get("scene_name") or "").strip(),
            }
    return out


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_case(case: dict[str, Any]) -> str:
    return (
        f"- `{case['want']}` [{case['sample_id']}]: "
        f"student=`{case['student_name']}` ({case['student_category']}), "
        f"teacher=`{case['teacher_name']}` ({case['teacher_category']}), "
        f"gt=`{case['gt_name']}` ({case['gt_category']}); "
        f"gt_rank={case['gt_student_rank']}, "
        f"gap={case['student_minus_gt_posterior']:.4f}, "
        f"root_delta={case['selected_minus_gt_root']:.4f}, "
        f"face_delta={case['selected_minus_gt_face']:.4f}"
    )


def main() -> None:
    args = parse_args()
    sample_meta = read_sample_meta(args.samples_jsonl)
    rows = read_csv_rows(args.candidate_scores)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split = row.get("split", "")
        sample_id = row.get("sample_id", "")
        if split == args.split and sample_id:
            grouped[(split, sample_id)].append(row)

    summary_counter: Counter[str] = Counter()
    by_want: dict[str, Counter[str]] = defaultdict(Counter)
    failure_cases: list[dict[str, Any]] = []
    feature_deltas: list[tuple[float, float]] = []
    gap_counter: Counter[str] = Counter()

    for key, group in sorted(grouped.items()):
        meta = sample_meta.get(key, {})
        want = meta.get("want", "")
        student = next((row for row in group if row.get("is_student_selected") == "1"), None)
        teacher = next((row for row in group if row.get("is_teacher_selected") == "1"), None)
        gt = next((row for row in group if row.get("is_gt") == "1"), None)
        if student is None or teacher is None or gt is None:
            raise ValueError(f"Missing student/teacher/gt row for sample {key}")

        student_correct = student["object_id"] == gt["object_id"]
        teacher_correct = teacher["object_id"] == gt["object_id"]
        if student_correct and teacher_correct:
            bucket = "both_correct"
        elif student_correct and not teacher_correct:
            bucket = "student_only_correct"
        elif not student_correct and teacher_correct:
            bucket = "teacher_only_correct"
        else:
            bucket = "both_wrong"
        summary_counter[bucket] += 1
        by_want[want]["total"] += 1
        by_want[want][bucket] += 1

        if student_correct:
            continue

        selected_minus_gt_root = parse_float(student.get("root_horizontal_alignment")) - parse_float(gt.get("root_horizontal_alignment"))
        selected_minus_gt_face = parse_float(student.get("face_normal_alignment")) - parse_float(gt.get("face_normal_alignment"))
        student_minus_gt_posterior = parse_float(student.get("student_posterior")) - parse_float(gt.get("student_posterior"))
        feature_deltas.append((selected_minus_gt_root, selected_minus_gt_face))

        if student_minus_gt_posterior < 0.02:
            gap_counter["lt_0_02"] += 1
        elif student_minus_gt_posterior < 0.05:
            gap_counter["0_02_to_0_05"] += 1
        else:
            gap_counter["ge_0_05"] += 1

        failure_cases.append(
            {
                "sample_id": key[1],
                "scene_name": meta.get("scene_name", ""),
                "want": want,
                "bucket": bucket,
                "student_name": student.get("name", ""),
                "student_category": student.get("category", ""),
                "teacher_name": teacher.get("name", ""),
                "teacher_category": teacher.get("category", ""),
                "gt_name": gt.get("name", ""),
                "gt_category": gt.get("category", ""),
                "student_teacher_same_object": int(student["object_id"] == teacher["object_id"]),
                "teacher_correct": int(teacher_correct),
                "gt_student_rank": int(student.get("student_rank", "0")) if student_correct else int(gt.get("student_rank", "0")),
                "gt_teacher_rank": int(gt.get("teacher_rank", "0")),
                "student_posterior": parse_float(student.get("student_posterior")),
                "gt_student_posterior": parse_float(gt.get("student_posterior")),
                "student_minus_gt_posterior": student_minus_gt_posterior,
                "selected_minus_gt_root": selected_minus_gt_root,
                "selected_minus_gt_face": selected_minus_gt_face,
            }
        )

    total_samples = sum(summary_counter.values())
    failure_count = len(failure_cases)
    mean_root_delta = sum(delta[0] for delta in feature_deltas) / max(len(feature_deltas), 1)
    mean_face_delta = sum(delta[1] for delta in feature_deltas) / max(len(feature_deltas), 1)

    by_want_rows = []
    for want, counter in sorted(by_want.items()):
        total = counter["total"]
        by_want_rows.append(
            {
                "want": want,
                "total": total,
                "student_accuracy": (counter["both_correct"] + counter["student_only_correct"]) / max(total, 1),
                "teacher_accuracy": (counter["both_correct"] + counter["teacher_only_correct"]) / max(total, 1),
                "both_correct": counter["both_correct"],
                "teacher_only_correct": counter["teacher_only_correct"],
                "student_only_correct": counter["student_only_correct"],
                "both_wrong": counter["both_wrong"],
            }
        )

    failure_cases_sorted = sorted(
        failure_cases,
        key=lambda row: (
            0 if row["bucket"] == "teacher_only_correct" else 1,
            row["gt_student_rank"],
            -row["student_minus_gt_posterior"],
        ),
    )

    teacher_only_cases = [row for row in failure_cases_sorted if row["bucket"] == "teacher_only_correct"]
    both_wrong_cases = [row for row in failure_cases_sorted if row["bucket"] == "both_wrong"]
    student_only_cases = [
        {
            "sample_id": key[1],
            "want": sample_meta.get(key, {}).get("want", ""),
            "student_name": next(row for row in group if row.get("is_student_selected") == "1").get("name", ""),
            "teacher_name": next(row for row in group if row.get("is_teacher_selected") == "1").get("name", ""),
            "gt_name": next(row for row in group if row.get("is_gt") == "1").get("name", ""),
        }
        for key, group in sorted(grouped.items())
        if (
            next(row for row in group if row.get("is_student_selected") == "1")["object_id"]
            == next(row for row in group if row.get("is_gt") == "1")["object_id"]
            and next(row for row in group if row.get("is_teacher_selected") == "1")["object_id"]
            != next(row for row in group if row.get("is_gt") == "1")["object_id"]
        )
    ]

    summary = {
        "candidate_scores": str(args.candidate_scores),
        "samples_jsonl": str(args.samples_jsonl),
        "split": args.split,
        "total_samples": total_samples,
        "student_correct": summary_counter["both_correct"] + summary_counter["student_only_correct"],
        "teacher_correct": summary_counter["both_correct"] + summary_counter["teacher_only_correct"],
        "both_correct": summary_counter["both_correct"],
        "teacher_only_correct": summary_counter["teacher_only_correct"],
        "student_only_correct": summary_counter["student_only_correct"],
        "both_wrong": summary_counter["both_wrong"],
        "failure_count": failure_count,
        "failure_gap_bins": dict(gap_counter),
        "mean_selected_minus_gt_root": mean_root_delta,
        "mean_selected_minus_gt_face": mean_face_delta,
        "by_want": by_want_rows,
        "representative_teacher_only_correct": teacher_only_cases[:6],
        "representative_both_wrong": both_wrong_cases[:6],
        "student_only_correct_cases": student_only_cases,
    }

    report_lines = [
        "# Student Failure Analysis",
        "",
        f"- Split: `{args.split}`",
        f"- Total samples: `{total_samples}`",
        f"- Student exact: `{summary['student_correct']}/{total_samples}`",
        f"- Teacher exact: `{summary['teacher_correct']}/{total_samples}`",
        f"- `teacher_only_correct`: `{summary_counter['teacher_only_correct']}`",
        f"- `student_only_correct`: `{summary_counter['student_only_correct']}`",
        f"- `both_wrong`: `{summary_counter['both_wrong']}`",
        "",
        "## Main Findings",
        "",
        f"- Most student failures are cases the teacher can solve: `{summary_counter['teacher_only_correct']}/{failure_count}`.",
        f"- On failed samples, the chosen object has higher geometry than the GT on average: "
        f"`root_horizontal_alignment +{mean_root_delta:.4f}`, "
        f"`face_normal_alignment +{mean_face_delta:.4f}`.",
        f"- Most failures are near-ties under the student posterior: "
        f"`<0.02`: `{gap_counter.get('lt_0_02', 0)}/{failure_count}`, "
        f"`0.02-0.05`: `{gap_counter.get('0_02_to_0_05', 0)}/{failure_count}`.",
        "",
        "## By Want",
        "",
    ]
    for row in by_want_rows:
        report_lines.append(
            f"- `{row['want']}`: student `{row['student_accuracy']:.3f}`, teacher `{row['teacher_accuracy']:.3f}` "
            f"over `{row['total']}` samples "
            f"(teacher_only={row['teacher_only_correct']}, both_wrong={row['both_wrong']})"
        )

    report_lines.extend(
        [
            "",
            "## Representative Teacher-Only Correct Cases",
            "",
        ]
    )
    report_lines.extend(format_case(case) for case in teacher_only_cases[:8])

    report_lines.extend(
        [
            "",
            "## Representative Both-Wrong Cases",
            "",
        ]
    )
    report_lines.extend(format_case(case) for case in both_wrong_cases[:8])

    report_lines.extend(
        [
            "",
            "## Student-Only Correct Cases",
            "",
        ]
    )
    for case in student_only_cases:
        report_lines.append(
            f"- `{case['want']}` [{case['sample_id']}]: student=`{case['student_name']}`, "
            f"teacher=`{case['teacher_name']}`, gt=`{case['gt_name']}`"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "failure_cases.csv", failure_cases_sorted)
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        f"[done] student={summary['student_correct']}/{total_samples}, "
        f"teacher={summary['teacher_correct']}/{total_samples}, "
        f"teacher_only={summary_counter['teacher_only_correct']}, "
        f"both_wrong={summary_counter['both_wrong']}"
    )


if __name__ == "__main__":
    main()
