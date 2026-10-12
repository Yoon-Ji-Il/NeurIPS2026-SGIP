#!/usr/bin/env python3
"""Compare HIK student and teacher selected-object point-cloud caches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path(
    "data/HIK/"
    "hik_preprocessed/H25F50_nonwalking/SLICES_888s"
)
DEFAULT_TEACHER = (
    "final_object_qwen3_14b_action_geometry_h25f50_"
    "t25_gt29_points_1024.pth"
)
DEFAULT_STUDENT = (
    "final_object_student_hik_motion_c_kl_h25f50_nonwalking_"
    "seed7_points_1024.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--teacher-name", default=DEFAULT_TEACHER)
    parser.add_argument("--student-name", default=DEFAULT_STUDENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./runs_hik/"
            "hik_student_teacher_object_comparison"
        ),
    )
    parser.add_argument("--chamfer-subsample", type=int, default=128)
    parser.add_argument("--chamfer-batch-size", type=int, default=32)
    return parser.parse_args()


def stem(name: str) -> str:
    if not name.endswith(".pth"):
        raise ValueError(f"Expected .pth cache name: {name}")
    return name[:-4]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty comparison CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chamfer_subsampled(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    sample_count: int,
    batch_size: int,
) -> torch.Tensor:
    count = min(sample_count, student.shape[1], teacher.shape[1])
    student_indices = torch.linspace(
        0,
        student.shape[1] - 1,
        count,
    ).round().long()
    teacher_indices = torch.linspace(
        0,
        teacher.shape[1] - 1,
        count,
    ).round().long()
    output = []
    for start in range(0, student.shape[0], batch_size):
        student_batch = student[start : start + batch_size, student_indices]
        teacher_batch = teacher[start : start + batch_size, teacher_indices]
        distances = torch.cdist(student_batch, teacher_batch)
        symmetric = (
            distances.min(dim=2).values.mean(dim=1)
            + distances.min(dim=1).values.mean(dim=1)
        ) * 0.5
        output.append(symmetric)
    return torch.cat(output, dim=0)


def main() -> None:
    args = parse_args()
    if args.chamfer_subsample <= 0 or args.chamfer_batch_size <= 0:
        raise ValueError("Chamfer sampling and batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in ("train", "test"):
        split_root = args.cache_root / split
        teacher = torch.load(
            split_root / args.teacher_name,
            map_location="cpu",
            weights_only=False,
        ).float()
        student = torch.load(
            split_root / args.student_name,
            map_location="cpu",
            weights_only=False,
        ).float()
        teacher_mask = torch.load(
            split_root / f"{stem(args.teacher_name)}_mask.pth",
            map_location="cpu",
            weights_only=False,
        ).bool()
        student_mask = torch.load(
            split_root / f"{stem(args.student_name)}_mask.pth",
            map_location="cpu",
            weights_only=False,
        ).bool()
        teacher_manifest = read_json(
            split_root / f"{stem(args.teacher_name)}_manifest.json"
        )
        student_manifest = read_json(
            split_root / f"{stem(args.student_name)}_manifest.json"
        )

        expected_shape = (len(teacher_manifest), 1024, 3)
        if tuple(teacher.shape) != expected_shape:
            raise ValueError(
                f"Unexpected teacher shape for {split}: {tuple(teacher.shape)}"
            )
        if tuple(student.shape) != expected_shape:
            raise ValueError(
                f"Unexpected student shape for {split}: {tuple(student.shape)}"
            )
        if len(student_manifest) != len(teacher_manifest):
            raise ValueError(f"Manifest length mismatch for {split}")
        if teacher_mask.shape != student_mask.shape:
            raise ValueError(f"Mask shape mismatch for {split}")

        teacher_ids = [str(row["sample_id"]) for row in teacher_manifest]
        student_ids = [str(row["sample_id"]) for row in student_manifest]
        if teacher_ids != student_ids:
            raise ValueError(f"Sample order mismatch for {split}")

        valid_pair = teacher_mask & student_mask
        exact_points = torch.isclose(
            student,
            teacher,
            rtol=0.0,
            atol=0.0,
        ).all(dim=2).all(dim=1)
        allclose_points = torch.isclose(
            student,
            teacher,
            rtol=1.0e-5,
            atol=1.0e-6,
        ).all(dim=2).all(dim=1)
        centroid_distance = torch.linalg.norm(
            student.mean(dim=1) - teacher.mean(dim=1),
            dim=1,
        )
        pointwise_rmse = (
            (student - teacher).square().sum(dim=2).mean(dim=1).sqrt()
        )
        chamfer = chamfer_subsampled(
            student,
            teacher,
            sample_count=args.chamfer_subsample,
            batch_size=args.chamfer_batch_size,
        )

        action_stats: dict[str, dict[str, int]] = {}
        split_rows = []
        for index, (teacher_row, student_row) in enumerate(
            zip(teacher_manifest, student_manifest, strict=True)
        ):
            action = str(teacher_row.get("event_action", ""))
            teacher_object_id = str(teacher_row.get("selected_object_id", ""))
            student_object_id = str(student_row.get("selected_object_id", ""))
            teacher_object_type = str(
                teacher_row.get(
                    "object_type",
                    teacher_row.get("selected_object_name", ""),
                )
            )
            student_object_type = str(
                student_row.get(
                    "object_type",
                    student_row.get("selected_object_name", ""),
                )
            )
            object_match = teacher_object_id == student_object_id
            type_match = teacher_object_type == student_object_type
            stats = action_stats.setdefault(
                action,
                {"count": 0, "object_id_matches": 0, "object_type_matches": 0},
            )
            stats["count"] += 1
            stats["object_id_matches"] += int(object_match)
            stats["object_type_matches"] += int(type_match)
            split_rows.append(
                {
                    "split": split,
                    "index": index,
                    "sample_id": teacher_row["sample_id"],
                    "event_action": action,
                    "teacher_object_id": teacher_object_id,
                    "student_object_id": student_object_id,
                    "object_id_match": int(object_match),
                    "teacher_object_type": teacher_object_type,
                    "student_object_type": student_object_type,
                    "object_type_match": int(type_match),
                    "teacher_valid": int(teacher_mask[index]),
                    "student_valid": int(student_mask[index]),
                    "point_cloud_exact_match": int(exact_points[index]),
                    "point_cloud_allclose": int(allclose_points[index]),
                    "centroid_distance": float(centroid_distance[index]),
                    "pointwise_rmse": float(pointwise_rmse[index]),
                    "chamfer_approx": float(chamfer[index]),
                }
            )
        all_rows.extend(split_rows)

        valid_count = int(valid_pair.sum())
        object_matches = sum(int(row["object_id_match"]) for row in split_rows)
        type_matches = sum(int(row["object_type_match"]) for row in split_rows)
        for stats in action_stats.values():
            stats["object_id_match_rate"] = (
                stats["object_id_matches"] / stats["count"]
            )
            stats["object_type_match_rate"] = (
                stats["object_type_matches"] / stats["count"]
            )
        split_summaries[split] = {
            "count": len(split_rows),
            "teacher_shape": list(teacher.shape),
            "student_shape": list(student.shape),
            "teacher_valid": int(teacher_mask.sum()),
            "student_valid": int(student_mask.sum()),
            "valid_pairs": valid_count,
            "object_id_matches": object_matches,
            "object_id_match_rate": object_matches / len(split_rows),
            "object_type_matches": type_matches,
            "object_type_match_rate": type_matches / len(split_rows),
            "point_cloud_exact_matches": int((exact_points & valid_pair).sum()),
            "point_cloud_allclose_matches": int(
                (allclose_points & valid_pair).sum()
            ),
            "mean_centroid_distance": float(centroid_distance[valid_pair].mean()),
            "mean_pointwise_rmse": float(pointwise_rmse[valid_pair].mean()),
            "mean_chamfer_approx": float(chamfer[valid_pair].mean()),
            "action_breakdown": action_stats,
        }

    total = len(all_rows)
    total_id_matches = sum(int(row["object_id_match"]) for row in all_rows)
    total_type_matches = sum(int(row["object_type_match"]) for row in all_rows)
    summary = {
        "cache_root": str(args.cache_root.resolve()),
        "teacher_name": args.teacher_name,
        "student_name": args.student_name,
        "comparison_note": (
            "The .pth files contain selected-object point clouds. The model "
            "computes learned object embeddings online with its object encoder."
        ),
        "chamfer_subsample": args.chamfer_subsample,
        "splits": split_summaries,
        "overall": {
            "count": total,
            "object_id_matches": total_id_matches,
            "object_id_match_rate": total_id_matches / total,
            "object_type_matches": total_type_matches,
            "object_type_match_rate": total_type_matches / total,
        },
    }
    write_csv(args.output_dir / "sample_comparison.csv", all_rows)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
