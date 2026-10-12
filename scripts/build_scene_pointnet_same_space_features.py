#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pointnet_plus2 import PointNet2SemSegSSGShape  # noqa: E402


DEFAULT_CANDIDATE_CSV = (
    REPO_ROOT
    / "object_selector"
    / "teacher_targets"
    / "qwen3_14b_sequence_selector_gpu1_5"
    / "motion_c_fusion_score_final"
    / "teacher_candidate_scores.csv"
)
DEFAULT_SCENE_GRAPH_ROOT = REPO_ROOT / "object_selector" / "scene_graphs"
DEFAULT_SLICE_ROOT = Path("data/GIMO/SLICES_888s")
DEFAULT_CHECKPOINT = Path("./runs/sif3d_repro/15.pth")
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "object_selector"
    / "same_space_feature_inputs"
    / "teacher_candidate_scores_orig_sif3d_pointnet_mean.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append sequence-level SIF3D PointNet++ scene features to a same-space "
            "selector candidate CSV. This uses only scene_points_32768.pth and a "
            "trained SIF3D checkpoint; it never reads final_object_*_points_1024.pth."
        )
    )
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--scene-graph-root", type=Path, default=DEFAULT_SCENE_GRAPH_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scene-points-name", type=str, default="scene_points_32768.pth")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--feature-kind",
        choices=("mean", "max", "global", "mean_global"),
        default="mean",
        help=(
            "mean/max pool the per-point scene_feats used by SIF3D attention, "
            "or use the PointNet++ bottleneck global feature."
        ),
    )
    return parser.parse_args()


def clean_row(raw_row: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        row[clean_key] = str(value).strip() if value is not None else ""
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_scene_sample_ids(scene_graph_root: Path, split: str) -> list[str]:
    path = scene_graph_root / split / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene graph file: {path}")
    sample_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_ids.append(str(payload.get("sample_id", "")))
    return sample_ids


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def load_pointnet(checkpoint_path: Path, device: torch.device) -> PointNet2SemSegSSGShape:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = PointNet2SemSegSSGShape({"feat_dim": 256})
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint state dict, got {type(state)!r}")
    pointnet_state = {
        key.removeprefix("pointnet."): value
        for key, value in state.items()
        if key.startswith("pointnet.")
    }
    if not pointnet_state:
        raise KeyError(f"No pointnet.* keys found in {checkpoint_path}")
    missing, unexpected = model.load_state_dict(pointnet_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"PointNet state mismatch. missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
        )
    model.to(device)
    model.eval()
    return model


def feature_columns(kind: str, dim: int = 256) -> list[str]:
    if kind == "mean_global":
        return [f"scene_pn_mean_{idx:03d}" for idx in range(dim)] + [
            f"scene_pn_global_{idx:03d}" for idx in range(dim)
        ]
    prefix = {
        "mean": "scene_pn_mean",
        "max": "scene_pn_max",
        "global": "scene_pn_global",
    }[kind]
    return [f"{prefix}_{idx:03d}" for idx in range(dim)]


@torch.no_grad()
def extract_split_features(
    *,
    pointnet: PointNet2SemSegSSGShape,
    scene_points_path: Path,
    sample_ids: list[str],
    kind: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    scene_points = torch.load(scene_points_path, map_location="cpu", weights_only=False).float()
    if scene_points.ndim != 3 or scene_points.shape[-1] != 3:
        raise TypeError(f"Expected [N,P,3] scene points in {scene_points_path}, got {tuple(scene_points.shape)}")
    if int(scene_points.shape[0]) != len(sample_ids):
        raise ValueError(
            f"{scene_points_path}: tensor length {scene_points.shape[0]} does not match "
            f"scene graph samples {len(sample_ids)}"
        )

    features_by_sample: dict[str, torch.Tensor] = {}
    duplicate_mismatches: list[dict[str, Any]] = []
    safe_batch_size = max(int(batch_size), 1)
    for start in range(0, len(sample_ids), safe_batch_size):
        end = min(start + safe_batch_size, len(sample_ids))
        batch = scene_points[start:end].to(device=device, non_blocking=True)
        scene_feats, global_feats = pointnet(batch.repeat(1, 1, 2))
        scene_feats = scene_feats.detach().cpu()
        global_feats = global_feats.detach().cpu()
        if kind == "mean":
            batch_features = scene_feats.mean(dim=-1)
        elif kind == "max":
            batch_features = scene_feats.amax(dim=-1)
        elif kind == "global":
            batch_features = global_feats
        elif kind == "mean_global":
            batch_features = torch.cat([scene_feats.mean(dim=-1), global_feats], dim=-1)
        else:
            raise ValueError(f"Unsupported feature kind: {kind}")

        for offset, sample_id in enumerate(sample_ids[start:end]):
            feature = batch_features[offset].float().clone()
            previous = features_by_sample.get(sample_id)
            if previous is not None:
                max_abs_diff = float((previous - feature).abs().max().item())
                if max_abs_diff > 1.0e-5:
                    duplicate_mismatches.append(
                        {
                            "sample_id": sample_id,
                            "first_index_feature_norm": float(previous.norm().item()),
                            "duplicate_index": start + offset,
                            "max_abs_diff": max_abs_diff,
                        }
                    )
                continue
            features_by_sample[sample_id] = feature

    if duplicate_mismatches:
        raise RuntimeError(f"Duplicate sample ids produced different scene PointNet features: {duplicate_mismatches[:5]}")

    meta = {
        "scene_points_path": str(scene_points_path),
        "scene_points_shape": list(scene_points.shape),
        "num_scene_graph_samples": len(sample_ids),
        "num_unique_sample_ids": len(features_by_sample),
        "feature_dim": int(next(iter(features_by_sample.values())).shape[0]) if features_by_sample else 0,
    }
    return features_by_sample, meta


def main() -> None:
    args = parse_args()
    rows = read_csv(args.candidate_csv)
    if not rows:
        raise RuntimeError(f"No candidate rows in {args.candidate_csv}")
    required = {"split", "sample_id"}
    missing_required = sorted(required - set(rows[0]))
    if missing_required:
        raise KeyError(f"Missing required columns in {args.candidate_csv}: {missing_required}")

    device = choose_device(args.device)
    pointnet = load_pointnet(args.checkpoint, device)

    features_by_split: dict[str, dict[str, torch.Tensor]] = {}
    meta: dict[str, Any] = {
        "candidate_csv": str(args.candidate_csv),
        "output_csv": str(args.output_csv),
        "checkpoint": str(args.checkpoint),
        "feature_kind": args.feature_kind,
        "scene_points_name": args.scene_points_name,
        "device": str(device),
        "splits": {},
    }
    for split in ("train", "test"):
        sample_ids = load_scene_sample_ids(args.scene_graph_root, split)
        features, split_meta = extract_split_features(
            pointnet=pointnet,
            scene_points_path=args.slice_root / split / args.scene_points_name,
            sample_ids=sample_ids,
            kind=args.feature_kind,
            batch_size=args.batch_size,
            device=device,
        )
        features_by_split[split] = features
        meta["splits"][split] = split_meta

    columns = feature_columns(args.feature_kind, dim=meta["splits"]["test"]["feature_dim"] // (2 if args.feature_kind == "mean_global" else 1))
    if len(columns) != int(meta["splits"]["test"]["feature_dim"]):
        raise RuntimeError(f"Feature column count {len(columns)} does not match extracted dim {meta['splits']['test']['feature_dim']}")

    missing_rows: list[dict[str, str]] = []
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        split = row.get("split", "")
        sample_id = row.get("sample_id", "")
        feature = features_by_split.get(split, {}).get(sample_id)
        if feature is None:
            missing_rows.append({"split": split, "sample_id": sample_id})
            values = [0.0 for _ in columns]
        else:
            values = [float(value) for value in feature.tolist()]
        output = dict(row)
        output.update(dict(zip(columns, values, strict=True)))
        output_rows.append(output)

    fieldnames = list(rows[0].keys())
    for column in columns:
        if column not in fieldnames:
            fieldnames.append(column)
    write_csv(args.output_csv, output_rows, fieldnames)

    meta.update(
        {
            "num_input_rows": len(rows),
            "feature_columns": columns,
            "num_missing_rows": len(missing_rows),
            "missing_rows": missing_rows[:20],
        }
    )
    meta_path = args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
        file.write("\n")

    print(f"wrote {args.output_csv}")
    print(f"feature_kind={args.feature_kind} dim={len(columns)} missing_rows={len(missing_rows)}")
    print(f"meta={meta_path}")


if __name__ == "__main__":
    main()
