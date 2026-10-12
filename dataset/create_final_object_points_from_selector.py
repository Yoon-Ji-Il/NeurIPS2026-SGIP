import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SIF3D object point tensors from an object_selector sample_summary CSV."
    )
    parser.add_argument(
        "--selector-summary-csv",
        default=(
            "./object_selector/results/"
            "qwen3_14b_fan120_threshold_scenehit_maxtok8192_parallel5/"
            "qwen3_14b_fan120_threshold_scenehit_maxtok8192_merged5_sample_summary.csv"
        ),
        help="Merged selector sample_summary CSV with selected_object_id per sample.",
    )
    parser.add_argument(
        "--gimo-dataset-csv",
        default="data/GIMO/dataset.csv",
        help="Canonical GIMO dataset.csv used to define train/test sample order.",
    )
    parser.add_argument(
        "--gimo-root",
        default="data/GIMO",
        help="GIMO root containing scene transform files.",
    )
    parser.add_argument(
        "--openins3d-root",
        default="data/OpenIns3D/output/results_demo",
        help="Root containing scene label_pointcloud_export.json files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/GIMO/SLICES_888s",
        help="SLICES_888s root. The script writes into train/ and test/.",
    )
    parser.add_argument(
        "--num-object-points",
        default=1024,
        type=int,
        help="Number of object points stored per sample.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output .pth filename under each split directory. Defaults to "
            "final_object_selector_points_<num-object-points>.pth"
        ),
    )
    return parser.parse_args()


def clean_row(raw_row):
    row = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        clean_key = key.lstrip("\ufeff").strip()
        if not clean_key:
            continue
        row[clean_key] = value.strip() if isinstance(value, str) else value
    return row


def load_csv_rows(path: Path, required=()):
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as f:
                rows = [clean_row(row) for row in csv.DictReader(f)]
            break
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    else:
        raise UnicodeDecodeError(
            "csv",
            b"",
            0,
            1,
            f"Could not decode {path}; tried encodings: {errors}",
        )

    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    return rows


def norm_value(value):
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def sample_key(row):
    return (
        norm_value(row["scene"]),
        norm_value(row["sequence_path"]),
        norm_value(row["start_frame"]),
        norm_value(row["end_frame"]),
    )


def parse_selector_key(row):
    sample_id = norm_value(row.get("sample_id", ""))
    parts = sample_id.split("|")
    if len(parts) == 4:
        return tuple(norm_value(part) for part in parts)
    raise KeyError(
        "Selector summary must contain sample_id formatted as "
        "scene|sequence_path|start_frame|end_frame"
    )


def normalize_instance_id(raw_value):
    value = norm_value(raw_value)
    if not value:
        return ""
    if value.startswith("instance_"):
        return value
    if value.isdigit():
        return f"instance_{int(value):04d}"
    return value


def build_selector_lookup(rows):
    lookup = {}
    duplicate_conflicts = []
    for row in rows:
        key = parse_selector_key(row)
        selected_id = normalize_instance_id(row.get("selected_object_id", ""))
        if key in lookup and lookup[key].get("selected_object_id") != selected_id:
            duplicate_conflicts.append(
                {
                    "sample_key": list(key),
                    "previous_selected_object_id": lookup[key].get("selected_object_id", ""),
                    "new_selected_object_id": selected_id,
                }
            )
            continue
        normalized = dict(row)
        normalized["selected_object_id"] = selected_id
        lookup[key] = normalized
    return lookup, duplicate_conflicts


def build_scene_instance_index(openins3d_root: Path):
    scene_to_instances = {}
    for export_path in sorted(openins3d_root.glob("*/label_pointcloud_export.json")):
        with export_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        scene_name = data.get("scene_name") or export_path.parent.name
        instance_map = {}
        for item in data.get("instances", []):
            instance_id = normalize_instance_id(item.get("instance_id", ""))
            ply_rel = item.get("ply_path")
            if not instance_id or not ply_rel:
                continue
            ply_path = (export_path.parent / ply_rel).resolve()
            instance_map[instance_id] = {
                "ply_path": str(ply_path),
                "semantic_label_name": item.get("semantic_label_name", ""),
                "category": item.get("category", ""),
                "num_points": item.get("num_points", 0),
            }
        scene_to_instances[scene_name] = instance_map
    return scene_to_instances


def load_raw_points(ply_path: Path):
    mesh = trimesh.load(ply_path, process=False)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise ValueError(f"No vertices attribute in {ply_path}")
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty point cloud with shape (N, 3), got {points.shape}")
    return points


def load_transform_context(gimo_root: Path, row):
    scene_name = norm_value(row["scene"])
    sequence_name = norm_value(row["sequence_path"])
    transform_name = norm_value(row["transformation"])

    transform_info_path = gimo_root / scene_name / sequence_name / transform_name
    with transform_info_path.open("r", encoding="utf-8") as f:
        transform_info = json.load(f)

    scale = float(transform_info["scale"])
    transform_norm = np.loadtxt(
        gimo_root / scene_name / "scene_obj" / "transform_norm.txt",
        dtype=np.float32,
    ).reshape(4, 4)
    transform_norm[:3, 3] /= scale
    return scale, transform_norm


def apply_scene_normalization(points, scale, transform_norm):
    scaled = np.asarray(points, dtype=np.float32) * (1.0 / scale)
    return (transform_norm[:3, :3] @ scaled.T + transform_norm[:3, 3:]).T.astype(np.float32)


def stable_seed(key, instance_id):
    joined = "|".join(str(part) for part in (*key, instance_id))
    digest = hashlib.md5(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_object_points(points, num_points, seed):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud with shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("Object point cloud is empty")
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < num_points
    indices = rng.choice(points.shape[0], size=num_points, replace=replace)
    return points[indices].astype(np.float32, copy=False), replace


def build_split(
    split_name,
    gimo_rows,
    selector_lookup,
    scene_index,
    output_root: Path,
    output_name: str,
    gimo_root: Path,
    gimo_dataset_csv: Path,
    selector_summary_csv: Path,
    openins3d_root: Path,
    num_object_points: int,
    duplicate_conflicts,
):
    split_value = "1" if split_name == "train" else "0"
    split_rows = [row for row in gimo_rows if norm_value(row["training"]) == split_value]
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    tensor = torch.zeros((len(split_rows), num_object_points, 3), dtype=torch.float32)
    mask = torch.zeros((len(split_rows),), dtype=torch.bool)
    manifest = []
    missing = []
    raw_points_cache = {}

    for index, row in enumerate(split_rows):
        key = sample_key(row)
        scene_name = key[0]
        selector_row = selector_lookup.get(key)
        manifest_entry = {
            "index": index,
            "split": split_name,
            "scene_name": scene_name,
            "sequence_name": key[1],
            "start_frame": key[2],
            "end_frame": key[3],
            "training": norm_value(row["training"]),
            "transformation": norm_value(row["transformation"]),
            "sample_key": list(key),
        }

        if selector_row is None:
            manifest_entry["status"] = "missing_selector_row"
            missing.append(manifest_entry)
            manifest.append(manifest_entry)
            continue

        selected_id = normalize_instance_id(selector_row.get("selected_object_id", ""))
        manifest_entry.update(
            {
                "selected_object_id": selected_id,
                "selected_object_name": selector_row.get("selected_object_name", ""),
                "selected_matches_gt": selector_row.get("selected_matches_gt", ""),
                "gt_object_id": selector_row.get("gt_object_id", ""),
                "gt_object": selector_row.get("gt_object", ""),
                "gt_action": selector_row.get("gt_action", ""),
            }
        )

        if not selected_id:
            manifest_entry["status"] = "empty_selected_object_id"
            missing.append(manifest_entry)
            manifest.append(manifest_entry)
            continue

        scene_instances = scene_index.get(scene_name)
        if scene_instances is None:
            manifest_entry["status"] = "missing_scene_json"
            missing.append(manifest_entry)
            manifest.append(manifest_entry)
            continue

        instance_info = scene_instances.get(selected_id)
        if instance_info is None:
            manifest_entry["status"] = "missing_selected_instance_in_scene_json"
            missing.append(manifest_entry)
            manifest.append(manifest_entry)
            continue

        ply_path = Path(instance_info["ply_path"])
        manifest_entry["object_ply_path"] = str(ply_path)
        manifest_entry["object_name_from_json"] = instance_info.get("semantic_label_name", "")
        manifest_entry["object_num_points_from_json"] = instance_info.get("num_points", 0)

        try:
            if ply_path not in raw_points_cache:
                raw_points_cache[ply_path] = load_raw_points(ply_path)
            scale, transform_norm = load_transform_context(gimo_root, row)
            aligned_points = apply_scene_normalization(raw_points_cache[ply_path], scale, transform_norm)
            sampled_points, used_replacement = sample_object_points(
                aligned_points,
                num_object_points,
                stable_seed(key, selected_id),
            )
            tensor[index] = torch.from_numpy(sampled_points)
            mask[index] = True
            manifest_entry["raw_point_count"] = int(aligned_points.shape[0])
            manifest_entry["stored_point_count"] = int(num_object_points)
            manifest_entry["used_replacement"] = bool(used_replacement)
            manifest_entry["status"] = "ok"
        except Exception as exc:
            manifest_entry["status"] = f"object_points_build_failed: {exc}"
            missing.append(manifest_entry)

        manifest.append(manifest_entry)

    output_path = split_dir / output_name
    mask_path = split_dir / output_name.replace(".pth", "_mask.pth")
    meta_path = split_dir / output_name.replace(".pth", "_meta.json")
    missing_path = split_dir / output_name.replace(".pth", "_missing.json")
    manifest_path = split_dir / output_name.replace(".pth", "_manifest.json")

    torch.save(tensor, output_path)
    torch.save(mask, mask_path)

    meta = {
        "split": split_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "valid_count": int(mask.sum().item()),
        "missing_count": len(missing),
        "num_object_points": num_object_points,
        "output_path": str(output_path),
        "mask_path": str(mask_path),
        "gimo_dataset_csv": str(gimo_dataset_csv),
        "selector_summary_csv": str(selector_summary_csv),
        "gimo_root": str(gimo_root),
        "openins3d_root": str(openins3d_root),
        "selector_duplicate_conflict_count": len(duplicate_conflicts),
        "selector_duplicate_conflicts": duplicate_conflicts,
        "sampling_rule": (
            "random sample from aligned selected OpenIns3D instance point cloud; "
            "replacement is used only when source has fewer points than requested"
        ),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with missing_path.open("w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[{split_name}] wrote {output_path}")
    print(f"[{split_name}] valid={meta['valid_count']} missing={meta['missing_count']} shape={meta['shape']}")


def main():
    args = parse_args()
    selector_summary_csv = Path(args.selector_summary_csv)
    gimo_dataset_csv = Path(args.gimo_dataset_csv)
    gimo_root = Path(args.gimo_root)
    openins3d_root = Path(args.openins3d_root)
    output_root = Path(args.output_root)
    output_name = args.output_name or f"final_object_selector_points_{args.num_object_points}.pth"

    if not output_name.endswith(".pth"):
        raise ValueError(f"--output-name must end with .pth, got: {output_name}")

    gimo_rows = load_csv_rows(
        gimo_dataset_csv,
        required=("scene", "sequence_path", "start_frame", "end_frame", "training", "transformation"),
    )
    selector_rows = load_csv_rows(
        selector_summary_csv,
        required=("sample_id", "selected_object_id"),
    )
    selector_lookup, duplicate_conflicts = build_selector_lookup(selector_rows)
    scene_index = build_scene_instance_index(openins3d_root)

    for split_name in ("train", "test"):
        build_split(
            split_name=split_name,
            gimo_rows=gimo_rows,
            selector_lookup=selector_lookup,
            scene_index=scene_index,
            output_root=output_root,
            output_name=output_name,
            gimo_root=gimo_root,
            gimo_dataset_csv=gimo_dataset_csv,
            selector_summary_csv=selector_summary_csv,
            openins3d_root=openins3d_root,
            num_object_points=args.num_object_points,
            duplicate_conflicts=duplicate_conflicts,
        )


if __name__ == "__main__":
    main()
