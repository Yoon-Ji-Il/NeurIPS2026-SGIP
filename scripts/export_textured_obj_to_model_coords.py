#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite OBJ vertex positions into GIMO model coordinates while preserving texture assets."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/GIMO/dataset.csv"),
    )
    parser.add_argument(
        "--gimo-root",
        type=Path,
        default=Path("data/GIMO"),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "test", "all"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def norm(value: object) -> str:
    return str(value or "").strip()


def load_rows(dataset_csv: Path, scene_name: str, split: str) -> list[tuple[int, dict[str, str]]]:
    rows = list(csv.DictReader(dataset_csv.open("r", encoding="utf-8-sig", newline="")))
    out = []
    for idx, row in enumerate(rows):
        if scene_name and norm(row.get("scene")) != scene_name:
            continue
        training = norm(row.get("training"))
        if split == "train" and training != "1":
            continue
        if split == "test" and training != "0":
            continue
        out.append((idx, row))
    return out


def resolve_source_obj(scene_root: Path) -> Path:
    candidates = [
        scene_root / "scene_obj" / "textured_output.obj",
        scene_root / "scene_obj" / "textured_output_render.obj",
        scene_root / "scene_obj" / "textured_output1.obj",
        scene_root / "scene_obj" / "scene_obj.obj",
        scene_root / "scene_obj" / "scene.obj",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No usable OBJ asset found under {scene_root / 'scene_obj'}")


def load_raw_to_model(scene_root: Path, sequence_name: str, transform_name: str) -> tuple[np.ndarray, float]:
    transform_info = json.loads((scene_root / sequence_name / transform_name).read_text(encoding="utf-8"))
    scale = float(transform_info["scale"])
    transform_norm = np.loadtxt(scene_root / "scene_obj" / "transform_norm.txt", dtype=np.float64).reshape(4, 4)
    transform_norm[:3, 3] /= scale
    scale_inv = np.eye(4, dtype=np.float64)
    scale_inv[0, 0] = 1.0 / scale
    scale_inv[1, 1] = 1.0 / scale
    scale_inv[2, 2] = 1.0 / scale
    return transform_norm @ scale_inv, scale


def resolve_material_filename(src_obj: Path) -> tuple[str | None, Path | None]:
    referenced_name = None
    with src_obj.open("rb") as src:
        for line in src:
            if line.startswith(b"mtllib "):
                referenced_name = line.decode("latin1").strip().split(maxsplit=1)[1]
                referenced_name = Path(referenced_name).name
                break

    if referenced_name is not None:
        referenced_path = src_obj.parent / referenced_name
        if referenced_path.exists():
            return referenced_name, referenced_path

    fallback_candidates = [
        src_obj.with_suffix(".mtl"),
        src_obj.parent / "textured_output.mtl",
        src_obj.parent / "scene.mtl",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate.name, candidate
    return referenced_name, None


def transform_vertex_line(line: bytes, matrix: np.ndarray) -> bytes:
    parts = line.rstrip(b"\r\n").split()
    if len(parts) < 4 or parts[0] != b"v":
        return line
    xyz = np.array([[float(parts[1]), float(parts[2]), float(parts[3])]], dtype=np.float64)
    out = (matrix[:3, :3] @ xyz.T).T + matrix[:3, 3]
    transformed = out[0]
    suffix = parts[4:]
    prefix = f"v {transformed[0]:.8f} {transformed[1]:.8f} {transformed[2]:.8f}".encode("ascii")
    if suffix:
        prefix += b" " + b" ".join(suffix)
    return prefix + b"\n"


def rewrite_obj(src_obj: Path, dst_obj: Path, matrix: np.ndarray, target_mtl_name: str | None) -> None:
    with src_obj.open("rb") as src, dst_obj.open("wb") as dst:
        for line in src:
            if line.startswith(b"mtllib ") and target_mtl_name:
                dst.write(f"mtllib ./{target_mtl_name}\n".encode("ascii"))
            elif line.startswith(b"v "):
                dst.write(transform_vertex_line(line, matrix))
            else:
                dst.write(line)


def copy_texture_sidecars(src_obj: Path, dst_dir: Path, mtl_path: Path | None) -> None:
    if mtl_path is not None:
        shutil.copy2(mtl_path, dst_dir / mtl_path.name)
    for suffix in (".jpg", ".png", ".jpeg"):
        for path in src_obj.parent.glob(f"*{suffix}"):
            shutil.copy2(path, dst_dir / path.name)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_csv, args.scene, args.split)
    if not rows:
        raise RuntimeError(f"No rows selected for scene={args.scene!r}, split={args.split!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exports = []
    prepared_scenes: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for row_index, row in rows:
        scene_name = norm(row["scene"])
        sequence_name = norm(row["sequence_path"])
        transform_name = norm(row["transformation"])
        key = (scene_name, sequence_name, transform_name)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        scene_root = args.gimo_root / scene_name
        src_obj = resolve_source_obj(scene_root)
        scene_dir = args.output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        target_mtl_name, target_mtl_path = resolve_material_filename(src_obj)
        if scene_name not in prepared_scenes:
            copy_texture_sidecars(src_obj, scene_dir, target_mtl_path)
            prepared_scenes.add(scene_name)

        raw_to_model, scale = load_raw_to_model(scene_root, sequence_name, transform_name)
        out_obj = scene_dir / f"{sequence_name}__{Path(transform_name).stem}__row{row_index:03d}_model_textured.obj"
        rewrite_obj(src_obj, out_obj, raw_to_model, target_mtl_name)
        exports.append(
            {
                "dataset_row_index": row_index,
                "scene": scene_name,
                "sequence_path": sequence_name,
                "transformation": transform_name,
                "scale": scale,
                "source_obj": str(src_obj),
                "material_name": target_mtl_name,
                "material_path": None if target_mtl_path is None else str(target_mtl_path),
                "output_obj": str(out_obj),
                "raw_to_model_matrix": raw_to_model.tolist(),
            }
        )

    manifest = {
        "scene": args.scene,
        "split": args.split,
        "num_exports": len(exports),
        "exports": exports,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(exports)} textured OBJ exports to {args.output_dir}")


if __name__ == "__main__":
    main()
