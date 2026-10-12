#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


PLY_SCALAR_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def norm(value: object) -> str:
    return str(value or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sequence scene meshes in each sequence's motion-local coordinate frame."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/GIMO/dataset.csv"),
        help="Path to the GIMO dataset.csv file.",
    )
    parser.add_argument(
        "--gimo-root",
        type=Path,
        default=Path("data/GIMO"),
        help="Path to the GIMO dataset root.",
    )
    parser.add_argument(
        "--scene-mesh-dir",
        type=Path,
        default=Path("data/scene_mesh"),
        help="Directory containing raw scene meshes named <scene>_vertexcolor_mesh_hq.ply.",
    )
    parser.add_argument(
        "--asset-source",
        type=str,
        default="hq_ply",
        choices=("hq_ply", "gimo_scene_obj", "auto"),
        help=(
            "Which source asset to transform. "
            "'hq_ply' uses data/scene_mesh/*.ply. "
            "'gimo_scene_obj' prefers GIMO scene_obj assets such as textured_output.obj/scene.obj. "
            "'auto' tries GIMO scene_obj assets first and falls back to hq_ply."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where transformed meshes and a manifest will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of selected sequence-transform pairs to export. 0 means no limit.",
    )
    parser.add_argument(
        "--distinct-scenes",
        action="store_true",
        help="Select at most one sequence-transform pair per scene.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip mesh export when the output file already exists.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("all", "train", "test"),
        help="Which dataset split rows to export based on dataset.csv training flag.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="smplx_local",
        choices=("smplx_local", "model"),
        help=(
            "Target coordinate frame. "
            "'smplx_local' exports the raw scene into the sequence-local motion frame used by "
            "smplx_local/*.pkl. 'model' exports into the SLICES_888s model frame used by "
            "scene_points/poses_input."
        ),
    )
    parser.add_argument(
        "--file-type",
        type=str,
        default="ply",
        choices=("ply", "glb"),
        help="Output mesh file format.",
    )
    return parser.parse_args()


def read_rows(dataset_csv: Path) -> list[dict[str, str]]:
    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    limit: int,
    distinct_scenes: bool,
    split: str,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_sequence_transform: set[tuple[str, str, str]] = set()
    seen_scenes: set[str] = set()

    for row_index, row in enumerate(rows):
        training_value = norm(row.get("training"))
        if split == "train" and training_value != "1":
            continue
        if split == "test" and training_value != "0":
            continue

        scene_name = norm(row.get("scene"))
        sequence_name = norm(row.get("sequence_path"))
        transform_name = norm(row.get("transformation"))
        if not scene_name or not sequence_name or not transform_name:
            continue

        key = (scene_name, sequence_name, transform_name)
        if key in seen_sequence_transform:
            continue
        seen_sequence_transform.add(key)

        if distinct_scenes and scene_name in seen_scenes:
            continue
        seen_scenes.add(scene_name)

        item = dict(row)
        item["dataset_row_index"] = row_index
        item["scene_name"] = scene_name
        item["sequence_name"] = sequence_name
        item["transform_name"] = transform_name
        selected.append(item)

        if limit > 0 and len(selected) >= limit:
            break

    return selected


def resolve_hq_scene_mesh_path(scene_mesh_dir: Path, scene_name: str) -> Path:
    direct = scene_mesh_dir / f"{scene_name}_vertexcolor_mesh_hq.ply"
    if direct.exists():
        return direct

    matches = sorted(scene_mesh_dir.glob(f"{scene_name}*.ply"))
    if not matches:
        raise FileNotFoundError(f"Could not find a scene mesh for {scene_name!r} in {scene_mesh_dir}")
    if len(matches) > 1:
        raise FileExistsError(f"Scene {scene_name!r} matched multiple meshes: {matches}")
    return matches[0]


def resolve_gimo_scene_asset(scene_root: Path) -> Path | None:
    scene_obj_dir = scene_root / "scene_obj"
    candidates = [
        "textured_output.obj",
        "textured_output_render.obj",
        "textured_output1.obj",
        "scene_obj.obj",
        "scene.obj",
        "scene_downsampled.ply",
        "scene_normalized.ply",
    ]
    for name in candidates:
        path = scene_obj_dir / name
        if path.exists():
            return path
    return None


def resolve_source_asset(
    *,
    asset_source: str,
    scene_mesh_dir: Path,
    gimo_root: Path,
    scene_name: str,
) -> Path:
    scene_root = gimo_root / scene_name
    if asset_source in {"gimo_scene_obj", "auto"}:
        gimo_asset = resolve_gimo_scene_asset(scene_root)
        if gimo_asset is not None:
            return gimo_asset
        if asset_source == "gimo_scene_obj":
            raise FileNotFoundError(f"Could not find a usable GIMO scene_obj asset for {scene_name!r}")

    return resolve_hq_scene_mesh_path(scene_mesh_dir, scene_name)


def load_mesh(mesh_path: Path):
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        return mesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a trimesh.Trimesh or trimesh.Scene for {mesh_path}, got {type(mesh)!r}")
    return mesh


def parse_binary_ply_vertex_layout(ply_path: Path) -> tuple[bytes, int, np.dtype]:
    with ply_path.open("rb") as handle:
        header_lines: list[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is truncated: {ply_path}")
            header_lines.append(line)
            if line.rstrip(b"\r\n") == b"end_header":
                break

    header_bytes = b"".join(header_lines)
    current_element = None
    vertex_count = None
    vertex_fields: list[tuple[str, str]] = []
    format_name = None

    for raw_line in header_lines:
        line = raw_line.decode("latin1").strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "format":
            format_name = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError(f"Unsupported list property in vertex element for {ply_path}")
            dtype_code = PLY_SCALAR_DTYPES.get(parts[1])
            if dtype_code is None:
                raise ValueError(f"Unsupported PLY dtype {parts[1]!r} in {ply_path}")
            vertex_fields.append((parts[2], dtype_code))

    if format_name != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY, got {format_name!r} for {ply_path}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"Could not parse vertex count from {ply_path}")
    if not vertex_fields:
        raise ValueError(f"Could not parse vertex properties from {ply_path}")
    field_names = {name for name, _ in vertex_fields}
    if not {"x", "y", "z"}.issubset(field_names):
        raise ValueError(f"PLY vertex layout must contain x/y/z fields: {ply_path}")

    return header_bytes, vertex_count, np.dtype(vertex_fields)


def export_ply_preserving_payload(src_path: Path, dst_path: Path, matrix: np.ndarray) -> None:
    header_bytes, vertex_count, vertex_dtype = parse_binary_ply_vertex_layout(src_path)
    vertex_block_size = vertex_dtype.itemsize * vertex_count

    with src_path.open("rb") as src, dst_path.open("wb") as dst:
        src.seek(len(header_bytes))
        vertex_block = src.read(vertex_block_size)
        if len(vertex_block) != vertex_block_size:
            raise ValueError(f"Vertex block size mismatch while reading {src_path}")

        vertex_bytes = bytearray(vertex_block)
        vertex_array = np.frombuffer(vertex_bytes, dtype=vertex_dtype, count=vertex_count)
        vertices = np.stack(
            [vertex_array["x"], vertex_array["y"], vertex_array["z"]],
            axis=1,
        ).astype(np.float32, copy=False)
        transformed = transform_vertices(vertices, matrix)
        vertex_array["x"] = transformed[:, 0]
        vertex_array["y"] = transformed[:, 1]
        vertex_array["z"] = transformed[:, 2]

        dst.write(header_bytes)
        dst.write(vertex_array.tobytes())

        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def load_transform_info(transform_info_path: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(transform_info_path.read_text(encoding="utf-8"))
    scale = float(payload["scale"])
    transform = np.asarray(payload["transformation"], dtype=np.float64).reshape(4, 4)
    return transform, scale


def load_transform_norm(scene_root: Path, scale: float) -> np.ndarray:
    transform_norm = np.loadtxt(scene_root / "scene_obj" / "transform_norm.txt", dtype=np.float64).reshape(4, 4)
    transform_norm[:3, 3] /= scale
    return transform_norm


def build_raw_to_target_matrix(
    *,
    frame: str,
    scene_root: Path,
    transform_info_path: Path,
) -> tuple[np.ndarray, float]:
    transform_pose2scene, scale = load_transform_info(transform_info_path)
    scale_inv = np.eye(4, dtype=np.float64)
    scale_inv[0, 0] = 1.0 / scale
    scale_inv[1, 1] = 1.0 / scale
    scale_inv[2, 2] = 1.0 / scale

    if frame == "smplx_local":
        # GIMO preprocessing first divides the pose->scene translation by scale and then
        # applies that rigid transform to smplx_local coordinates. To invert that mapping
        # for raw scene vertices, we first scale raw scene points by 1/scale and then
        # apply the inverse adjusted pose->scene transform.
        adjusted_pose2scene = transform_pose2scene.copy()
        adjusted_pose2scene[:3, 3] /= scale
        raw_to_target = np.linalg.inv(adjusted_pose2scene) @ scale_inv
        return raw_to_target, scale

    if frame == "model":
        transform_norm = load_transform_norm(scene_root, scale)
        raw_to_target = transform_norm @ scale_inv
        return raw_to_target, scale

    raise ValueError(f"Unsupported frame: {frame}")


def transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    rotated = (matrix[:3, :3] @ verts.T).T
    translated = rotated + matrix[:3, 3]
    return translated.astype(np.float32)


def export_one(
    row: dict[str, object],
    gimo_root: Path,
    scene_mesh_dir: Path,
    output_dir: Path,
    skip_existing: bool,
    frame: str,
    file_type: str,
    asset_source: str,
) -> dict[str, object]:
    scene_name = str(row["scene_name"])
    sequence_name = str(row["sequence_name"])
    transform_name = str(row["transform_name"])

    transform_info_path = gimo_root / scene_name / sequence_name / transform_name
    scene_mesh_path = resolve_source_asset(
        asset_source=asset_source,
        scene_mesh_dir=scene_mesh_dir,
        gimo_root=gimo_root,
        scene_name=scene_name,
    )
    scene_root = gimo_root / scene_name
    raw_to_target, scale = build_raw_to_target_matrix(
        frame=frame,
        scene_root=scene_root,
        transform_info_path=transform_info_path,
    )

    scene_output_dir = output_dir / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    output_name = (
        f"{sequence_name}__{Path(transform_name).stem}"
        f"__row{int(row['dataset_row_index']):03d}_{frame}.{file_type}"
    )
    output_path = scene_output_dir / output_name

    if not (skip_existing and output_path.exists()):
        if file_type == "ply" and scene_mesh_path.suffix.lower() == ".ply":
            export_ply_preserving_payload(scene_mesh_path, output_path, raw_to_target)
        else:
            mesh = load_mesh(scene_mesh_path)
            transformed_mesh = mesh.copy()
            if isinstance(transformed_mesh, trimesh.Scene):
                transformed_mesh.apply_transform(raw_to_target)
            else:
                transformed_mesh.vertices = transform_vertices(np.asarray(transformed_mesh.vertices), raw_to_target)
            transformed_mesh.export(output_path)

    return {
        "dataset_row_index": int(row["dataset_row_index"]),
        "scene": scene_name,
        "sequence_path": sequence_name,
        "transformation": transform_name,
        "start_frame": norm(row.get("start_frame")),
        "end_frame": norm(row.get("end_frame")),
        "training": norm(row.get("training")),
        "scale": scale,
        "target_frame": frame,
        "file_type": file_type,
        "asset_source": asset_source,
        "source_mesh_path": str(scene_mesh_path),
        "transform_info_path": str(transform_info_path),
        "output_mesh_path": str(output_path),
        "raw_to_target_matrix": raw_to_target.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.dataset_csv)
    selected_rows = select_rows(rows, args.limit, args.distinct_scenes, args.split)
    if not selected_rows:
        raise RuntimeError("No rows were selected from dataset.csv")

    manifest = []
    for row in selected_rows:
        manifest.append(
            export_one(
                row=row,
                gimo_root=args.gimo_root,
                scene_mesh_dir=args.scene_mesh_dir,
                output_dir=args.output_dir,
                skip_existing=args.skip_existing,
                frame=args.frame,
                file_type=args.file_type,
                asset_source=args.asset_source,
            )
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_payload = {
        "dataset_csv": str(args.dataset_csv),
        "gimo_root": str(args.gimo_root),
        "scene_mesh_dir": str(args.scene_mesh_dir),
        "limit": args.limit,
        "distinct_scenes": bool(args.distinct_scenes),
        "split": args.split,
        "target_frame": args.frame,
        "file_type": args.file_type,
        "asset_source": args.asset_source,
        "num_exports": len(manifest),
        "exports": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(manifest)} transformed scene meshes to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
