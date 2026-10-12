#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove likely wall and ceiling faces from a textured OBJ while preserving UV/material references."
    )
    parser.add_argument("inputs", nargs="+", help="Input OBJ paths.")
    parser.add_argument("--wall-margin", type=float, default=0.12, help="Boundary margin in meters for wall removal.")
    parser.add_argument("--ceiling-margin", type=float, default=0.18, help="Top margin in meters for ceiling removal.")
    parser.add_argument(
        "--suffix",
        type=str,
        default="_no_walls_ceiling",
        help="Suffix appended before .obj for outputs.",
    )
    return parser.parse_args()


def parse_face_indices(face_tokens: list[str]) -> tuple[int, int, int]:
    indices = []
    for token in face_tokens[:3]:
        first = token.split("/")[0]
        indices.append(int(first))
    return indices[0], indices[1], indices[2]


def load_obj(path: Path):
    lines = path.read_text(encoding="latin1").splitlines(keepends=True)
    vertices: list[list[float]] = []
    face_line_indices: list[int] = []
    faces: list[tuple[int, int, int]] = []

    for line_index, line in enumerate(lines):
        if line.startswith("v "):
            parts = line.strip().split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.strip().split()[1:]
            if len(parts) != 3:
                continue
            face_line_indices.append(line_index)
            faces.append(parse_face_indices(parts))

    if not vertices or not faces:
        raise ValueError(f"Could not parse vertices/faces from {path}")

    return lines, np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), face_line_indices


def compute_remove_mask(vertices: np.ndarray, faces: np.ndarray, wall_margin: float, ceiling_margin: float) -> np.ndarray:
    tri = vertices[faces - 1]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-8, None)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    ceiling = (centers[:, 1] > (maxs[1] - ceiling_margin)) & (np.abs(normals[:, 1]) > 0.6)

    wall_x = (np.abs(normals[:, 1]) < 0.35) & (np.abs(normals[:, 0]) >= np.abs(normals[:, 2])) & (
        (centers[:, 0] < (mins[0] + wall_margin)) | (centers[:, 0] > (maxs[0] - wall_margin))
    )
    wall_z = (np.abs(normals[:, 1]) < 0.35) & (np.abs(normals[:, 2]) > np.abs(normals[:, 0])) & (
        (centers[:, 2] < (mins[2] + wall_margin)) | (centers[:, 2] > (maxs[2] - wall_margin))
    )

    return ceiling | wall_x | wall_z


def write_filtered_obj(
    input_path: Path,
    output_path: Path,
    lines: list[str],
    face_line_indices: list[int],
    remove_mask: np.ndarray,
) -> None:
    face_remove_lookup = {
        face_line_indices[idx]
        for idx, should_remove in enumerate(remove_mask.tolist())
        if should_remove
    }
    with output_path.open("w", encoding="latin1", newline="") as dst:
        for line_index, line in enumerate(lines):
            if line_index in face_remove_lookup:
                continue
            dst.write(line)


def process_one(path: Path, wall_margin: float, ceiling_margin: float, suffix: str) -> None:
    lines, vertices, faces, face_line_indices = load_obj(path)
    remove_mask = compute_remove_mask(vertices, faces, wall_margin, ceiling_margin)
    output_path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    write_filtered_obj(path, output_path, lines, face_line_indices, remove_mask)
    print(
        f"{path} -> {output_path} | removed_faces={int(remove_mask.sum())} "
        f"kept_faces={int((~remove_mask).sum())}"
    )


def main() -> None:
    args = parse_args()
    for item in args.inputs:
        process_one(Path(item), args.wall_margin, args.ceiling_margin, args.suffix)


if __name__ == "__main__":
    main()
