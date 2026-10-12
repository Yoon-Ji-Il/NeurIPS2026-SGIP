"""Strict split-aligned cache loader for native HIK 29-joint forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EgoEvalDataset(Dataset):
    """Load direct joints separately while reusing the existing scene/object cache."""

    def __init__(self, config, train=False):
        if config.input_seq_len != 25 or config.output_seq_len != 50:
            raise ValueError("Direct29 HIK requires 25 input + 50 future frames")
        self.split = "train" if bool(train) else "test"
        context_dir = Path(config.dataroot) / "SLICES_888s" / self.split
        direct_dir = Path(config.direct29_root) / "SLICES_888s" / self.split
        object_name = config.object_points_name
        mask_name = (
            config.object_mask_name
            or object_name.replace(".pth", "_mask.pth")
        )
        paths = {
            "objects": context_dir / object_name,
            "object_mask": context_dir / mask_name,
            "scenes": context_dir / f"scene_points_{config.sample_points}.pth",
            "context_manifest": context_dir / "hik_h25f50_cache_manifest.json",
            "joints_input": direct_dir / "joints29_input.pth",
            "joints_label": direct_dir / "joints29_label.pth",
            "orient_input": direct_dir / "orient_input.pth",
            "orient_label": direct_dir / "orient_label.pth",
            "direct_manifest": direct_dir / "hik_direct29_cache_manifest.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "HIK direct29 cache is incomplete: " + ", ".join(missing)
            )

        self.objects = torch.load(
            paths["objects"], weights_only=False
        ).float()
        self.object_mask = torch.load(
            paths["object_mask"], weights_only=False
        ).bool()
        self.scenes = torch.load(paths["scenes"], weights_only=False).float()
        self.joints_input = torch.load(
            paths["joints_input"], weights_only=False
        ).float()
        self.joints_label = torch.load(
            paths["joints_label"], weights_only=False
        ).float()
        self.orient_input = torch.load(
            paths["orient_input"], weights_only=False
        ).float()
        self.orient_label = torch.load(
            paths["orient_label"], weights_only=False
        ).float()
        self.manifest = json.loads(
            paths["direct_manifest"].read_text(encoding="utf-8")
        )
        context_manifest = json.loads(
            paths["context_manifest"].read_text(encoding="utf-8")
        )
        direct_ids = [row["sample_id"] for row in self.manifest]
        context_ids = [row["sample_id"] for row in context_manifest]
        if direct_ids != context_ids:
            raise ValueError(
                f"Direct29/context manifest order differs for {self.split}"
            )

        count = len(self.manifest)
        expected = {
            "objects": (self.objects, (count, 1024, 3)),
            "object_mask": (self.object_mask, (count,)),
            "scenes": (self.scenes, (count, 1000, 3)),
            "joints_input": (self.joints_input, (count, 25, 29, 3)),
            "joints_label": (self.joints_label, (count, 50, 29, 3)),
            "orient_input": (self.orient_input, (count, 25, 3)),
            "orient_label": (self.orient_label, (count, 50, 3)),
        }
        for name, (tensor, shape) in expected.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} != {shape}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
        self.objects = self.objects.clone()
        self.objects[~self.object_mask] = 0

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest[index]
        return (
            self.objects[index],
            self.object_mask[index],
            self.joints_input[index],
            self.joints_label[index],
            self.orient_input[index],
            self.orient_label[index],
            self.scenes[index],
            row["sample_id"],
            row["kitchen"],
        )
