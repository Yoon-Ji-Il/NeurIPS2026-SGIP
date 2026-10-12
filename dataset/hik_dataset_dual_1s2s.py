"""Strict cache loader for the native HIK H25F50 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EgoEvalDataset(Dataset):
    def __init__(self, config, train=False):
        if config.input_seq_len != 25 or config.output_seq_len != 50:
            raise ValueError("Native HIK requires 25 input + 50 future frames")
        self.config = config
        self.train = bool(train)
        self.split = "train" if self.train else "test"
        self.cache_dir = (
            Path(config.dataroot) / "SLICES_888s" / self.split
        )
        object_name = config.object_points_name
        mask_name = (
            config.object_mask_name
            or object_name.replace(".pth", "_mask.pth")
        )
        paths = {
            "objects": self.cache_dir / object_name,
            "object_mask": self.cache_dir / mask_name,
            "poses_input": self.cache_dir / "poses_input.pth",
            "poses_label": self.cache_dir / "poses_label.pth",
            "joints_input": self.cache_dir / "joints_input.pth",
            "joints_label": self.cache_dir / "joints_label.pth",
            "scenes": self.cache_dir / f"scene_points_{config.sample_points}.pth",
            "manifest": self.cache_dir / "hik_h25f50_cache_manifest.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "HIK H25F50 cache is incomplete: " + ", ".join(missing)
            )
        self.objects = torch.load(paths["objects"], weights_only=False).float()
        self.object_mask = torch.load(
            paths["object_mask"], weights_only=False
        ).bool()
        self.poses_input = torch.load(
            paths["poses_input"], weights_only=False
        ).float()
        self.poses_label = torch.load(
            paths["poses_label"], weights_only=False
        ).float()
        self.joints_input = torch.load(
            paths["joints_input"], weights_only=False
        ).float()
        self.joints_label = torch.load(
            paths["joints_label"], weights_only=False
        ).float()
        self.scenes = torch.load(paths["scenes"], weights_only=False).float()
        self.manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        count = len(self.manifest)
        tensors = {
            "objects": (self.objects, (count, 1024, 3)),
            "object_mask": (self.object_mask, (count,)),
            "poses_input": (self.poses_input, (count, 25, 38)),
            "poses_label": (self.poses_label, (count, 50, 38)),
            "joints_input": (self.joints_input, (count, 25, 23, 3)),
            "joints_label": (self.joints_label, (count, 50, 23, 3)),
            "scenes": (self.scenes, (count, 1000, 3)),
        }
        for name, (tensor, expected) in tensors.items():
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} != {expected}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
        self.objects = self.objects.clone()
        self.objects[~self.object_mask] = 0
        self.gt_object_point_prior = torch.zeros(
            (count, config.sample_points),
            dtype=torch.float32,
        )
        self.gt_object_point_prior_mask = torch.zeros(
            (count,),
            dtype=torch.bool,
        )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest[index]
        return (
            self.objects[index],
            self.object_mask[index],
            self.gt_object_point_prior[index],
            self.gt_object_point_prior_mask[index],
            self.poses_input[index],
            self.poses_label[index],
            self.joints_input[index],
            self.joints_label[index],
            self.scenes[index],
            row["sample_id"],
            row["kitchen"],
        )
