"""Strict split-aligned cache loader for HOI-M3 HIK-scale direct24."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EgoEvalDataset(Dataset):
    def __init__(self, config, train=False):
        if (
            config.input_seq_len,
            config.output_seq_len,
            config.fps,
            config.sample_points,
        ) != (30, 60, 30, 1000):
            raise ValueError("HOI-M3 direct24 requires H30F60 at 30 FPS and P1000")
        self.split = "train" if bool(train) else "test"
        split_dir = Path(config.dataroot) / "SLICES_888s" / self.split
        object_name = config.object_points_name
        object_mask_name = (
            config.object_mask_name
            or object_name.replace(".pth", "_mask.pth")
        )
        paths = {
            "objects": split_dir / object_name,
            "object_mask": split_dir / object_mask_name,
            "joints_input": split_dir / "joints24_input.pth",
            "joints_label": split_dir / "joints24_label.pth",
            "orient_input": split_dir / "orient_input.pth",
            "orient_label": split_dir / "orient_label.pth",
            "scenes": split_dir / "scene_points_1000.pth",
            "manifest": split_dir / "hoim3_hikscale_cache_manifest.json",
        }
        if config.use_kd:
            if not config.teacher_object_points_name:
                raise ValueError("--teacher_object_points_name is required with --use_kd")
            teacher_mask_name = (
                config.teacher_object_mask_name
                or config.teacher_object_points_name.replace(".pth", "_mask.pth")
            )
            paths["teacher_objects"] = split_dir / config.teacher_object_points_name
            paths["teacher_object_mask"] = split_dir / teacher_mask_name
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "HOI-M3 direct24 cache is incomplete: " + ", ".join(missing)
            )

        self.objects = torch.load(paths["objects"], weights_only=False).float()
        self.object_mask = torch.load(
            paths["object_mask"], weights_only=False
        ).bool()
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
        self.scenes = torch.load(paths["scenes"], weights_only=False).float()
        self.manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        self.teacher_objects = None
        self.teacher_object_mask = None
        if config.use_kd:
            self.teacher_objects = torch.load(
                paths["teacher_objects"], weights_only=False
            ).float()
            self.teacher_object_mask = torch.load(
                paths["teacher_object_mask"], weights_only=False
            ).bool()

        count = len(self.manifest)
        expected = {
            "objects": (self.objects, (count, 1024, 3)),
            "object_mask": (self.object_mask, (count,)),
            "joints_input": (self.joints_input, (count, 30, 24, 3)),
            "joints_label": (self.joints_label, (count, 60, 24, 3)),
            "orient_input": (self.orient_input, (count, 30, 3)),
            "orient_label": (self.orient_label, (count, 60, 3)),
            "scenes": (self.scenes, (count, 1000, 3)),
        }
        if self.teacher_objects is not None:
            expected["teacher_objects"] = (
                self.teacher_objects,
                (count, 1024, 3),
            )
            expected["teacher_object_mask"] = (
                self.teacher_object_mask,
                (count,),
            )
        for name, (tensor, shape) in expected.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")

        self.objects = self.objects.clone()
        self.objects[~self.object_mask] = 0
        if self.teacher_objects is not None:
            self.teacher_objects = self.teacher_objects.clone()
            self.teacher_objects[~self.teacher_object_mask] = 0

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest[index]
        sample = {
            "objects": self.objects[index],
            "object_mask": self.object_mask[index],
            "joints_input": self.joints_input[index],
            "joints_label": self.joints_label[index],
            "orient_input": self.orient_input[index],
            "orient_label": self.orient_label[index],
            "scenes": self.scenes[index],
            "sample_id": row["sample_id"],
            "sequence": row["sequence"],
        }
        if self.teacher_objects is not None:
            sample["teacher_objects"] = self.teacher_objects[index]
            sample["teacher_object_mask"] = self.teacher_object_mask[index]
        return sample
