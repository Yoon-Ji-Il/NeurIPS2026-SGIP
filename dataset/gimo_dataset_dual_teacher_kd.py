import os

import torch

from dataset.gimo_dataset_dual import (
    EgoEvalDataset as BaseEgoEvalDataset,
    resolve_split_cache_path,
)


class EgoEvalTeacherKDDataset(BaseEgoEvalDataset):
    def __init__(self, config, train=False):
        super().__init__(config, train=train)

        folder = "SLICES_888s"
        split = "train" if train else "test"
        base_dir = os.path.join(self.dataroot, folder, split)

        teacher_object_points_name = getattr(config, "teacher_object_points_name", None)
        if not teacher_object_points_name:
            raise ValueError("teacher_object_points_name is required for teacher KD training.")
        teacher_object_mask_name = getattr(config, "teacher_object_mask_name", None)
        if teacher_object_mask_name is None:
            if not teacher_object_points_name.endswith(".pth"):
                raise ValueError(
                    f"Expected teacher_object_points_name to be a .pth file, got: {teacher_object_points_name}"
                )
            teacher_object_mask_name = teacher_object_points_name.replace(".pth", "_mask.pth")

        teacher_object_path = resolve_split_cache_path(
            base_dir, teacher_object_points_name, split
        )
        teacher_object_mask_path = resolve_split_cache_path(
            base_dir, teacher_object_mask_name, split
        )
        if not os.path.exists(teacher_object_path):
            raise FileNotFoundError(f"Missing teacher object points: {teacher_object_path}")
        if not os.path.exists(teacher_object_mask_path):
            raise FileNotFoundError(f"Missing teacher object mask: {teacher_object_mask_path}")

        self.teacher_objects = torch.load(teacher_object_path, weights_only=False)
        self.teacher_object_mask = torch.load(teacher_object_mask_path, weights_only=False).bool()

        num_samples = self.poses_input.shape[0]
        if self.teacher_objects.shape[0] != num_samples:
            raise ValueError(
                f"Teacher object tensor length {self.teacher_objects.shape[0]} does not match pose length {num_samples}"
            )
        if self.teacher_object_mask.shape[0] != num_samples:
            raise ValueError(
                f"Teacher object mask length {self.teacher_object_mask.shape[0]} does not match pose length {num_samples}"
            )

        self.teacher_objects = self.teacher_objects.clone()
        self.teacher_objects[~self.teacher_object_mask] = 0

    def __getitem__(self, index):
        base_sample = super().__getitem__(index)
        return (
            base_sample[0],
            base_sample[1],
            self.teacher_objects[index],
            self.teacher_object_mask[index],
            *base_sample[2:],
        )
