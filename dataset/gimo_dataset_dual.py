import os

import pandas as pd
import torch
import torch.utils.data as data


def resolve_split_cache_path(base_dir, path_value, split):
    """Resolve a cache path, allowing one shared CLI value for both splits."""
    formatted = os.fspath(path_value).replace("{split}", split)
    if os.path.isabs(formatted):
        return os.path.normpath(formatted)
    return os.path.normpath(os.path.join(base_dir, formatted))


class EgoEvalDataset(data.Dataset):
    def __init__(self, config, train=False):
        folder = "SLICES_888s"
        split = "train" if train else "test"

        self.config = config
        self.train = train
        self.dataroot = config.dataroot
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.fps = config.fps

        self.sequences_path_list = []
        self.scenes_path_list = []
        self.trans_path_list = []
        self.poses_path_list = []
        self.start_end_list = []
        self.csv_row_list = []

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, "dataset.csv"))
        self.parse_data_info()

        base_dir = os.path.join(self.dataroot, folder, split)               #모든 .pth 파일의 공통 부모 폴더를 만듬
        self.cache_dir = base_dir
        gt_prior_name = f"gt_object_point_prior_{config.sample_points}.pth" #scene point수에 맞는 GT prior 파일 이름을 만듬
        object_points_name = getattr(config, "object_points_name", None) or "final_object_gt_points_1024.pth"
        object_mask_name = getattr(config, "object_mask_name", None)
        if object_mask_name is None:
            if not object_points_name.endswith(".pth"):
                raise ValueError(
                    f"Expected object_points_name to be a .pth file, got: {object_points_name}"
                )
            object_mask_name = object_points_name.replace(".pth", "_mask.pth")
        required_paths = {
            "objects": resolve_split_cache_path(base_dir, object_points_name, split),
            "object_mask": resolve_split_cache_path(base_dir, object_mask_name, split),
            "poses_input": os.path.join(base_dir, "poses_input.pth"),
            "poses_label": os.path.join(base_dir, "poses_label.pth"),
            "joints_input": os.path.join(base_dir, "joints_input.pth"),
            "joints_label": os.path.join(base_dir, "joints_label.pth"),
            "scene_points": os.path.join(base_dir, f"scene_points_{config.sample_points}.pth"),
        }
        optional_paths = {
            "gt_object_point_prior": os.path.join(base_dir, gt_prior_name),
            "gt_object_point_prior_mask": os.path.join(base_dir, gt_prior_name.replace(".pth", "_mask.pth")),
        }

        missing = [name for name, path in required_paths.items() if not os.path.exists(path)]
        if missing:
            missing_text = ", ".join(f"{name}={required_paths[name]}" for name in missing)
            raise FileNotFoundError("dual dataset cache is incomplete. Missing: " + missing_text)

        self.objects = torch.load(required_paths["objects"], weights_only=False)
        self.object_mask = torch.load(required_paths["object_mask"], weights_only=False).bool()
        self.poses_input = torch.load(required_paths["poses_input"], weights_only=False)
        self.poses_label = torch.load(required_paths["poses_label"], weights_only=False)
        self.joints_input = torch.load(required_paths["joints_input"], weights_only=False)
        self.joints_label = torch.load(required_paths["joints_label"], weights_only=False)
        self.scenes = torch.load(required_paths["scene_points"], weights_only=False)

        has_gt_prior = all(os.path.exists(path) for path in optional_paths.values())
        if has_gt_prior:
            self.gt_object_point_prior = torch.load(optional_paths["gt_object_point_prior"], weights_only=False).float()
            self.gt_object_point_prior_mask = torch.load(
                optional_paths["gt_object_point_prior_mask"],
                weights_only=False,
            ).bool()
        else:
            self.gt_object_point_prior = torch.zeros(
                (self.scenes.shape[0], self.scenes.shape[1]),
                dtype=torch.float32,
            )
            self.gt_object_point_prior_mask = torch.zeros((self.scenes.shape[0],), dtype=torch.bool)

        require_gt_prior = (
            getattr(config, "require_gt_object_point_prior", False)
            or getattr(config, "model_variant", "") == "tia_attcat_sca_gtprior"
        )
        if require_gt_prior and not has_gt_prior:
            missing_text = ", ".join(f"{name}={path}" for name, path in optional_paths.items())
            raise FileNotFoundError("GT prior cache is required for tia_attcat_sca_gtprior. Missing: " + missing_text)

        num_samples = self.poses_input.shape[0]
        if self.objects.ndim == 4:
            if self.objects.shape[0] != num_samples:
                raise ValueError(
                    f"Object tensor length {self.objects.shape[0]} does not match pose length {num_samples}"
                )
        elif self.objects.ndim == 3:
            if self.objects.shape[0] != num_samples:
                raise ValueError(
                    f"Object tensor length {self.objects.shape[0]} does not match pose length {num_samples}"
                )
            if self.objects.shape[-1] != 3:
                raise ValueError(
                    f"Expected point-cloud objects with last dim 3, got {self.objects.shape}"
                )
        else:
            raise ValueError(f"Unsupported objects tensor shape: {self.objects.shape}")
        if self.object_mask.shape[0] != num_samples:
            raise ValueError(
                f"Object mask length {self.object_mask.shape[0]} does not match pose length {num_samples}"
            )
        if self.gt_object_point_prior.shape[0] != num_samples:
            raise ValueError(
                f"GT prior length {self.gt_object_point_prior.shape[0]} does not match pose length {num_samples}"
            )
        if self.gt_object_point_prior.shape[1] != self.scenes.shape[1]:
            raise ValueError(
                f"GT prior point count {self.gt_object_point_prior.shape[1]} does not match scene point count {self.scenes.shape[1]}"
            )
        if self.gt_object_point_prior_mask.shape[0] != num_samples:
            raise ValueError(
                f"GT prior mask length {self.gt_object_point_prior_mask.shape[0]} does not match pose length {num_samples}"
            )

        self.objects = self.objects.clone()
        self.objects[~self.object_mask] = 0
        self.gt_object_point_prior = self.gt_object_point_prior.clone()
        self.gt_object_point_prior[~self.gt_object_point_prior_mask] = 0

    def __getitem__(self, index):
        scene = self.scenes_path_list[index]
        seq = self.sequences_path_list[index]

        objects = self.objects[index]
        object_mask = self.object_mask[index]
        gt_object_point_prior = self.gt_object_point_prior[index]
        gt_object_point_prior_mask = self.gt_object_point_prior_mask[index]
        poses_input = self.poses_input[index]
        poses_label = self.poses_label[index]
        joints_input = self.joints_input[index]
        joints_label = self.joints_label[index]
        scene_points = self.scenes[index]

        sample = (
            objects,
            object_mask,
            gt_object_point_prior,
            gt_object_point_prior_mask,
            poses_input,
            poses_label,
            joints_input,
            joints_label,
            scene_points,
            seq,
            scene,
        )
        return sample

    def __len__(self):
        return len(self.poses_path_list)

    def parse_data_info(self):
        for i, seq in enumerate(self.dataset_info["sequence_path"]):
            if self.dataset_info["training"][i] != self.train:
                continue
            self.csv_row_list.append(i)
            self.poses_path_list.append(self.dataset_info["start_frame"][i])
            self.sequences_path_list.append(seq)
            self.scenes_path_list.append(self.dataset_info["scene"][i])
            self.trans_path_list.append(self.dataset_info["transformation"][i])
            self.start_end_list.append(
                [self.dataset_info["start_frame"][i], self.dataset_info["end_frame"][i]]
            )


class EgoEvalTeacherKDDataset(EgoEvalDataset):
    """EgoEvalDataset variant that also loads teacher's object point cloud.

    Returns 12-tuple instead of base 10-tuple:
      (objects, object_mask, teacher_objects, teacher_object_mask,
       gt_object_point_prior, gt_object_point_prior_mask,
       poses_input, poses_label, joints_input, joints_label,
       scene_points, seq, scene)

    The first two slots remain student's (objects, object_mask). The two
    new teacher slots are inserted right after.
    """

    def __init__(self, config, train=False):
        super().__init__(config, train=train)

        folder = "SLICES_888s"
        split = "train" if train else "test"
        base_dir = os.path.join(self.dataroot, folder, split)

        teacher_object_points_name = getattr(config, "teacher_object_points_name", None)
        if not teacher_object_points_name:
            raise ValueError(
                "teacher_object_points_name is required for teacher KD training."
            )

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
