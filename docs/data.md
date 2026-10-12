# Data and external models

All paths below are local and ignored by Git. Nothing in this repository
automatically downloads a dataset or model weight.

## GIMO cache contract

`dataset/gimo_dataset_dual.py` expects:

```text
data/GIMO/
  dataset.csv
  SLICES_888s/
    train/
      poses_input.pth
      poses_label.pth
      joints_input.pth
      joints_label.pth
      scene_points_32768.pth
      final_object_qwen3_14b_motion_c_final_points_1024.pth
      final_object_qwen3_14b_motion_c_final_points_1024_mask.pth
    test/
      (the same filenames for the test split)
```

Object cache names are configurable with `--object_points_name` and
`--object_mask_name`. Without an explicit mask name, `_mask.pth` replaces
`.pth`. `{split}` in either path is replaced with `train` or `test`.
`--sample_points` controls the scene-point filename.

`gt_object_point_prior_<sample_points>.pth` and its `_mask.pth` companion
are optional for the default loader, but required when
`--require_gt_object_point_prior` is selected.

The CSV, tensor row ordering, coordinate system, and split must match.
The release contains loaders and object-cache exporters, not the original
prepared GIMO tensor dataset or a verified raw-GIMO-to-cache pipeline.

## Object selection inputs

`object_selector/preprocess_scene_graphs.py` reads the GIMO metadata, motion
caches, and OpenIns3D inventory under
`data/OpenIns3D/output/results_demo/<scene>/label_pointcloud_export.json`.
See its `--help` for path overrides.

The selector exporter produces teacher CSVs. Student/reranker scripts consume
these CSVs and write local outputs. `dataset/create_final_object_points_from_selector.py`
converts selection summaries into object-point caches.

`data/samples.jsonl` is not shipped. Build it with
`object_selector_new/scripts/build_samples_jsonl.py`, supplying the scene graph
root and matching metadata with `--dataset-csv`.

## Body models

The GIMO model requires a complete VPoser v1.0 directory and the neutral SMPL-X
NPZ model. Obtain them separately through the upstream projects listed in the
README. Point `--vposer_path` at the complete snapshot directory and
`--smplx_path` at the parent body-model directory, typically:

```text
data/body_models/smplx/SMPLX_NEUTRAL.npz
vposer_v1_0/
  (upstream snapshot configuration, Python definition, and weights)
```

The research workspace's VPoser definition contains a local rotation-conversion
compatibility adjustment. That external snapshot is not redistributed here.
Validate the installed VPoser/torchgeometry combination before attempting full
reproduction; upstream defaults have not been compared numerically here.

## HIK, HOI-M3, and PROX

Additional scripts default to `data/HIK`, `data/HOIm3`, and `data/PROX`.
Their cache layouts and protocols differ from GIMO. Read the corresponding
configuration and preprocessing script before running them; these directories
are not interchangeable. Local transformer weights default to `pretrained_models/`.
