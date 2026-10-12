# Scene graphs and object selection

This package contains scene-graph preprocessing, teacher target generation,
and the same-space student selector. All inputs and outputs are external to
the source release; see [data preparation](../docs/data.md).

From the repository root:

```bash
python object_selector/preprocess_scene_graphs.py --help
python object_selector/scripts/export_ours_intent_teacher.py --help
python scripts/train_same_space_student_selector.py --help
python dataset/create_final_object_points_from_selector.py --help
```

The default input locations are `data/GIMO` and
`data/OpenIns3D/output/results_demo`. CLI arguments allow other locations.
Scene graphs are written under `object_selector/scene_graphs/`; generated
teacher targets and selector runs are ignored by Git.

`common/` defines schemas and vocabularies, `teacher/` contains scoring and
backend clients, `same_space/` provides the selector model and dataset, and
`training/` builds teacher targets. Some scripts also use the sibling
[`object_selector_new`](../object_selector_new/README.md) package.
