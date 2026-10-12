# Research entry points

The workspace contains multiple generations of experiments. This release
preserves their import paths so that the original implementations remain
available. Filenames such as `old`, `v2`, and `third1423` are historical labels,
not a statement about which results belong in the paper.

| Family | Entry points |
| --- | --- |
| GIMO teacher | `train_dual.py`, `eval_dual.py`, `scripts/run_teacher_*.sh` |
| GIMO motion distillation | `train_dual.py --use_kd`, `train_dual_old_0351_kd_third1423.py`, other `train_dual_kd_*.py` variants |
| Object teacher | `object_selector/scripts/export_ours_intent_teacher.py`, `object_selector_new/scripts/export_teacher_csv.py` |
| Student object selector | `scripts/train_same_space_student_selector.py`, `object_selector_new/scripts/train_student.py` |
| Residual reranker | `scripts/train_same_space_residual_reranker.py`, `object_selector_new/scripts/train_residual.py` |
| HIK | `train_hik_1s2s.py`, `train_hik_direct29_1s2s.py` |
| HOI-M3 | `train_hoim3_direct24_1s2s.py` |
| Segmentation ablations | `scripts/create_*segmentation*.py`, `scripts/run_segmentation_noise_*.sh` |
| Intent ablations | `scripts/run_*intent*.py`, the intent experiment notes in this folder |

Research launchers encode their original GPU indices and hyperparameters.
Inspect these scripts before execution and adjust device selection for your
machine. Launchers that start multiple jobs require the corresponding devices.
Do not use scripts named `smoke`, `retry`, or `corrections` as an independent
paper benchmark definition.

Retained experiment notes describe historical runs. Referenced outputs,
checkpoints, and CSVs are not included, and their metrics were not rerun during
release preparation.
