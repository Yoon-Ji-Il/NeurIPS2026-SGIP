# Teacher/student object selector

The teacher uses intention/semantic information and geometry to score object
candidates. The student learns from teacher distributions; the residual
reranker adjusts baseline scores. Feature choices and optional evaluation
tie-breaks are controlled by each script's arguments.

| Directory | Purpose |
| --- | --- |
| `common/` | Input schemas and action/intention taxonomies |
| `teacher/` | Semantic scoring, geometric fusion, local/remote backends |
| `student/` | Student model, feature normalization, candidate batching |
| `residual/` | Residual reranker and input feature construction |
| `scripts/` | Input construction, teacher export, training, analysis |

Top-level compatibility modules are retained because research scripts import
them. The `teacher/`, `student/`, and `residual/` packages contain the main
implementations.

## Workflow

Run from the repository root after installing the dependencies for your chosen
backend. Input data and pretrained models are not bundled.

1. Prepare scene graphs and matching dataset metadata, then build samples:

   ```bash
   python -m object_selector_new.scripts.build_samples_jsonl \
     --scene-graph-root object_selector/scene_graphs \
     --dataset-csv /path/to/dataset_with_want.csv \
     --output-jsonl data/samples.jsonl
   ```

2. Export teacher scores. Use `--help` to choose the backend and feature columns.
   A mock backend is available for format checks:

   ```bash
   python -m object_selector_new.scripts.export_teacher_csv \
     --input-jsonl data/samples.jsonl \
     --output-csv runs/teacher/candidates.csv \
     --llm-backend mock
   ```

3. Train the student, then the residual reranker as needed:

   ```bash
   python -m object_selector_new.scripts.train_student \
     --input-csv runs/teacher/candidates.csv --output-dir runs/student
   python -m object_selector_new.scripts.train_residual --help
   ```

For motion-model inputs, export the selection summary and use the appropriate
object-point exporter in `dataset/`. Preserve the metadata/cache row ordering.

Keep API credentials in environment variables or a local `.env` file, which
is ignored by Git. Ollama examples default to `http://127.0.0.1:11434`; set
`OLLAMA_BASE_URL` or the corresponding CLI option for your own server.
