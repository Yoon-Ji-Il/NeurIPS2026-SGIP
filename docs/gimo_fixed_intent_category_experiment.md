# GIMO fixed-category intent classification

Input:

- `./dataset/dataset_3_with_want.csv`
- model input: `action` + `object`
- GT: `want`

The model must select exactly one of these 16 GIMO intent labels:

`sit, sleep, drink, eat, read, use, take, open, light, clean, dress, throw, cook, write, fix, unknown`

All requests use temperature 0, seed 0, thinking disabled, and an Ollama JSON
schema whose `selected_intent` field is an enum of the 16 allowed labels.

Start the locally installed Ollama server on GPU 0:

```bash
cd .
GPU_INDEX=0 bash scripts/start_local_ollama.sh
```

Download Phi-4 14B and Gemma-3 12B to `.ollama_models`:

```bash
bash scripts/pull_local_intent_models.sh
```

The local downloader intentionally does not pull Qwen3-14B.

Run Phi-4 14B and Gemma-3 12B sequentially:

```bash
cd .
bash scripts/run_phi4_gemma3_intent_categories_sequential.sh
```

Individual commands:

```bash
python scripts/run_remote_intent_category_classifier.py \
  --model-name phi4:14b \
  --output-dir runs/intent_category_models/phi4_14b

python scripts/run_remote_intent_category_classifier.py \
  --model-name gemma3:12b \
  --output-dir runs/intent_category_models/gemma3_12b
```

Each model run writes:

- `intent_predictions.csv`: all rows with Action, Object, GT, selection, correctness
- `unique_pair_predictions.csv`: 83 unique Action+Object pairs
- `video_name_intent_category.csv`: row-aligned loader-compatible output
- `metrics.json`: overall, train, and test accuracy
- `per_intent_metrics.csv`: precision, recall, and F1 for all 16 intents
- `confusion_matrix.csv`: 16×16 confusion matrix
- `predictions.jsonl`: resumable checkpoint

The combined comparison is written under:

`runs/intent_category_models/comparison/`

Run Qwen3-14B, Phi-4 14B, and Gemma-3 12B with the same prompt and compare all
three only if Qwen3-14B is already available in the local Ollama model store:

```bash
cd .
bash scripts/run_qwen14_phi4_gemma3_intent_categories_sequential.sh
```

The three-model comparison is written under:

`runs/intent_category_models/comparison_qwen14_phi4_gemma3/`

Additional comparison files:

- `intent_category_model_comparison.csv`: GT and all three predictions/reasons
- `intent_category_model_summary.csv`: overall, train, test, and macro-F1
- `pairwise_model_agreement.csv`: Qwen/Phi/Gemma pairwise agreement
- `model_disagreements.csv`: only rows where the three predictions differ
- `any_model_errors.csv`: rows where at least one model is wrong
