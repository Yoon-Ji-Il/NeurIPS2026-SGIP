# GIMO VLM intent agreement ablation

## Question

How stable are the categorical GIMO intents selected by
Qwen3-VL-8B-Instruct when the model is replaced with:

1. the smaller Qwen3-VL-4B-Instruct model; and
2. a similarly sized cross-family VLM, Gemma-3-4B-IT?

The primary metric is exact categorical intent agreement with the same-prompt
Qwen3-VL-8B-Instruct run. It is not token overlap, phrase similarity, object
selection accuracy, or visual grounding accuracy.

## Controlled setup

- Input: the same 215 GIMO rows and 83 unique Action+Object pairs.
- Taxonomy: `sit, sleep, drink, eat, read, use, take, open, light, clean,
  dress, throw, cook, write, fix, unknown`.
- Prompt version: `gimo_fixed_16_intent_zero_shot_v1`.
- Decoding: temperature 0, seed 0, thinking disabled.
- Runtime: the same local Ollama server.
- Quantization: Q4_K_M for all three variants.
- Images: none. This is a text-only Action+Object intent ablation using VLM
  backbones, not a visual-grounding experiment.

Exact model tags:

- `qwen3-vl:8b-instruct-q4_K_M`
- `qwen3-vl:4b-instruct-q4_K_M`
- `gemma3:4b-it-q4_K_M`

## Reproduce

```bash
cd .

GPU_INDEX=4 bash scripts/start_local_ollama.sh
bash scripts/pull_local_vlm_intent_models.sh
bash scripts/run_qwen3vl_vlm_intent_ablation.sh
```

Runs are resumable from each model's `predictions.jsonl`. A completed model is
unloaded from VRAM before the next model starts; its local model files remain.

## Results

| Model | Dataset-want accuracy | Agreement with 8B (rows) | Agreement with 8B (unique pairs) | Cohen's kappa |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 79.07% | 215/215 (100.00%) | 83/83 (100.00%) | 1.0000 |
| Qwen3-VL-4B-Instruct | 79.07% | 195/215 (90.70%) | 70/83 (84.34%) | 0.8914 |
| Gemma-3-4B-IT | 66.98% | 183/215 (85.12%) | 60/83 (72.29%) | 0.8265 |

Qwen3-VL-4B changed 20 row-level labels relative to the 8B reference. Nine
changes corrected an 8B error, nine changed an 8B-correct result into an
error, and two changed one wrong label into another wrong label. This explains
why its dataset-want accuracy equals the 8B result despite non-perfect
agreement.

Gemma-3-4B changed 32 row-level labels relative to the 8B reference. Two
changes corrected an 8B error, 28 changed an 8B-correct result into an error,
and two changed one wrong label into another wrong label.

The largest Qwen3-VL-4B composition changes were fewer `take` predictions
(-8 rows) and more `unknown` predictions (+7 rows). Gemma-3-4B predicted
`take` more often (+11 rows) and `sit` less often (-9 rows).

## Output files

The report directory is:

`runs/vlm_intent_ablation/report/`

- `vlm_intent_ablation_report.md`: paper-oriented result summary.
- `vlm_intent_agreement_summary.csv`: main agreement and accuracy metrics.
- `vlm_intent_composition.csv`: all 16 label counts, rates, and deltas.
- `vlm_per_intent_agreement.csv`: agreement conditioned on the 8B intent.
- `vlm_intent_predictions_comparison.csv`: all 215 aligned predictions.
- `vlm_intent_disagreements.csv`: rows where the three models differ.
- `vlm_pairwise_intent_agreement.csv`: row and unique-pair agreement matrix.

## Reporting language

Recommended wording:

> Under an identical deterministic 16-way GIMO intent prompt,
> Qwen3-VL-4B retained 90.70% row-level and 84.34% unique-pair intent
> agreement with Qwen3-VL-8B. A similarly sized cross-family VLM,
> Gemma-3-4B, retained 85.12% row-level and 72.29% unique-pair agreement.
> The 4B Qwen variant mainly shifted `take` predictions toward `unknown`,
> whereas Gemma-3-4B over-produced `take` and under-produced `sit`.

Do not describe these numbers as image grounding accuracy: no image was
provided in this ablation.
