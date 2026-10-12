from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_dual import MotionFromGazeDualConfig, SMPLX_evalutor, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = MotionFromGazeDualConfig()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", type=str, default="best_*.pth")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/best_eval_results_via_train.txt",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/best_eval_summary_via_train.json",
    )
    return parser.parse_args()


def main() -> None:
    config = parse_args()
    seed_everything(config.seed, config.deterministic)

    run_dir = config.run_dir.resolve()
    checkpoints = sorted(run_dir.glob(config.checkpoint_pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {config.checkpoint_pattern!r} in {run_dir}")

    results_file = config.results_file or (run_dir / "best_eval_results_via_train.txt")
    summary_json = config.summary_json or (run_dir / "best_eval_summary_via_train.json")

    rows: list[dict[str, float | str]] = []
    with results_file.open("w", encoding="utf-8") as out:
        for checkpoint in checkpoints:
            config.load_model_dir = str(checkpoint)
            evaluator = SMPLX_evalutor(config)
            evaluator.model.eval()
            with torch.no_grad():
                metrics = evaluator.test(-1)
            row = {"checkpoint": checkpoint.name, **metrics}
            rows.append(row)

            header = f"===== Evaluating {checkpoint.relative_to(REPO_ROOT)} ====="
            print(header, flush=True)
            print(metrics, flush=True)
            out.write(header + "\n")
            out.write(json.dumps(row) + "\n")

        best = min(rows, key=lambda row: float(row["loss_total"]))
        print("===== Best Overall =====", flush=True)
        print(f"best_ckpt={best['checkpoint']}", flush=True)
        print(f"best_total={best['loss_total']}", flush=True)
        out.write("===== Best Overall =====\n")
        out.write(f"best_ckpt={best['checkpoint']}\n")
        out.write(f"best_total={best['loss_total']}\n")

    payload = {
        "results": rows,
        "best_overall": best,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
