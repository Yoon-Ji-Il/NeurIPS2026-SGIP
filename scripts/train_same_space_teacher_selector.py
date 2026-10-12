from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_selector.same_space import (  # noqa: E402
    DEFAULT_SAME_SPACE_FEATURE_COLUMNS,
    DEFAULT_SOURCE_CSV,
    SameSpaceDataset,
    SameSpaceSelector,
    build_samples_from_csv,
    collate_same_space_samples,
    compute_feature_stats,
    normalize_samples,
    parse_feature_columns,
    save_feature_stats,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "object_selector" / "same_space_runs" / "teacher_no_want_mlp_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a no-want same-space teacher selector with an unmasked posterior."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=",".join(DEFAULT_SAME_SPACE_FEATURE_COLUMNS),
        help="Comma-separated numeric columns shared by teacher and student.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def candidate_cross_entropy(logits: torch.Tensor, valid_mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    return F.cross_entropy(masked_logits, target_idx)


@torch.no_grad()
def evaluate(model: SameSpaceSelector, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    gt_hits = 0

    for batch in loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        gt_idx = batch["gt_idx"].to(device)
        outputs = model(features, valid_mask)
        loss = candidate_cross_entropy(outputs["logits"], valid_mask, gt_idx)
        pred_idx = outputs["distribution"].argmax(dim=1)

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        gt_hits += int((pred_idx == gt_idx).sum().item())

    return {
        "loss": total_loss / max(total_samples, 1),
        "gt_accuracy": gt_hits / max(total_samples, 1),
        "gt_exact_count": gt_hits,
        "total_samples": total_samples,
    }


def export_predictions(
    *,
    model: SameSpaceSelector,
    raw_samples: list,
    normalized_samples: list,
    feature_columns: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
    posterior_temperature: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    candidate_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for raw_sample, norm_sample in zip(raw_samples, normalized_samples, strict=True):
        features = norm_sample.features.unsqueeze(0).to(device)
        valid_mask = torch.ones((1, norm_sample.features.shape[0]), dtype=torch.bool, device=device)
        outputs = model(features, valid_mask)
        logits = outputs["logits"][0].detach().cpu()
        posterior = torch.softmax(logits / max(float(posterior_temperature), 1.0e-6), dim=0)
        pred_idx = int(torch.argmax(posterior).item())

        for candidate_index, object_id in enumerate(raw_sample.candidate_ids):
            row = {
                "split": raw_sample.split,
                "sample_id": raw_sample.sample_id,
                "object_id": object_id,
                "name": raw_sample.candidate_names[candidate_index],
                "category": raw_sample.candidate_categories[candidate_index],
                "teacher_score": float(logits[candidate_index].item()),
                "teacher_posterior": float(posterior[candidate_index].item()),
                "teacher_rank": int(torch.argsort(posterior, descending=True).tolist().index(candidate_index) + 1),
                "is_teacher_selected": int(candidate_index == pred_idx),
                "is_gt": int(candidate_index == raw_sample.gt_idx),
            }
            for feature_offset, column in enumerate(feature_columns):
                row[column] = float(raw_sample.features[candidate_index, feature_offset].item())
            candidate_rows.append(row)

        sample_rows.append(
            {
                "split": raw_sample.split,
                "sample_id": raw_sample.sample_id,
                "selected_object_id": raw_sample.candidate_ids[pred_idx],
                "selected_object_name": raw_sample.candidate_names[pred_idx],
                "gt_object_id": raw_sample.gt_object_id,
                "gt_object_name": raw_sample.candidate_names[raw_sample.gt_idx],
                "num_candidates": len(raw_sample.candidate_ids),
                "selected_confidence": float(posterior[pred_idx].item()),
                "is_gt_exact": int(pred_idx == raw_sample.gt_idx),
            }
        )

    candidate_path = output_dir / "candidate_scores.csv"
    sample_path = output_dir / "sample_summary.csv"
    write_csv(candidate_path, candidate_rows)
    write_csv(sample_path, sample_rows)
    return candidate_rows, sample_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty csv: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = parse_feature_columns(args.feature_columns)
    set_seed(args.seed)
    device = choose_device(args.device)

    raw_samples = build_samples_from_csv(args.input_csv, feature_columns=feature_columns)
    train_raw = [sample for sample in raw_samples if sample.split == "train"]
    test_raw = [sample for sample in raw_samples if sample.split == "test"]
    if not train_raw or not test_raw:
        raise RuntimeError("Expected both train and test samples in the input csv")

    mean, std = compute_feature_stats(train_raw)
    stats_path = output_dir / "feature_stats.json"
    save_feature_stats(stats_path, feature_columns=feature_columns, mean=mean, std=std)

    normalized_train = normalize_samples(train_raw, mean, std)
    normalized_test = normalize_samples(test_raw, mean, std)

    train_loader = DataLoader(
        SameSpaceDataset(normalized_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_same_space_samples,
    )
    test_loader = DataLoader(
        SameSpaceDataset(normalized_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_same_space_samples,
    )

    model = SameSpaceSelector(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            gt_idx = batch["gt_idx"].to(device)
            outputs = model(features, valid_mask)
            loss = candidate_cross_entropy(outputs["logits"], valid_mask, gt_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = features.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_epoch_loss": epoch_loss / max(epoch_samples, 1),
            "train_gt_accuracy": train_metrics["gt_accuracy"],
            "test_gt_accuracy": test_metrics["gt_accuracy"],
            "test_gt_exact_count": test_metrics["gt_exact_count"],
        }
        history.append(record)

        if best_metrics is None or test_metrics["gt_accuracy"] > best_metrics["gt_accuracy"]:
            best_metrics = dict(test_metrics)
            best_metrics["epoch"] = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Teacher training did not produce a best checkpoint")

    checkpoint = {
        "model_state": best_state,
        "feature_columns": list(feature_columns),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "posterior_temperature": args.posterior_temperature,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")

    model.load_state_dict(best_state)
    normalized_all = normalize_samples(raw_samples, mean, std)
    _, sample_rows = export_predictions(
        model=model,
        raw_samples=raw_samples,
        normalized_samples=normalized_all,
        feature_columns=feature_columns,
        device=device,
        output_dir=output_dir,
        posterior_temperature=args.posterior_temperature,
    )

    summary = {
        "mode": "same_space_teacher_no_want",
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
        "feature_columns": list(feature_columns),
        "num_train_samples": len(train_raw),
        "num_test_samples": len(test_raw),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "posterior_temperature": args.posterior_temperature,
        "feature_stats_json": str(stats_path),
        "best_epoch": int(best_metrics["epoch"]),
        "best_test_gt_accuracy": float(best_metrics["gt_accuracy"]),
        "best_test_gt_exact_count": int(best_metrics["gt_exact_count"]),
        "best_test_total": int(best_metrics["total_samples"]),
        "history": history,
        "sample_summary_csv": str(output_dir / "sample_summary.csv"),
        "candidate_scores_csv": str(output_dir / "candidate_scores.csv"),
    }
    write_json(output_dir / "summary.json", summary)

    test_selected = [row for row in sample_rows if row["split"] == "test"]
    print(
        f"Best epoch {best_metrics['epoch']}: "
        f"test_gt={best_metrics['gt_exact_count']}/{best_metrics['total_samples']} "
        f"({best_metrics['gt_accuracy']:.4f})"
    )
    print(f"Exported {len(test_selected)} test predictions to {output_dir}")


if __name__ == "__main__":
    main()
