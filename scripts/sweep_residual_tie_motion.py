"""Sweep score-tie motion reranking over a residual export.

This does not retrain anything. It evaluates inference-time motion reranking
only on samples whose top posterior candidates are within a small tie margin.

Usage:
    python scripts/sweep_residual_tie_motion.py \
        --candidate-scores runs/residual/feature_group_sweep_v1/b2_only/best_gt_export/candidate_scores.csv \
        --output-dir runs/residual/feature_group_sweep_v1/b2_only/tie_motion_sweep
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-scores", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--join-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional CSV(s) to join by (split, sample_id, object_id) for extra motion columns.",
    )
    p.add_argument("--top-k", type=int, nargs="+", default=[2, 3])
    return p.parse_args()


def merge_aux_features(rows, join_paths):
    if not join_paths:
        return rows

    merged = [dict(row) for row in rows]
    row_by_key = {
        (row["split"], row["sample_id"], row["object_id"]): row for row in merged
    }

    for path in join_paths:
        with path.open(encoding="utf-8") as f:
            for aux in csv.DictReader(f):
                key = (aux["split"], aux["sample_id"], aux["object_id"])
                dst = row_by_key.get(key)
                if dst is None:
                    continue
                for col, value in aux.items():
                    if col in ("split", "sample_id", "object_id"):
                        continue
                    if col not in dst or dst[col] == "":
                        dst[col] = value
    return merged


def load_samples(path: Path, join_paths):
    samples = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = merge_aux_features(rows, join_paths)
    for row in rows:
            samples[row["sample_id"]].append(row)
    return samples


def gt_index(group):
    for i, row in enumerate(group):
        if row["is_gt"] == "1":
            return i
    return -1


def w(weights: dict[str, float]) -> Callable[[dict], float]:
    def f(row):
        score = 0.0
        for col, weight in weights.items():
            if col == "face_line_inv_1p5":
                d = float(row.get("face_normal_line_distance_xz", 0.0))
                score += weight * (1.0 / (1.0 + d) ** 1.5)
            elif col == "body_h":
                rh = float(row["root_horizontal_alignment"])
                th = float(row["torso_horizontal_alignment"])
                score += weight * (rh + th) / 2.0
            else:
                score += weight * float(row[col])
        return score

    return f


MOTION_CUES = {
    "root_only": w({"root_horizontal_alignment": 1.0}),
    "all_horiz": w(
        {
            "root_horizontal_alignment": 1.0,
            "torso_horizontal_alignment": 1.0,
            "face_normal_horizontal_alignment": 1.0,
        }
    ),
    "default_root_face": w(
        {
            "root_horizontal_alignment": 1.0,
            "face_normal_alignment": 1.0,
        }
    ),
    "teacher_t1": w(
        {
            "root_horizontal_alignment": 1.0,
            "face_normal_alignment": 0.1,
            "face_line_inv_1p5": 0.05,
        }
    ),
    "teacher_t3": w(
        {
            "root_horizontal_alignment": 5.0,
            "body_h": 0.25,
            "face_normal_alignment": 0.5,
            "face_line_inv_1p5": 0.25,
        }
    ),
}


def compute_top_k_indices(group, top_k):
    posts = [(i, float(row["student_posterior"])) for i, row in enumerate(group)]
    posts.sort(key=lambda item: -item[1])
    return [i for i, _ in posts[:top_k]]


def get_margin_subset(group, top_idx, threshold):
    posts = [float(group[i]["student_posterior"]) for i in top_idx]
    top1 = max(posts)
    subset = [i for i, p in zip(top_idx, posts) if (top1 - p) <= threshold]
    if len(subset) < 2:
        return []
    return subset


def apply_tie_motion(group, motion_fn, margin_threshold, alpha_m, top_k):
    top_idx = compute_top_k_indices(group, top_k)
    active = get_margin_subset(group, top_idx, margin_threshold)
    if len(active) < 2:
        posts = [(i, float(row["student_posterior"])) for i, row in enumerate(group)]
        return max(posts, key=lambda item: item[1])[0], False, active

    raw = [motion_fn(group[i]) for i in active]
    raw_min, raw_max = min(raw), max(raw)
    if raw_max - raw_min > 1e-9:
        motion_norm = {
            i: (motion_fn(group[i]) - raw_min) / (raw_max - raw_min) for i in active
        }
    else:
        motion_norm = {i: 0.0 for i in active}

    fused = []
    for i, row in enumerate(group):
        p = float(row["student_posterior"])
        score = math.log(max(p, 1e-12))
        if i in motion_norm:
            score += alpha_m * motion_norm[i]
        fused.append((i, score))
    winner = max(fused, key=lambda item: item[1])[0]
    pre = max(
        ((i, float(row["student_posterior"])) for i, row in enumerate(group)),
        key=lambda item: item[1],
    )[0]
    changed = winner != pre
    return winner, changed, active


def init_bucket():
    return {
        "correct": 0,
        "total": 0,
        "triggered": 0,
        "changed": 0,
        "helped": 0,
        "hurt": 0,
        "pre_hits": 0,
        "pre_misses": 0,
        "miss_covered": 0,
        "hit_touched": 0,
    }


def evaluate(samples, motion_name, motion_fn, margin_threshold, alpha_m, top_k):
    buckets = {"train": init_bucket(), "test": init_bucket()}
    for _, group in samples.items():
        gt_i = gt_index(group)
        if gt_i < 0:
            continue
        split = group[0]["split"]
        bucket = buckets[split]

        posts = [(i, float(row["student_posterior"])) for i, row in enumerate(group)]
        pre_win = max(posts, key=lambda item: item[1])[0]
        pre_correct = pre_win == gt_i

        top_idx = compute_top_k_indices(group, top_k)
        active = get_margin_subset(group, top_idx, margin_threshold)
        triggered = len(active) >= 2

        if pre_correct:
            bucket["pre_hits"] += 1
        else:
            bucket["pre_misses"] += 1

        if triggered:
            bucket["triggered"] += 1
            if (not pre_correct) and (gt_i in active):
                bucket["miss_covered"] += 1
            if pre_correct and (pre_win in active):
                bucket["hit_touched"] += 1

        post_win, changed, _ = apply_tie_motion(
            group, motion_fn, margin_threshold, alpha_m, top_k
        )
        post_correct = post_win == gt_i

        bucket["total"] += 1
        if post_correct:
            bucket["correct"] += 1
        if changed:
            bucket["changed"] += 1
        if (not pre_correct) and post_correct:
            bucket["helped"] += 1
        if pre_correct and (not post_correct):
            bucket["hurt"] += 1

    return {
        "motion": motion_name,
        "margin": margin_threshold,
        "alpha": alpha_m,
        "top_k": top_k,
        "train": buckets["train"],
        "test": buckets["test"],
    }


def baseline_stats(samples):
    buckets = {"train": init_bucket(), "test": init_bucket()}
    for _, group in samples.items():
        gt_i = gt_index(group)
        if gt_i < 0:
            continue
        split = group[0]["split"]
        bucket = buckets[split]
        posts = [(i, float(row["student_posterior"])) for i, row in enumerate(group)]
        win = max(posts, key=lambda item: item[1])[0]
        bucket["total"] += 1
        if win == gt_i:
            bucket["correct"] += 1
            bucket["pre_hits"] += 1
        else:
            bucket["pre_misses"] += 1
    return buckets


def safe_rate(num, den):
    if den == 0:
        return 0.0
    return num / den


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.candidate_scores, args.join_csv)

    base = baseline_stats(samples)
    print("=== Residual Baseline (pre tie-motion) ===")
    print(
        f"  test:  {base['test']['correct']}/{base['test']['total']} "
        f"(misses={base['test']['pre_misses']}, hits={base['test']['pre_hits']})"
    )
    print(
        f"  train: {base['train']['correct']}/{base['train']['total']} "
        f"(misses={base['train']['pre_misses']}, hits={base['train']['pre_hits']})"
    )
    print(
        f"  all:   {base['test']['correct'] + base['train']['correct']}/"
        f"{base['test']['total'] + base['train']['total']}"
    )
    print()

    margins = [0.01, 0.02, 0.03, 0.05]
    alphas = [0.3, 0.5, 0.8, 1.0]

    results = []
    for motion_name, motion_fn in MOTION_CUES.items():
        for top_k in args.top_k:
            for margin in margins:
                for alpha in alphas:
                    results.append(
                        evaluate(samples, motion_name, motion_fn, margin, alpha, top_k)
                    )

    out_csv = args.output_dir / "tie_motion_sweep_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "motion",
                "top_k",
                "margin",
                "alpha",
                "test_c",
                "test_t",
                "train_c",
                "train_t",
                "test_triggered",
                "train_triggered",
                "test_changed",
                "train_changed",
                "test_helped",
                "test_hurt",
                "train_helped",
                "train_hurt",
                "test_miss_covered",
                "test_pre_misses",
                "test_hit_touched",
                "test_pre_hits",
                "train_miss_covered",
                "train_pre_misses",
                "train_hit_touched",
                "train_pre_hits",
            ]
        )
        for r in results:
            test = r["test"]
            train = r["train"]
            writer.writerow(
                [
                    r["motion"],
                    r["top_k"],
                    r["margin"],
                    r["alpha"],
                    test["correct"],
                    test["total"],
                    train["correct"],
                    train["total"],
                    test["triggered"],
                    train["triggered"],
                    test["changed"],
                    train["changed"],
                    test["helped"],
                    test["hurt"],
                    train["helped"],
                    train["hurt"],
                    test["miss_covered"],
                    test["pre_misses"],
                    test["hit_touched"],
                    test["pre_hits"],
                    train["miss_covered"],
                    train["pre_misses"],
                    train["hit_touched"],
                    train["pre_hits"],
                ]
            )

    print(f"Saved {len(results)} configs to {out_csv}")
    print()

    results.sort(
        key=lambda r: (
            r["test"]["correct"],
            -(r["test"]["hurt"]),
            r["test"]["helped"] - r["test"]["hurt"],
            r["train"]["correct"],
        ),
        reverse=True,
    )

    print("=== Top 20 configs by test GT, then low hurt ===")
    print(
        f"{'motion':<18} {'k':<3} {'margin':<6} {'alpha':<5} "
        f"{'test':<8} {'train':<10} {'t_h/h':<9} {'tr_h/h':<9} "
        f"{'misscov':<10} {'hittouch':<10}"
    )
    for r in results[:20]:
        test = r["test"]
        train = r["train"]
        print(
            f"{r['motion']:<18} {r['top_k']:<3} {r['margin']:<6.2f} {r['alpha']:<5.1f} "
            f"{test['correct']:>3}/{test['total']:<4} {train['correct']:>3}/{train['total']:<6} "
            f"{str(test['helped']) + '/' + str(test['hurt']):<9} "
            f"{str(train['helped']) + '/' + str(train['hurt']):<9} "
            f"{str(test['miss_covered']) + '/' + str(test['pre_misses']):<10} "
            f"{str(test['hit_touched']) + '/' + str(test['pre_hits']):<10}"
        )

    print()
    print("=== Best config per motion cue ===")
    best_by_motion = defaultdict(list)
    for r in results:
        best_by_motion[r["motion"]].append(r)
    for motion, runs in best_by_motion.items():
        best = max(
            runs,
            key=lambda r: (
                r["test"]["correct"],
                -(r["test"]["hurt"]),
                r["test"]["helped"] - r["test"]["hurt"],
                r["train"]["correct"],
            ),
        )
        test = best["test"]
        train = best["train"]
        print(
            f"  {motion:<18} test={test['correct']}/{test['total']} "
            f"train={train['correct']}/{train['total']} "
            f"k={best['top_k']} m={best['margin']:.2f} a={best['alpha']:.1f} "
            f"test helped/hurt={test['helped']}/{test['hurt']} "
            f"misscov={test['miss_covered']}/{test['pre_misses']} "
            f"hittouch={test['hit_touched']}/{test['pre_hits']}"
        )


if __name__ == "__main__":
    main()
