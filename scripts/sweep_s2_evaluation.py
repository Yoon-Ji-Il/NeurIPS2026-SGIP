"""S2 inference-time sweep over a trained student's candidate_scores.csv.

This script does NOT retrain the student. It re-runs Stage-S2 with different
motion cue definitions, trigger modes, margin thresholds, and motion weights
on a single trained student checkpoint, and reports GT accuracy for each
configuration.

Usage:
    python scripts/sweep_s2_evaluation.py \
        --candidate-scores runs/student/loo/base/candidate_scores.csv \
        --output-dir runs/student/loo/base/s2_sweep
"""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-scores", type=Path, required=True,
                   help="Path to candidate_scores.csv from a trained student.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def load_samples(path: Path):
    samples = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samples[r["sample_id"]].append(r)
    return samples


def gt_index(group):
    for i, r in enumerate(group):
        if r["is_gt"] == "1":
            return i
    return -1


# ---------------------------------------------------------------------------
# Motion cue definitions
# ---------------------------------------------------------------------------
MOTION_CUES: dict[str, Callable[[dict], float]] = {
    "default":     lambda r: float(r["root_horizontal_alignment"]) + float(r["face_normal_alignment"]),
    "root_h_2x":   lambda r: 2.0 * float(r["root_horizontal_alignment"]) + float(r["face_normal_alignment"]),
    "face_2x":     lambda r: float(r["root_horizontal_alignment"]) + 2.0 * float(r["face_normal_alignment"]),
    "face_only":   lambda r: float(r["face_normal_alignment"]),
    "root_only":   lambda r: float(r["root_horizontal_alignment"]),
    "all_horiz":   lambda r: (float(r["root_horizontal_alignment"]) +
                              float(r["torso_horizontal_alignment"]) +
                              float(r["face_normal_horizontal_alignment"])),
}


def compute_top_k_indices(group, k):
    """Return top-k candidate indices by student_posterior."""
    posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
    posts.sort(key=lambda t: -t[1])
    return [i for i, _ in posts[:k]]


def get_collision_subset(group, top_idx):
    """Indices in top_idx whose name appears more than once."""
    name_counts = Counter(group[i]["name"] for i in top_idx)
    return [i for i in top_idx if name_counts[group[i]["name"]] >= 2]


def get_margin_subset(group, top_idx, threshold):
    """Indices in top_idx whose posterior is within `threshold` of top-1."""
    posts = [float(group[i]["student_posterior"]) for i in top_idx]
    top1 = max(posts)
    subset = [i for i, p in zip(top_idx, posts) if (top1 - p) <= threshold]
    if len(subset) < 2:
        return []
    return subset


def apply_s2(group, motion_fn, trigger, margin_threshold, alpha_m, top_k):
    top_idx = compute_top_k_indices(group, top_k)

    if trigger == "name_collision":
        active = get_collision_subset(group, top_idx)
    elif trigger == "score_margin":
        active = get_margin_subset(group, top_idx, margin_threshold)
    elif trigger == "either":
        c = set(get_collision_subset(group, top_idx))
        m = set(get_margin_subset(group, top_idx, margin_threshold))
        active = sorted(c | m, key=lambda j: -float(group[j]["student_posterior"]))
    else:
        raise ValueError(trigger)

    if len(active) < 2:
        # No trigger — plain argmax
        posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
        return max(posts, key=lambda t: t[1])[0]

    # Compute motion within active subset (minmax)
    m_raw = [motion_fn(group[j]) for j in active]
    m_min, m_max = min(m_raw), max(m_raw)
    if m_max - m_min > 1e-9:
        m_norm = {j: (motion_fn(group[j]) - m_min) / (m_max - m_min) for j in active}
    else:
        m_norm = {j: 0.0 for j in active}

    # Logit-space fusion: log s + alpha * m_norm  (only for active)
    scores = []
    for i, r in enumerate(group):
        s = float(r["student_posterior"])
        log_s = math.log(max(s, 1e-12))
        if i in m_norm:
            log_s = log_s + alpha_m * m_norm[i]
        scores.append((i, log_s))
    return max(scores, key=lambda t: t[1])[0]


def evaluate(samples, motion_name, motion_fn, trigger, margin_threshold, alpha_m, top_k):
    train_c, test_c = 0, 0
    train_t, test_t = 0, 0
    triggered = 0
    helped, hurt = 0, 0
    for sid, group in samples.items():
        gt_i = gt_index(group)
        if gt_i < 0:
            continue
        is_test = group[0]["split"] == "test"

        # Pre-S2 winner (plain argmax)
        posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
        pre_win = max(posts, key=lambda t: t[1])[0]
        # Post-S2 winner
        post_win = apply_s2(group, motion_fn, trigger, margin_threshold, alpha_m, top_k)

        if pre_win != post_win:
            triggered += 1
            if pre_win != gt_i and post_win == gt_i:
                helped += 1
            if pre_win == gt_i and post_win != gt_i:
                hurt += 1

        correct = (post_win == gt_i)
        if is_test:
            test_t += 1
            if correct:
                test_c += 1
        else:
            train_t += 1
            if correct:
                train_c += 1
    return {
        "motion": motion_name,
        "trigger": trigger,
        "margin": margin_threshold,
        "alpha": alpha_m,
        "test_c": test_c, "test_t": test_t,
        "train_c": train_c, "train_t": train_t,
        "triggered": triggered,
        "helped": helped, "hurt": hurt,
    }


def main():
    args = parse_args()
    samples = load_samples(args.candidate_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-S2 baseline
    pre_train_c, pre_test_c = 0, 0
    pre_train_t, pre_test_t = 0, 0
    for sid, group in samples.items():
        gt_i = gt_index(group)
        if gt_i < 0:
            continue
        posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
        winner = max(posts, key=lambda t: t[1])[0]
        is_test = group[0]["split"] == "test"
        correct = (winner == gt_i)
        if is_test:
            pre_test_t += 1
            if correct:
                pre_test_c += 1
        else:
            pre_train_t += 1
            if correct:
                pre_train_c += 1

    print(f"=== Pre-S2 baseline (plain argmax) ===")
    print(f"  test:  {pre_test_c}/{pre_test_t}")
    print(f"  train: {pre_train_c}/{pre_train_t}")
    print(f"  all:   {pre_test_c+pre_train_c}/{pre_test_t+pre_train_t}")
    print()

    # Sweep configurations
    triggers = ["name_collision", "score_margin", "either"]
    margins = [0.02, 0.05, 0.10, 0.15]
    alphas = [0.5, 1.0, 2.0, 3.0]

    results = []
    for motion_name, motion_fn in MOTION_CUES.items():
        for trigger in triggers:
            margin_iter = margins if trigger in ("score_margin", "either") else [0.0]
            for margin in margin_iter:
                for alpha in alphas:
                    r = evaluate(samples, motion_name, motion_fn, trigger,
                                 margin, alpha, args.top_k)
                    results.append(r)

    # Save full results
    out_csv = args.output_dir / "s2_sweep_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["motion", "trigger", "margin", "alpha",
                    "test_c", "test_t", "train_c", "train_t",
                    "triggered", "helped", "hurt", "net"])
        for r in results:
            w.writerow([r["motion"], r["trigger"], r["margin"], r["alpha"],
                        r["test_c"], r["test_t"], r["train_c"], r["train_t"],
                        r["triggered"], r["helped"], r["hurt"], r["helped"]-r["hurt"]])
    print(f"Saved {len(results)} configs to {out_csv}")
    print()

    # Top 15 by test
    results.sort(key=lambda r: (r["test_c"], r["helped"]-r["hurt"]), reverse=True)
    print(f"=== Top 15 configs by test GT ===")
    print(f"{'motion':<12} {'trigger':<16} {'margin':<8} {'alpha':<6} "
          f"{'test':<8} {'train':<10} {'trig':<5} {'help':<5} {'hurt':<5} {'net':<5}")
    for r in results[:15]:
        net = r["helped"] - r["hurt"]
        m_str = f"{r['margin']:.2f}" if r["trigger"] in ("score_margin", "either") else "-"
        print(f"{r['motion']:<12} {r['trigger']:<16} {m_str:<8} {r['alpha']:<6.1f} "
              f"{r['test_c']:>3}/{r['test_t']:<4} {r['train_c']:>3}/{r['train_t']:<6} "
              f"{r['triggered']:<5} {r['helped']:<5} {r['hurt']:<5} {net:<+5}")
    print()

    # Best per (motion, trigger) combo
    print(f"=== Best config per (motion, trigger) combo ===")
    by_combo = defaultdict(list)
    for r in results:
        by_combo[(r["motion"], r["trigger"])].append(r)
    print(f"{'motion':<12} {'trigger':<16} {'best test':<12} {'(margin,alpha)':<20}")
    for combo, runs in by_combo.items():
        best = max(runs, key=lambda r: r["test_c"])
        m_str = f"{best['margin']:.2f}" if best["trigger"] in ("score_margin", "either") else "-"
        print(f"{combo[0]:<12} {combo[1]:<16} {best['test_c']:>3}/{best['test_t']:<8} "
              f"({m_str}, α={best['alpha']:.1f})")


if __name__ == "__main__":
    main()
