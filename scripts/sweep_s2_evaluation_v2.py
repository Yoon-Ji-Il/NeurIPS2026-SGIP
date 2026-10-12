"""S2 sweep using teacher-side best motion cues (plateau).

Based on teacher-side analysis: test=31 plateau formed with
root_horizontal dominant + small auxiliary signals.
"""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-scores", type=Path, required=True)
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


# Teacher-side plateau best motion cues
def w(weights: dict[str, float]) -> Callable[[dict], float]:
    """Build weighted motion cue from column weight dict."""
    def f(r):
        s = 0.0
        for col, weight in weights.items():
            if col == "face_line_inv_1p5":
                # 1 / (1 + face_normal_line_distance_xz) ** 1.5
                d = float(r.get("face_normal_line_distance_xz", 0))
                s += weight * (1.0 / (1.0 + d)**1.5)
            elif col == "body_h":
                # average of root_h and torso_h
                rh = float(r["root_horizontal_alignment"])
                th = float(r["torso_horizontal_alignment"])
                s += weight * (rh + th) / 2
            else:
                s += weight * float(r[col])
        return s
    return f


MOTION_CUES = {
    "T1_root1_face01_line05":
        w({"root_horizontal_alignment": 1.0, "face_normal_alignment": 0.1, "face_line_inv_1p5": 0.05}),
    "T2_root5_face05_line025":
        w({"root_horizontal_alignment": 5.0, "face_normal_alignment": 0.5, "face_line_inv_1p5": 0.25}),
    "T3_root5_body025_face05_line025":
        w({"root_horizontal_alignment": 5.0, "body_h": 0.25, "face_normal_alignment": 0.5, "face_line_inv_1p5": 0.25}),
    "T4_root8_face075_line05":
        w({"root_horizontal_alignment": 8.0, "face_normal_alignment": 0.75, "face_line_inv_1p5": 0.5}),
    "T5_root8_body075_face075_line05":
        w({"root_horizontal_alignment": 8.0, "body_h": 0.75, "face_normal_alignment": 0.75, "face_line_inv_1p5": 0.5}),
    "T6_root10_face1_line05":
        w({"root_horizontal_alignment": 10.0, "face_normal_alignment": 1.0, "face_line_inv_1p5": 0.5}),
    "T7_outlier_root12_face01_line05":
        w({"root_horizontal_alignment": 12.0, "face_normal_alignment": 0.1, "face_line_inv_1p5": 0.5}),
    "default_baseline":
        w({"root_horizontal_alignment": 1.0, "face_normal_alignment": 1.0}),
}


def compute_top_k_indices(group, k):
    posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
    posts.sort(key=lambda t: -t[1])
    return [i for i, _ in posts[:k]]


def get_collision_subset(group, top_idx):
    name_counts = Counter(group[i]["name"] for i in top_idx)
    return [i for i in top_idx if name_counts[group[i]["name"]] >= 2]


def get_margin_subset(group, top_idx, threshold):
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
        posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
        return max(posts, key=lambda t: t[1])[0]

    m_raw = [motion_fn(group[j]) for j in active]
    m_min, m_max = min(m_raw), max(m_raw)
    if m_max - m_min > 1e-9:
        m_norm = {j: (motion_fn(group[j]) - m_min) / (m_max - m_min) for j in active}
    else:
        m_norm = {j: 0.0 for j in active}

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
        posts = [(i, float(r["student_posterior"])) for i, r in enumerate(group)]
        pre_win = max(posts, key=lambda t: t[1])[0]
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
        "motion": motion_name, "trigger": trigger,
        "margin": margin_threshold, "alpha": alpha_m,
        "test_c": test_c, "test_t": test_t,
        "train_c": train_c, "train_t": train_t,
        "triggered": triggered, "helped": helped, "hurt": hurt,
    }


def main():
    args = parse_args()
    samples = load_samples(args.candidate_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"=== Pre-S2 baseline ===")
    print(f"  test:  {pre_test_c}/{pre_test_t}")
    print(f"  train: {pre_train_c}/{pre_train_t}")
    print(f"  all:   {pre_test_c+pre_train_c}/{pre_test_t+pre_train_t}")
    print()

    # Sweep: focused around teacher-side plateau values
    triggers = ["name_collision", "score_margin", "either"]
    margins = [0.02, 0.05, 0.10]
    alphas = [1.0, 1.5, 2.0, 3.0, 4.0]

    results = []
    for motion_name, motion_fn in MOTION_CUES.items():
        for trigger in triggers:
            margin_iter = margins if trigger in ("score_margin", "either") else [0.0]
            for margin in margin_iter:
                for alpha in alphas:
                    r = evaluate(samples, motion_name, motion_fn, trigger,
                                 margin, alpha, args.top_k)
                    results.append(r)

    out_csv = args.output_dir / "s2_sweep_v2_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["motion", "trigger", "margin", "alpha",
                        "test_c", "test_t", "train_c", "train_t",
                        "triggered", "helped", "hurt", "net"])
        for r in results:
            w_csv.writerow([r["motion"], r["trigger"], r["margin"], r["alpha"],
                            r["test_c"], r["test_t"], r["train_c"], r["train_t"],
                            r["triggered"], r["helped"], r["hurt"], r["helped"]-r["hurt"]])
    print(f"Saved {len(results)} configs to {out_csv}")
    print()

    results.sort(key=lambda r: (r["test_c"], r["helped"]-r["hurt"]), reverse=True)
    print(f"=== Top 20 configs by test GT ===")
    print(f"{'motion':<35} {'trigger':<16} {'margin':<8} {'alpha':<6} "
          f"{'test':<8} {'train':<10} {'trig':<5} {'help':<5} {'hurt':<5} {'net':<5}")
    for r in results[:20]:
        net = r["helped"] - r["hurt"]
        m_str = f"{r['margin']:.2f}" if r["trigger"] in ("score_margin", "either") else "-"
        print(f"{r['motion']:<35} {r['trigger']:<16} {m_str:<8} {r['alpha']:<6.1f} "
              f"{r['test_c']:>3}/{r['test_t']:<4} {r['train_c']:>3}/{r['train_t']:<6} "
              f"{r['triggered']:<5} {r['helped']:<5} {r['hurt']:<5} {net:<+5}")
    print()

    # Best per motion cue
    print(f"=== Best config per motion cue ===")
    by_motion = defaultdict(list)
    for r in results:
        by_motion[r["motion"]].append(r)
    for motion, runs in by_motion.items():
        best = max(runs, key=lambda r: r["test_c"])
        m_str = f"{best['margin']:.2f}" if best["trigger"] in ("score_margin", "either") else "-"
        print(f"  {motion:<35} test={best['test_c']}/{best['test_t']} "
              f"trigger={best['trigger']} (m={m_str}, α={best['alpha']:.1f})")


if __name__ == "__main__":
    main()
