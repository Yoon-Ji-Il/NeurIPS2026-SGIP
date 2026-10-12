#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python}"

BASE_CSV="runs/student/loo/base/candidate_scores_dedup.csv"
SUPPORT_CSV="object_selector/same_space_feature_inputs/teacher_candidate_scores_support_context_geometry.csv"
MOTION_CSV="object_selector/same_space_feature_inputs/teacher_candidate_scores_input_motion_temporal_attention6.csv"
CONF_CSV="object_selector/same_space_feature_inputs/teacher_candidate_scores_support_rank_baseline_confidence.csv"
LOCAL19_CSV="object_selector/same_space_feature_inputs/teacher_candidate_scores_support_rank_input_motion_local19.csv"
OUT_ROOT="runs/residual/followup_combos_v1"

COMMON_ARGS=(
    --baseline-candidate-csv "$BASE_CSV"
    --feature-csv "$SUPPORT_CSV"
    --feature-csv "$MOTION_CSV"
    --feature-csv "$CONF_CSV"
    --feature-csv "$LOCAL19_CSV"
    --hidden-dim 128
    --max-residual 0.6
    --dropout 0.0
    --epochs 200
    --lr 1e-3
    --weight-decay 1e-4
    --lambda-kl 1.0
    --lambda-anchor 0.7
    --lambda-residual-l2 0.015
    --lambda-rank 0.7
    --lambda-teacher-ce 0.8
    --seed 7
)

B2="baseline_student_score,baseline_student_posterior"
CONE_PAIR="input_torso_object_cone_fraction,input_head_object_cone_fraction"
CHILD_SMALL="object_support_child_on_top_overlap_sum,object_support_max_child_score"
CHILD_VOL="object_support_child_on_top_volume_ratio_sum"
ROLE_FLOOR_SMALL="support_role_has_parent,support_role_has_child,support_role_is_stack_middle,support_role_is_isolated,support_floor_gap,support_floor_contact,support_floor_contact_x_child_score"
CONF_SMALL="baseline_top1_posterior_margin,baseline_distribution_norm_entropy,baseline_candidate_posterior_gap_to_top1,baseline_selected_confidence"
APPROACH_ALIGN_SMALL="input_root_approach_delta,input_root_motion_alignment,input_torso_object_alignment,input_head_object_alignment"

declare -a RUNS=(
    "cone_pair | $B2,$CONE_PAIR"
    "cone_pair_child_small | $B2,$CONE_PAIR,$CHILD_SMALL"
    "cone_pair_child_small_vol | $B2,$CONE_PAIR,$CHILD_SMALL,$CHILD_VOL"
    "cone_pair_role_floor | $B2,$CONE_PAIR,$ROLE_FLOOR_SMALL"
    "cone_pair_conf_small | $B2,$CONE_PAIR,$CONF_SMALL"
    "cone_pair_role_floor_conf | $B2,$CONE_PAIR,$ROLE_FLOOR_SMALL,$CONF_SMALL"
    "cone_pair_approach_align | $B2,$CONE_PAIR,$APPROACH_ALIGN_SMALL"
    "cone_pair_child_role_floor | $B2,$CONE_PAIR,$CHILD_SMALL,$ROLE_FLOOR_SMALL"
)

mkdir -p "$OUT_ROOT"

echo "Reference single-feature runs (existing):"
$PYTHON - <<'PY'
import json
from pathlib import Path
refs = [
    ("torso_cone_only", Path("runs/residual/single_feature_sweep_v1/b2_plus_input_torso_object_cone_fraction/summary.json")),
    ("head_cone_only", Path("runs/residual/single_feature_sweep_v1/b2_plus_input_head_object_cone_fraction/summary.json")),
    ("child_overlap_only", Path("runs/residual/single_feature_sweep_v1/b2_plus_object_support_child_on_top_overlap_sum/summary.json")),
]
print(f"{'ref':<20} {'test_gt':<8} {'train_gt':<9} {'teacher_t':<10} {'teacher_tr':<10} {'epoch_g':<8}")
for name, path in refs:
    s = json.loads(path.read_text())
    print(
        f"{name:<20} "
        f"{str(s['best_gt_test_gt_exact_count']) + '/37':<8} "
        f"{str(s['best_gt_train_gt_exact_count']) + '/177':<9} "
        f"{str(s['best_gt_test_teacher_top1_count']) + '/37':<10} "
        f"{str(s['best_gt_train_teacher_top1_count']) + '/177':<10} "
        f"{s['best_gt_epoch']:<8}"
    )
PY
echo ""

for entry in "${RUNS[@]}"; do
    name=$(echo "$entry" | cut -d'|' -f1 | xargs)
    cols=$(echo "$entry" | cut -d'|' -f2- | xargs)
    out="$OUT_ROOT/$name"
    mkdir -p "$out"

    echo "==========================================="
    echo "[$(date +%H:%M:%S)] Running: $name"
    echo "  cols: $cols"
    echo "==========================================="

    $PYTHON scripts/train_same_space_residual_reranker.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "$out" \
        --feature-columns "$cols" \
        > "$out/train.log" 2>&1

    $PYTHON - <<PY
import json
from pathlib import Path
s = json.loads((Path("$out") / "summary.json").read_text())
print("  best_teacher_epoch=", s["best_epoch"])
print("  best_teacher test/train teacher=", f'{s["best_test_teacher_top1_count"]}/37', f'{s["best_train_teacher_top1_count"]}/177')
print("  best_teacher test/train gt=", f'{s["best_test_gt_exact_count"]}/37', f'{s["best_train_gt_exact_count"]}/177')
print("  best_gt_epoch=", s["best_gt_epoch"])
print("  best_gt test/train teacher=", f'{s["best_gt_test_teacher_top1_count"]}/37', f'{s["best_gt_train_teacher_top1_count"]}/177')
print("  best_gt test/train gt=", f'{s["best_gt_test_gt_exact_count"]}/37', f'{s["best_gt_train_gt_exact_count"]}/177')
PY
done

echo ""
echo "==========================================="
echo "FOLLOWUP COMBO SUMMARY"
echo "==========================================="
$PYTHON - <<'PY'
import json
from pathlib import Path

root = Path("runs/residual/followup_combos_v1")
print(f"{'run':<30} {'test_gt':<8} {'train_gt':<9} {'test_t':<8} {'train_t':<9} {'epoch_g':<8}")
rows = []
for path in root.glob("*/summary.json"):
    s = json.loads(path.read_text())
    rows.append(
        (
            s["best_gt_test_gt_exact_count"],
            s["best_gt_train_gt_exact_count"],
            s["best_gt_test_teacher_top1_count"],
            s["best_gt_train_teacher_top1_count"],
            s["best_gt_epoch"],
            path.parent.name,
        )
    )
rows.sort(reverse=True)
for test_gt, train_gt, test_t, train_t, epoch_g, name in rows:
    print(
        f"{name:<30} "
        f"{str(test_gt) + '/37':<8} "
        f"{str(train_gt) + '/177':<9} "
        f"{str(test_t) + '/37':<8} "
        f"{str(train_t) + '/177':<9} "
        f"{epoch_g:<8}"
    )
PY
