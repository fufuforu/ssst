#!/bin/bash
#SBATCH --job-name=B2-stopgrad
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B2-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B2-%j.err
#
# Condition B2: the single variable is group_recipe_assign_stop_shared_grad.
# Control = recipe_v2_pure4 (condition A).  Same step-0 state, same head mode,
# same plan/seeds/optimizer/schedule/weights.  Fail-fast smoke first.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
P=group_plus/implementation_audit_v1
W=workspace_group_plus/implementation_audit_v1

python -u scripts/b2_smoke.py \
  --control-init "$W/recipe_v2_pure4/run/ckpt_step0" \
  --head-mode pure4 --instance-outer-weight 0.05 \
  --out "$P/B2_smoke.json"

python -u scripts/train_group_locusgs.py \
  --arm g0 --recipe \
  --recipe-head-mode pure4 \
  --assign-stop-shared-grad \
  --out-dir "$W/recipe_v2_pure4_stopgrad/run" \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --instance-outer-weight 0.05 \
  --assign-coef 0.2 --assign-every 1 \
  --steps 6000 --save-steps 6000 \
  --minimal-checkpoint \
  --manifest-out "$P/B2_manifest.json"

python -u scripts/eval_recipe_v1.py \
  --checkpoint "$W/recipe_v2_pure4_stopgrad/run/ckpt_step6000" \
  --out-dir "$P/eval_B2_stopgrad" --label recipe_v2_pure4_stopgrad \
  --head-mode pure4 --instance-outer-weight 0.05 --assign-stop-shared-grad \
  --history "$W/recipe_v2_pure4_stopgrad/run/val_history.jsonl"

python -u scripts/audit_final_table.py \
  g0plus="$P/eval_g0plus" \
  recipe_v2_pure4="$P/eval_A_pure4" \
  recipe_v2_pure4_stopgrad="$P/eval_B2_stopgrad" \
  --out "$P/B2_final_table.json" --csv "$P/B2_final_table_per_instance.csv"
