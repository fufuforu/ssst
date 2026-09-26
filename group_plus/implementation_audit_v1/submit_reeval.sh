#!/bin/bash
#SBATCH --job-name=impl-reeval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/reeval-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/reeval-%j.err
#
# Corrected (read-only) re-evaluation of G0+ / recipe_v1 / recipe_v2 with the
# fixed harness.  Historical arms keep the legacy_prefix head mode.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
A=group_plus/implementation_audit_v1

python -u scripts/eval_recipe_v1.py \
  --checkpoint workspace_group_plus/arm_g0plus/ckpt_step6000 \
  --out-dir "$A/eval_g0plus" --label g0plus \
  --no-recipe --head-mode legacy_prefix --instance-outer-weight 0.05

python -u scripts/eval_recipe_v1.py \
  --checkpoint workspace_group_plus/recipe_v1/run/ckpt_step6000 \
  --out-dir "$A/eval_recipe_v1" --label recipe_v1 \
  --head-mode legacy_prefix --instance-outer-weight 0.1 \
  --paired-delta-source "$A/eval_g0plus/eval_per_instance.csv" \
  --history workspace_group_plus/recipe_v1/run/val_history.jsonl

python -u scripts/eval_recipe_v1.py \
  --checkpoint workspace_group_plus/recipe_v2/run/ckpt_step6000 \
  --out-dir "$A/eval_recipe_v2" --label recipe_v2 \
  --head-mode legacy_prefix --instance-outer-weight 0.05 \
  --history workspace_group_plus/recipe_v2/run/val_history.jsonl

python -u scripts/audit_reeval_compare.py
