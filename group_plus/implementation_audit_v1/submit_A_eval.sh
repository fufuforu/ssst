#!/bin/bash
#SBATCH --job-name=A-pure4-eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/A_eval-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/A_eval-%j.err
#
# Condition A final evaluation: recipe_v2_pure4 on the same 8 unseen windows with
# the corrected harness, then the paired table against G0+ / recipe_v1 / recipe_v2.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
P=group_plus/implementation_audit_v1

python -u scripts/eval_recipe_v1.py \
  --checkpoint workspace_group_plus/implementation_audit_v1/recipe_v2_pure4/run/ckpt_step6000 \
  --out-dir "$P/eval_A_pure4" --label recipe_v2_pure4 \
  --head-mode pure4 --instance-outer-weight 0.05 \
  --history workspace_group_plus/implementation_audit_v1/recipe_v2_pure4/run/val_history.jsonl

python -u scripts/audit_final_table.py \
  g0plus="$P/eval_g0plus" \
  recipe_v1="$P/eval_recipe_v1" \
  recipe_v2="$P/eval_recipe_v2" \
  recipe_v2_pure4="$P/eval_A_pure4" \
  --out "$P/final_table.json" --csv "$P/final_table_per_instance.csv"
