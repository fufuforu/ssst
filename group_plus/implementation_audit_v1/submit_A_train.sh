#!/bin/bash
#SBATCH --job-name=A-pure4-train
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/A_train-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/A_train-%j.err
#
# Condition A: recipe_v2_pure4 (specification-correct group head) vs the original
# recipe_v2.  Only the head mode differs from the control; every other field is
# identical (same step-0 init, deep seed 1743, plan, seeds, optimizer, schedule,
# 0.05 instance weight, CE 0.2 per step, no stop-gradient).
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
A=group_plus/implementation_audit_v1
W=workspace_group_plus/implementation_audit_v1/recipe_v2_pure4

python -u scripts/train_group_locusgs.py \
  --arm g0 --recipe \
  --recipe-head-mode pure4 \
  --out-dir "$W/run" \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --instance-outer-weight 0.05 \
  --assign-coef 0.2 --assign-every 1 \
  --steps 6000 --save-steps 0 3000 6000 \
  --probe-steps 0 1 2 1499 1500 1501 2999 3000 3001 5999 6000 \
  --probe-windows "$A/conditionA_windows.json" \
  --probe-log "$W/probe_mapping.jsonl" \
  --manifest-out "$A/A_manifest.json"
