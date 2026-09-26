#!/bin/bash
#SBATCH --job-name=A-pure4-smoke
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/A_smoke-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/A_smoke-%j.err
#
# Condition A smoke: recipe_v2_pure4 vs the original recipe_v2 step-0 control.
# No long training happens here.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/train_group_locusgs.py \
  --arm g0 --recipe \
  --recipe-head-mode pure4 \
  --out-dir workspace_group_plus/implementation_audit_v1/recipe_v2_pure4/run \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --reference-init workspace_group_plus/recipe_v2/run/ckpt_step0 \
  --reference-head-mode legacy_prefix \
  --instance-outer-weight 0.05 \
  --assign-coef 0.2 --assign-every 1 \
  --smoke-coef 0.2 --smoke-every 1 \
  --steps 6000 --save-steps 0 3000 6000 \
  --no-eval \
  --smoke-only group_plus/implementation_audit_v1/A_smoke.json
