#!/bin/bash
#SBATCH --job-name=recipe2-smoke
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/logs/recipe2_smoke.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/logs/recipe2_smoke.err
#
# recipe_v2 single-variable smoke: step-0 parameter/forward equality against the
# recipe_v1 step-0 checkpoint + the fixed-window training smoke (<=20 updates).
# No long training happens here.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/train_group_locusgs.py \
  --arm g0 --recipe \
  --out-dir workspace_group_plus/recipe_v2/run \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --reference-init workspace_group_plus/recipe_v1/run/ckpt_step0 \
  --instance-outer-weight 0.05 \
  --assign-coef 0.2 --assign-every 1 \
  --smoke-coef 0.2 --smoke-every 1 \
  --steps 6000 --save-steps 0 3000 6000 \
  --no-eval \
  --smoke-only group_plus/recipe_v2/smoke.json
