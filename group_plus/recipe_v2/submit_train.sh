#!/bin/bash
#SBATCH --job-name=recipe2-train
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/logs/recipe2_run.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/logs/recipe2_run.err
#
# recipe_v2 = recipe_v1 with the main instance/group outer weight 0.1 -> 0.05.
# Cold start from the same arm_g0/ckpt_step0, same plan, same seeds, same
# optimizer/schedule; every other field is identical (config_diff.json asserts
# exactly one differing field).
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/train_group_locusgs.py \
  --arm g0 --recipe \
  --out-dir workspace_group_plus/recipe_v2/run \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --instance-outer-weight 0.05 \
  --assign-coef 0.2 --assign-every 1 \
  --steps 6000 --save-steps 0 3000 6000 \
  --manifest-out group_plus/recipe_v2/manifest.json
