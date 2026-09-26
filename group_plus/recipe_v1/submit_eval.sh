#!/bin/bash
#SBATCH --job-name=recipe-eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/logs/recipe_eval4.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/logs/recipe_eval4.err
#
# Final (read-only) evaluation of the recipe-v1 step6000 checkpoint with the
# pre-registered acceptance criteria.  No training, no optimizer step, no
# threshold or data change.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/eval_recipe_v1.py \
  --checkpoint workspace_group_plus/recipe_v1/run/ckpt_step6000 \
  --baseline group_plus/recipe_v1/baseline.json \
  --out-dir group_plus/recipe_v1
