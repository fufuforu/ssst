#!/bin/bash
#SBATCH --job-name=recipe2-eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/logs/recipe2_eval.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/logs/recipe2_eval.err
#
# Final (read-only) evaluation of the recipe_v2 step6000 checkpoint with the
# pre-registered acceptance criteria, reusing scripts/eval_recipe_v1.py.
# Also reruns the TP-ranking diagnostic over G0+ / recipe_v1 / recipe_v2.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/eval_recipe_v1.py \
  --checkpoint workspace_group_plus/recipe_v2/run/ckpt_step6000 \
  --baseline group_plus/recipe_v1/baseline.json \
  --history workspace_group_plus/recipe_v2/run/val_history.jsonl \
  --out-dir group_plus/recipe_v2

python -u scripts/recipe_v2_three_way.py

python -u scripts/recipe_tp_rank_diag.py \
  --recipe-v2 workspace_group_plus/recipe_v2/run/ckpt_step6000 \
  --out group_plus/recipe_v2/tp_rank_diag.json

python -u scripts/recipe_v1_token_diag.py \
  --checkpoint workspace_group_plus/recipe_v2/run/ckpt_step6000 \
  --out group_plus/recipe_v2/token_diag.json
