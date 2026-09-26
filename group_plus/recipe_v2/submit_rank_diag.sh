#!/bin/bash
#SBATCH --job-name=recipe2-rank
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/logs/recipe2_rank.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/logs/recipe2_rank.err
#
# Read-only TP-ranking diagnostic for the GT-free read-out (G0+ step6000 and
# recipe_v1 step6000).  No training, no checkpoint write.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/recipe_tp_rank_diag.py \
  --out group_plus/recipe_v2/tp_rank_diag.json
