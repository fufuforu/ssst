#!/bin/bash
#SBATCH --job-name=grp-lgs-eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_group_locusgs/logs/eval-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_group_locusgs/logs/eval-%j.err
#
# Final paired development evaluation (per-scene rows, GT-free AP50, purity and
# token/group diagnostics, RGB + mask tiles).
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
STEP="${1:-6000}"

python -u scripts/eval_group_locusgs.py \
  --run "g0=workspace_group_locusgs/arm_g0/ckpt_step${STEP}" \
  --run "g1=workspace_group_locusgs/arm_g1/ckpt_step${STEP}" \
  --out "workspace_group_locusgs/eval_step${STEP}.json" \
  --images "workspace_group_locusgs/images"
