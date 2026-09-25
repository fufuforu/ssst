#!/bin/bash
#SBATCH --job-name=obj-lgs-eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_object_locusgs/logs/eval-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_object_locusgs/logs/eval-%j.err
#
# Final development-set evaluation (with the GT-assisted purity diagnostic) plus
# the single official 2+4 pairing shape/leakage smoke.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

STEP="${1:-6000}"

python -u scripts/eval_object_locusgs.py \
  --run "a=workspace_object_locusgs/arm_a/ckpt_step${STEP}" \
  --run "b=workspace_object_locusgs/arm_b/ckpt_step${STEP}" \
  --out "workspace_object_locusgs/eval_step${STEP}.json" \
  --images "workspace_object_locusgs/images" \
  --purity

python -u scripts/smoke_official_pair.py \
  --checkpoint "workspace_object_locusgs/arm_b/ckpt_step${STEP}" \
  --arm b \
  --report "workspace_object_locusgs/official_pair_smoke.json"
