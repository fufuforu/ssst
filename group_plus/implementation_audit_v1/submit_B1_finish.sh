#!/bin/bash
#SBATCH --job-name=B1-finish
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B1_finish-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B1_finish-%j.err
#
# B1 completion: export the G0+ arm (no recipe head) and run the pinned SIU3R
# evaluator on both arms' panoptic (full metrics) and semantic-only products.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
W=workspace_group_plus/implementation_audit_v1
P=group_plus/implementation_audit_v1
SIU3R_PY=/space/mawb/SIU3R/.venv_gpu_v4/bin/python

python -u scripts/group_official_export.py \
  --checkpoint workspace_group_plus/arm_g0plus/ckpt_step6000 \
  --output "$W/B1_full/g0plus" \
  --no-recipe --instance-outer-weight 0.05 \
  --products both
du -sh "$W/B1_full/g0plus"/* 2>/dev/null || true

for arm in recipe_v1 g0plus; do
  "$SIU3R_PY" scripts/invoke_siu3r_official_evaluator.py \
    --eval-path "$W/B1_full/$arm/official_predictions_panoptic" \
    --output "$P/B1_full/${arm}_official_panoptic.json" \
    --device cuda
  "$SIU3R_PY" scripts/invoke_siu3r_official_evaluator.py \
    --eval-path "$W/B1_full/$arm/official_predictions_semantic" \
    --output "$P/B1_full/${arm}_official_semantic.json" \
    --semantic-only --no-image-depth --device cuda
done
