#!/bin/bash
#SBATCH --job-name=B1-full
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B1_full-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B1_full-%j.err
#
# B1 full export over the official 1860-pair manifest for recipe_v1 and G0+,
# then the pinned SIU3R evaluator on both products (panoptic = full metrics;
# semantic-only = mIoU only, so a semantic map cannot fabricate instances).
#
# Requires the B1 smoke to have passed; refuses to run if it did not.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
W=workspace_group_plus/implementation_audit_v1
P=group_plus/implementation_audit_v1
SIU3R_PY=/space/mawb/SIU3R/.venv_gpu_v4/bin/python

if [ ! -f "$P/B1_smoke_ok.json" ]; then
  echo "B1 smoke has not passed (missing $P/B1_smoke_ok.json); refusing to export" >&2
  exit 1
fi

for arm in recipe_v1 g0plus; do
  if [ "$arm" = "recipe_v1" ]; then
    CKPT=workspace_group_plus/recipe_v1/run/ckpt_step6000
    EXTRA="--instance-outer-weight 0.1"
  else
    CKPT=workspace_group_plus/arm_g0plus/ckpt_step6000
    EXTRA="--instance-outer-weight 0.05"
  fi
  python -u scripts/group_official_export.py \
    --checkpoint "$CKPT" --output "$W/B1_full/$arm" \
    --head-mode legacy_prefix $EXTRA \
    --products both
  du -sh "$W/B1_full/$arm"/* 2>/dev/null || true
done

for arm in recipe_v1 g0plus; do
  "$SIU3R_PY" scripts/invoke_siu3r_official_evaluator.py \
    --eval-path "$W/B1_full/$arm/official_predictions_panoptic" \
    --output "$W/B1_full/$arm/official_panoptic.json" \
    --device cuda
  "$SIU3R_PY" scripts/invoke_siu3r_official_evaluator.py \
    --eval-path "$W/B1_full/$arm/official_predictions_semantic" \
    --output "$W/B1_full/$arm/official_semantic.json" \
    --semantic-only --no-image-depth --device cuda
done
