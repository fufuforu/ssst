#!/bin/bash
#SBATCH --job-name=B1-smoke
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B1_smoke-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/B1_smoke-%j.err
#
# B1 smoke: real-model export of the first official manifest pair for both arms,
# plus the official-reader semantic-only evaluation of that single pair.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
W=workspace_group_plus/implementation_audit_v1

for arm in recipe_v1 g0plus; do
  if [ "$arm" = "recipe_v1" ]; then
    CKPT=workspace_group_plus/recipe_v1/run/ckpt_step6000
    W_EXTRA="--instance-outer-weight 0.1"
    RCPT="--recipe"
  else
    CKPT=workspace_group_plus/arm_g0plus/ckpt_step6000
    W_EXTRA="--instance-outer-weight 0.05 --no-recipe"
    RCPT="--no-recipe"
  fi
  python -u scripts/group_official_export.py \
    --checkpoint "$CKPT" --output "$W/B1_smoke/$arm" \
    --head-mode legacy_prefix $W_EXTRA \
    --products both \
    --limit 2 --smoke
done

du -sh "$W/B1_smoke"/* 2>/dev/null

# official reader on the first pair of each arm (semantic-only product)
SIU3R_PY=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
for arm in recipe_v1 g0plus; do
  "$SIU3R_PY" scripts/invoke_siu3r_official_evaluator.py \
    --eval-path "$W/B1_smoke/$arm/official_predictions_semantic" \
    --output "$W/B1_smoke/$arm/official_semantic_smoke.json" \
    --semantic-only --no-image-depth --device cuda || true
done
python -u scripts/check_b1_smoke.py --root "$W/B1_smoke" \
  --out group_plus/implementation_audit_v1/B1_smoke_ok.json
