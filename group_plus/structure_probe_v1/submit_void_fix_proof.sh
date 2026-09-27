#!/bin/bash
#SBATCH --job-name=sp-voidfix
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v1/logs/voidfix-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v1/logs/voidfix-%j.err
#
# Re-export the same pairs that produced the pre-fix smoke tree and prove the
# reported void counter changed while every PNG/JSON byte and every official
# metric stayed identical.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
W=workspace_group_plus/structure_probe_v1/void_fix_after
P=group_plus/structure_probe_v1

python -u scripts/group_official_export.py \
  --checkpoint workspace_group_plus/recipe_v1/run/ckpt_step6000 \
  --output "$W/recipe_v1" --head-mode legacy_prefix \
  --instance-outer-weight 0.1 --products both --limit 2 --smoke
python -u scripts/group_official_export.py \
  --checkpoint workspace_group_plus/arm_g0plus/ckpt_step6000 \
  --output "$W/g0plus" --head-mode legacy_prefix \
  --instance-outer-weight 0.05 --no-recipe --products both --limit 2 --smoke

python -u scripts/compare_void_fix.py \
  --before workspace_group_plus/implementation_audit_v1/B1_smoke \
  --after "$W" --out "$P/void_counter_fix.json"
