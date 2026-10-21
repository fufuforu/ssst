#!/bin/bash
#SBATCH --job-name=grp-lgs-audit
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_group_locusgs/logs/audit-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_group_locusgs/logs/audit-%j.err
#
# READ-ONLY score audit: no training, no checkpoint writes, thresholds untouched.
# Re-scores the existing G0/G1 step3000+step6000 checkpoints under the legacy
# sigmoid(no-object logit) score and the CE-consistent P(thing) score.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/audit_group_scores.py \
  --out workspace_group_locusgs/audit_scores.json
