#!/bin/bash
#SBATCH --job-name=sp-attr
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v1/logs/attr-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v1/logs/attr-%j.err
#
# Phase 1: read-only attribution over the three fixed window groups.
# No optimizer.step, no checkpoint write.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p group_plus/structure_probe_v1/logs

if [ "${1:-full}" = "smoke" ]; then
  python -u scripts/structure_probe_attribution.py --smoke \
    --out group_plus/structure_probe_v1/attribution_smoke.json
else
  python -u scripts/structure_probe_attribution.py \
    --out group_plus/structure_probe_v1/attribution.json
fi
