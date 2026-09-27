#!/bin/bash
#SBATCH --job-name=sp-probe
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v1/logs/probe-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v1/logs/probe-%j.err
#
# Phase 2: isolated tiny-sample capacity probes (Probe-S / Probe-I).
# Saves only head deltas; never writes a model/optimizer checkpoint.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p group_plus/structure_probe_v1/logs
MODE="${1:-smoke}"
if [ "$MODE" = "smoke" ]; then
  python -u scripts/probe_group_capacity.py --smoke \
    --out-dir group_plus/structure_probe_v1
else
  python -u scripts/probe_group_capacity.py \
    --out-dir group_plus/structure_probe_v1
fi
