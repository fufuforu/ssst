#!/bin/bash
#SBATCH --job-name=obj-lgs-smoke
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_object_locusgs/logs/smoke-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_object_locusgs/logs/smoke-%j.err
#
# Pre-run smoke suite for the object-aware LocusGS A/B experiment.  No long
# training is started from this script.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /space/mawb/ssst

python -u scripts/smoke_object_locusgs.py \
  --out workspace_object_locusgs/smoke \
  --plan object_locusgs/plan_6000.json \
  --steps 3
