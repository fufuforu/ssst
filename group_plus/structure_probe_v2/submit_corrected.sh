#!/bin/bash
#SBATCH --job-name=sp2-corr
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v2/logs/corrected-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v2/logs/corrected-%j.err
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p group_plus/structure_probe_v2/logs
python -u scripts/structure_probe_v2_corrected.py \
  --out group_plus/structure_probe_v2/corrected_metrics.json
