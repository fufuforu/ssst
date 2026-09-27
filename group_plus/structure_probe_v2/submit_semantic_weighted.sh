#!/bin/bash
#SBATCH --job-name=sp2-semW
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v2/logs/semW-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v2/logs/semW-%j.err
# §5 single-variable semantic contrast: class-weighted NLL.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p group_plus/structure_probe_v2/logs
MODE="${1:-smoke}"
if [ "$MODE" = "smoke" ]; then
  python -u scripts/probe_semantic_weighted_nll.py --smoke \
    --out group_plus/structure_probe_v2/probe_semantic_weighted_smoke.json
else
  python -u scripts/probe_semantic_weighted_nll.py \
    --out group_plus/structure_probe_v2/probe_semantic_weighted.json
fi
