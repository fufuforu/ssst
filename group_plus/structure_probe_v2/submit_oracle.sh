#!/bin/bash
#SBATCH --job-name=sp2-oracle
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v2/logs/oracle-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v2/logs/oracle-%j.err
#
# §3 token-assignment oracle on the fixed sample (GT-assisted, per-scene fit).
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p group_plus/structure_probe_v2/logs
MODE="${1:-smoke}"
if [ "$MODE" = "smoke" ]; then
  python -u scripts/probe_token_assignment_oracle.py --smoke \
    --out group_plus/structure_probe_v2/oracle_smoke.json
else
  python -u scripts/probe_token_assignment_oracle.py \
    --out group_plus/structure_probe_v2/oracle_pair.json
fi
