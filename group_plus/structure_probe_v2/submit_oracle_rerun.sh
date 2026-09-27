#!/bin/bash
#SBATCH --job-name=sp2-orac2
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v2/logs/oracle_rerun-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v2/logs/oracle_rerun-%j.err
# Determinism re-run of the §3 oracle: same seed/config, saves the final A (small)
# for the §4.B per-Gaussian initialisation, and lets the two curves be compared.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p group_plus/structure_probe_v2/logs
python -u scripts/probe_token_assignment_oracle.py \
  --out group_plus/structure_probe_v2/oracle_pair_rerun.json
