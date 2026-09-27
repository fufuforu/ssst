#!/bin/bash
#SBATCH --job-name=sp-voidoff
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/structure_probe_v1/logs/voidoff-%j.log
#SBATCH --error=/space/mawb/ssst/group_plus/structure_probe_v1/logs/voidoff-%j.err
set -euo pipefail
cd /space/mawb/ssst
/space/mawb/SIU3R/.venv_gpu_v4/bin/python -u scripts/compare_void_fix.py \
  --before workspace_group_plus/implementation_audit_v1/B1_smoke \
  --after workspace_group_plus/structure_probe_v1/void_fix_after \
  --out group_plus/structure_probe_v1/void_counter_fix.json
