#!/bin/bash
#SBATCH --job-name=isv1
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/group_plus/instance_state_v1_repair/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/group_plus/instance_state_v1_repair/slurm-%j.err
#
# Usage:  sbatch scripts/submit_instance_state_v1.sh smoke
#         sbatch scripts/submit_instance_state_v1.sh paired        # C then E, 2000 steps
#         sbatch scripts/submit_instance_state_v1.sh full 5000
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
PHASE="${1:-smoke}"
if [ "$PHASE" = "full" ]; then
  python -u scripts/run_instance_state_v1.py --phase full --until-step "${2:-5000}" \
    --reports group_plus/instance_state_v1_repair \
    --run-root workspace_group_plus/instance_state_v1_repair
else
  python -u scripts/run_instance_state_v1.py --phase "$PHASE" \
    --reports group_plus/instance_state_v1_repair \
    --run-root workspace_group_plus/instance_state_v1_repair
fi
