#!/bin/bash
#SBATCH --job-name=impl-audit
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/audit-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/audit-%j.err
#
# Read-only runtime audit (sections 3..8 of the implementation-audit brief).
# No training, no optimizer.step, no checkpoint write.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/audit_implementation_v1.py \
  --out group_plus/implementation_audit_v1/runtime_diagnostics.json
