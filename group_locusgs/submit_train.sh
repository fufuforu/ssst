#!/bin/bash
#SBATCH --job-name=grp-lgs
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_group_locusgs/logs/train-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_group_locusgs/logs/train-%j.err
#
# One arm (g0 or g1) of the from-scratch group-feedback experiment.
#   sbatch group_locusgs/submit_train.sh g0
#   sbatch group_locusgs/submit_train.sh g1
set -euo pipefail
ARM="${1:-}"
if [[ "${ARM}" != "g0" && "${ARM}" != "g1" ]]; then
  echo "usage: sbatch $0 {g0|g1}" >&2
  exit 2
fi
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p "workspace_group_locusgs/arm_${ARM}"

python -u scripts/train_group_locusgs.py \
  --arm "${ARM}" \
  --plan object_locusgs/plan_6000.json \
  --split workspace_recon_diag/cross_scene/split.json \
  --steps 6000 --warmup 2000 --lr 1e-4 \
  --grad-clip 1.0 --weight-decay 0.05 --seed 42 \
  --log-every 100 --eval-every 500 \
  --save-steps 0 3000 6000 \
  --manifest-out "workspace_group_locusgs/arm_${ARM}/manifest.json" \
  --out-dir "workspace_group_locusgs/arm_${ARM}"
