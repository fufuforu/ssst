#!/bin/bash
#SBATCH --job-name=obj-lgs-ab
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_object_locusgs/logs/arm-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_object_locusgs/logs/arm-%j.err
#
# One arm (A or B) of the strictly paired object-aware LocusGS run.
#
#   sbatch object_locusgs/submit_arm.sh a
#   sbatch object_locusgs/submit_arm.sh b
#
# Both arms read the same pre-registered plan and the same source checkpoint.
set -euo pipefail
ARM="${1:-}"
if [[ "${ARM}" != "a" && "${ARM}" != "b" ]]; then
  echo "usage: sbatch $0 {a|b}" >&2
  exit 2
fi
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /space/mawb/ssst

mkdir -p "workspace_object_locusgs/arm_${ARM}"

python -u scripts/train_object_locusgs.py \
  --arm "${ARM}" \
  --plan object_locusgs/plan_6000.json \
  --split workspace_recon_diag/cross_scene/split.json \
  --source-ckpt workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step2000 \
  --source-sha256 f0e791b8bb9d49deeba160f0a5fcf75594e4a0638d79ed79b9021ccd6da31c6d \
  --steps 6000 --warmup 200 \
  --lr-existing 1e-5 --lr-new 1e-4 \
  --grad-clip 1.0 --weight-decay 0.05 \
  --seed 42 \
  --log-every 100 --eval-every 1000 \
  --save-steps 0 2000 6000 \
  --manifest-out "workspace_object_locusgs/arm_${ARM}/manifest.json" \
  --out-dir "workspace_object_locusgs/arm_${ARM}"
