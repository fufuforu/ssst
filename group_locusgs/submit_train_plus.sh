#!/bin/bash
#SBATCH --job-name=grp-lgs-plus
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-11,3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/logs/train-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/logs/train-%j.err
#
# G0+ : G0 plus background-slot pixel supervision (single variable).
# Starts from G0's own step-0 checkpoint (model + optimizer + RNG), verified
# block-by-block before the first update.  6000 steps, peak lr 1e-4, warmup 2000.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
mkdir -p workspace_group_plus/arm_g0plus

python -u scripts/train_group_locusgs.py \
  --arm g0plus \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --plan object_locusgs/plan_6000.json \
  --split workspace_recon_diag/cross_scene/split.json \
  --steps 6000 --warmup 2000 --lr 1e-4 \
  --grad-clip 1.0 --weight-decay 0.05 --seed 42 \
  --log-every 100 --eval-every 500 \
  --save-steps 0 3000 6000 \
  --manifest-out workspace_group_plus/arm_g0plus/manifest.json \
  --out-dir workspace_group_plus/arm_g0plus
