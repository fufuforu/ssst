#!/bin/bash
#SBATCH --job-name=lgs-lrcap2e5
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/slurm-%j.err
#
# Continuation branch of the full-split LocusGS run with ONE intervention:
#   lr = min(schedule_lr, 2e-5)
# Branch point: run/ckpt_step2500, the last complete (model+optimizer+scheduler+
# RNG+sampler) checkpoint before the monitor peak at step 7500.  Model weights,
# optimizer state (no reset, no re-warmup), data order, fp32, loss, view protocol,
# seed and the schedule shape (--steps 50000) are all inherited unchanged.
# See docs/locusgs_full_train_collapse.md.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/train_cross_scene.py \
  --split workspace_recon_diag/cross_scene/split.json \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius \
  --lr 1e-4 --lr-cap 2e-5 --steps 50000 --eval-every 2500 --log-every 100 \
  --dense-start 2500 --dense-end 12500 --dense-every 500 \
  --amp fp32 --seed 42 \
  --train-scenes all --require-disjoint-val-root \
  --monitor-scenes scene0011_00 scene0246_00 scene0458_01 scene0621_00 \
  --ckpt-every 2500 --keep-last 2 --keep-steps 5000 25000 50000 \
  --resume workspace_recon_diag/full_train/run/ckpt_step2500 \
  --manifest-out workspace_recon_diag/full_train/run_lrcap2e5/manifest.json \
  --budget-note "continuation from step 2500 (last complete pre-peak ckpt); only intervention lr=min(schedule,2e-5)" \
  --out-dir workspace_recon_diag/full_train/run_lrcap2e5 \
  "$@"
