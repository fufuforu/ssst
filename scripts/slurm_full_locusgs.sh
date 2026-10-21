#!/bin/bash
#SBATCH --job-name=lgs-full-50k
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-12
#SBATCH --output=/space/mawb/ssst/workspace_recon_diag/full_train/run/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_recon_diag/full_train/run/slurm-%j.err
#
# Full processed-ScanNet pure-LocusGS reconstruction training (the 32/8 recipe
# that was stable, with only these changes: the whole official SIU3R train split
# instead of 32 scenes, a 50000-step budget with the same 2000-step warmup, and a
# resumable checkpoint policy).  See docs/locusgs_full_train.md.
#
# Resume an interrupted job with the SAME schedule:
#   sbatch scripts/slurm_full_locusgs.sh --resume <out-dir>
# Extend the budget later (new recipe, cosine recomputed - use deliberately):
#   --steps <new total> --resume <out-dir> --allow-schedule-change
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst

python -u scripts/train_cross_scene.py \
  --split workspace_recon_diag/cross_scene/split.json \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius \
  --lr 1e-4 --steps 50000 --eval-every 2500 --log-every 100 \
  --dense-start 200 --dense-end 2600 --dense-every 200 \
  --amp fp32 --seed 42 \
  --train-scenes all --require-disjoint-val-root \
  --monitor-scenes scene0011_00 scene0246_00 scene0458_01 scene0621_00 \
  --ckpt-every 2500 --keep-last 2 --keep-steps 2500 12500 25000 50000 \
  --manifest-out workspace_recon_diag/full_train/run/manifest.json \
  --budget-note "initial budget 50000 optimizer steps; warmup 2000; peak lr 1e-4; converged status not claimed" \
  --out-dir workspace_recon_diag/full_train/run \
  "$@"
