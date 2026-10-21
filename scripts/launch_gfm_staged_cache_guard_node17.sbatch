#!/bin/bash
#SBATCH --job-name=gfm_staged_cacheguard
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-17
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt06_adapt2_joint4_node17/cacheguard-slurm-%j.out
set -euo pipefail
cd /space/mawb/ssst_gfm_cache_publication_fix_code_20261015
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/continue_staged_after_cache_failure.py
