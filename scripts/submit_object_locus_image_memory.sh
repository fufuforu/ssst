#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?single eight train}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REPORT=/space/mawb/ssst/group_plus/object_locus_image_memory_u128_full8gpu_v1
mkdir -p "$REPORT/slurm"
GPUS=8; [[ "$MODE" != single ]] || GPUS=1
sbatch --parsable --job-name="u128-fresh-$MODE" --partition=4090 --nodelist=3dimage-17 --nodes=1 --ntasks=1 --gres="gpu:$GPUS" \
  --cpus-per-task=32 --mem=128G --time=72:00:00 \
  --output="$REPORT/slurm/%x-%j.out" --error="$REPORT/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$REPO",TASK_MODE="$MODE",TASK_GPUS="$GPUS",OMP_NUM_THREADS=4 \
  --wrap='cd "$TASK_REPO" && /space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module scripts.train_object_locus_image_memory --arm u128 --mode "$TASK_MODE"'
