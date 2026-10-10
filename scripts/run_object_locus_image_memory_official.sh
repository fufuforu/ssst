#!/usr/bin/env bash
set -euo pipefail
cd "$EVAL_REPO"
if [[ "$EVAL_MODE" == smoke || "$EVAL_MODE" == predict ]]; then
  /space/mawb/anaconda3/envs/tokengs/bin/python -m scripts.eval_object_locus_image_memory_official "$EVAL_MODE" --shard "${SLURM_ARRAY_TASK_ID:-0}"
  if [[ "$EVAL_MODE" == smoke ]]; then
    /space/mawb/SIU3R/.venv_gpu_v4/bin/python -m scripts.eval_object_locus_image_memory_official smoke-check
  fi
else
  /space/mawb/SIU3R/.venv_gpu_v4/bin/python -m scripts.eval_object_locus_image_memory_official "$EVAL_MODE" --shard "${SLURM_ARRAY_TASK_ID:-0}"
fi
