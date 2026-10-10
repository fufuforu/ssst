#!/usr/bin/env bash
set -euo pipefail
cd "$EVAL_REPO"
case "$DEPTH_MODE" in
  smoke|predict)
    /space/mawb/anaconda3/envs/tokengs/bin/python -u -m scripts.eval_object_locus_image_memory_depth "$DEPTH_MODE" --shard "${SLURM_ARRAY_TASK_ID:-0}"
    OPTIONS=()
    [[ "$DEPTH_MODE" != smoke ]] || OPTIONS+=(--smoke)
    /space/mawb/SIU3R/.venv_gpu_v4/bin/python -u -m scripts.eval_object_locus_image_memory_depth score --shard "${SLURM_ARRAY_TASK_ID:-0}" "${OPTIONS[@]}"
    ;;
  reduce)
    exec /space/mawb/SIU3R/.venv_gpu_v4/bin/python -u -m scripts.eval_object_locus_image_memory_depth reduce
    ;;
  *) exit 2;;
esac
