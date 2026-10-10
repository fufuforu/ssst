#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?smoke|predict|prepare|aggregate|finish}"
DEPENDENCY="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REPORT=/space/mawb/ssst/group_plus/object_locus_image_memory_u128_full8gpu_v1/evaluation_epoch08
CODE_SHA="$(git -C "$REPO" rev-parse HEAD)"
test -z "$(git -C "$REPO" status --porcelain)"
mkdir -p "$REPORT/slurm"
OPTIONS=(--parsable --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1
  --cpus-per-task=8 --mem=48G --time=48:00:00
  --job-name="u128-e8-official-$MODE"
  --output="$REPORT/slurm/%x-%A_%a.out" --error="$REPORT/slurm/%x-%A_%a.err"
  --export="ALL,EVAL_REPO=$REPO,EVAL_MODE=$MODE,EVAL_CODE_SHA=$CODE_SHA,OMP_NUM_THREADS=4")
[[ -z "$DEPENDENCY" ]] || OPTIONS+=(--dependency="$DEPENDENCY")
case "$MODE" in
  smoke) OPTIONS+=(--gres=gpu:1);;
  predict) OPTIONS+=(--gres=gpu:1 --array=0-7);;
  aggregate) OPTIONS+=(--array=0-7);;
  prepare) ;;
  finish) OPTIONS+=(--mem=128G);;
  *) exit 2;;
esac
sbatch "${OPTIONS[@]}" --wrap='set -euo pipefail
cd "$EVAL_REPO"
if [[ "$EVAL_MODE" == smoke || "$EVAL_MODE" == predict ]]; then
  /space/mawb/anaconda3/envs/tokengs/bin/python -m scripts.eval_object_locus_image_memory_official "$EVAL_MODE" --shard "${SLURM_ARRAY_TASK_ID:-0}"
  if [[ "$EVAL_MODE" == smoke ]]; then
    /space/mawb/SIU3R/.venv_gpu_v4/bin/python -m scripts.eval_object_locus_image_memory_official smoke-check
  fi
else
  /space/mawb/SIU3R/.venv_gpu_v4/bin/python -m scripts.eval_object_locus_image_memory_official "$EVAL_MODE" --shard "${SLURM_ARRAY_TASK_ID:-0}"
fi'
