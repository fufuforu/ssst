#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?smoke|predict|reduce}"
DEPENDENCY="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT=/space/mawb/ssst/group_plus/object_locus_image_memory_u128_full8gpu_v1/evaluation_epoch08/depth_supplement
CODE_SHA="$(git -C "$REPO" rev-parse HEAD)"
test -z "$(git -C "$REPO" status --porcelain)"
mkdir -p "$ROOT/slurm"
OPTIONS=(--parsable --partition=3090,4090 --nodes=1 --ntasks=1
  --cpus-per-task=4 --mem=32G --time=24:00:00
  --job-name="u128-e8-depth-$MODE"
  --output="$ROOT/slurm/%x-%A_%a.out" --error="$ROOT/slurm/%x-%A_%a.err"
  --export="ALL,EVAL_REPO=$REPO,DEPTH_MODE=$MODE,EVAL_CODE_SHA=$CODE_SHA,OMP_NUM_THREADS=4")
[[ -z "$DEPENDENCY" ]] || OPTIONS+=(--dependency="$DEPENDENCY")
case "$MODE" in
  smoke) OPTIONS+=(--gres=gpu:1);;
  predict) OPTIONS+=(--gres=gpu:1 --array=0-7);;
  reduce) ;;
  *) exit 2;;
esac
sbatch "${OPTIONS[@]}" --wrap='exec /bin/bash "$EVAL_REPO/scripts/run_object_locus_image_memory_depth.sh"'
