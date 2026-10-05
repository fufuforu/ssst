#!/usr/bin/env bash
set -euo pipefail
EVAL_MODE="${1:?smoke|monitor|full}"
EVAL_EPOCH="${2:-8}"
EVAL_NODE="${3:-3dimage-13}"
EVAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_REPORT=/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu
case "$EVAL_NODE" in 3dimage-13|3dimage-11) ;; *) exit 2;; esac
case "$EVAL_MODE" in
 smoke) EVAL_ARRAY=0;;
 monitor) EVAL_ARRAY=0-5;;
 full) EVAL_ARRAY=0-7;;
 *) exit 2;;
esac
mkdir -p "$EVAL_REPORT/slurm"
sbatch --job-name="panoptic-full-eval-$EVAL_MODE-e$EVAL_EPOCH" --partition=3090 --nodelist="$EVAL_NODE" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=48:00:00 --array="$EVAL_ARRAY" --output="$EVAL_REPORT/slurm/%x-%A_%a.out" --error="$EVAL_REPORT/slurm/%x-%A_%a.err" --export=ALL,EVAL_REPO="$EVAL_REPO",EVAL_MODE="$EVAL_MODE",EVAL_EPOCH="$EVAL_EPOCH",OMP_NUM_THREADS=4 --wrap='cd "$EVAL_REPO" && exec /space/mawb/anaconda3/envs/tokengs/bin/python -m scripts.eval_object_locus_panoptic_full1201 "$EVAL_MODE" --epoch "$EVAL_EPOCH" --shard "$SLURM_ARRAY_TASK_ID" --shards 8'
