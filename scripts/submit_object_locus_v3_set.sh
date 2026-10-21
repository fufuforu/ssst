#!/usr/bin/env bash
set -euo pipefail
TASK_PHASE="${1:-train}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
NODE="3dimage-13"
REPORTS="/space/mawb/ssst/group_plus/object_locus_v3_set/slurm"
mkdir -p "$REPORTS"
if [[ "$TASK_PHASE" == "smoke" ]]; then
  SCRIPT="scripts/smoke_object_locus_v3_set.py"; ARGS=""
else
  SCRIPT="scripts/train_object_locus_v3_set.py"; ARGS="--phase train --device cuda"
fi
sbatch --partition=3090 --nodelist="$NODE" --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --export=ALL,REPO="$REPO",PYTHON="$PYTHON",SCRIPT="$SCRIPT",ARGS="$ARGS" \
  --output="$REPORTS/%x-%j.out" \
  --wrap='cd "$REPO" && "$PYTHON" -u "$SCRIPT" $ARGS'
