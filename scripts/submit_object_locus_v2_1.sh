#!/usr/bin/env bash
# Submit only to the registered RTX3090 nodes. A busy node 13 is queued in place.
set -euo pipefail
TASK_PHASE="${1:-train}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
NODE13="3dimage-13"
STATE13="$(sinfo -N -h -n "$NODE13" -o '%T %G' | head -n1 || true)"
STATE13_LOWER="${STATE13,,}"
if [[ -z "$STATE13" || "$STATE13_LOWER" == *down* || "$STATE13_LOWER" == *drain* || "$STATE13" != *3090* ]]; then
  NODE="3dimage-11"
else
  NODE="$NODE13"
fi
REPORTS_DIR="/space/mawb/ssst/group_plus/object_locus_v2_1/slurm"
mkdir -p "$REPORTS_DIR"
if [[ "$TASK_PHASE" == "smoke" ]]; then
  SCRIPT="scripts/smoke_object_locus_v2_1.py"
  ARGS=""
else
  SCRIPT="scripts/train_object_locus_v2_1.py"
  ARGS="--phase train --device cuda"
fi
sbatch --partition=3090 --nodelist="$NODE" --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=24:00:00 --export=ALL,REPO="$REPO",PYTHON="$PYTHON",TASK_PHASE="$TASK_PHASE",SCRIPT="$SCRIPT",ARGS="$ARGS" \
  --output="$REPORTS_DIR/%x-%j.out" \
  --wrap='cd "$REPO" && "$PYTHON" -u "$SCRIPT" $ARGS'
