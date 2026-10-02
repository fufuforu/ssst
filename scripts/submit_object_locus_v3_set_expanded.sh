#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-train}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
NODE="3dimage-13"
REPORTS="/space/mawb/ssst/group_plus/object_locus_v3_set_expanded/slurm"
mkdir -p "$REPORTS"
if [[ "$MODE" == "smoke" ]]; then
  SCRIPT="scripts/smoke_object_locus_v3_set_expanded.py"
  ARGS=""
elif [[ "$MODE" == "train" ]]; then
  SCRIPT="scripts/train_object_locus_v3_set_expanded.py"
  ARGS="--phase train"
elif [[ "$MODE" == "resume" ]]; then
  SCRIPT="scripts/train_object_locus_v3_set_expanded.py"
  ARGS="--phase train --resume"
else
  echo "usage: $0 [smoke|train|resume]" >&2
  exit 2
fi
sbatch --partition=3090 --nodelist="$NODE" --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --export=ALL,REPO="$REPO",PYTHON="$PYTHON",SCRIPT="$SCRIPT",ARGS="$ARGS" \
  --output="$REPORTS/v3set-expanded-%j.out" --error="$REPORTS/v3set-expanded-%j.err" \
  --wrap='cd "$REPO" && "$PYTHON" -u "$SCRIPT" $ARGS'
