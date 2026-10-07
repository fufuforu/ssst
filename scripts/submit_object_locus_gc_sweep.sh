#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:?phase required: single|eight|train}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
REPORT=/space/mawb/ssst/group_plus/object_locus_gc_sweep_v1
mkdir -p "$REPORT/slurm"
case "$PHASE" in
  single) GPUS=1; MODULE=scripts.smoke_object_locus_gc_sweep; ARGS=--single; DEP=""; NAME=gc-sweep-single ;;
  eight) GPUS=8; MODULE=scripts.smoke_object_locus_gc_sweep; ARGS=--eight; DEP=""; NAME=gc-sweep-eight ;;
  gc001|gc010|gc100) GPUS=8; MODULE=scripts.train_object_locus_gc_sweep; ARGS="--arm $PHASE"; NAME="gc-sweep-$PHASE"; DEP="${2:-}" ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
DEP_ARGS=()
if [[ -n "$DEP" ]]; then DEP_ARGS+=("--dependency=afterany:$DEP"); fi
sbatch --parsable --job-name="$NAME" --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres="gpu:$GPUS" \
  --cpus-per-task=32 --mem=128G --time=72:00:00 --exclusive \
  --output="$REPORT/slurm/%x-%j.out" --error="$REPORT/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$REPO",TASK_PYTHON="$PYTHON",TASK_GPUS="$GPUS",TASK_MODULE="$MODULE",TASK_ARGS="$ARGS",OMP_NUM_THREADS=4 \
  "${DEP_ARGS[@]}" --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module "$TASK_MODULE" $TASK_ARGS'
