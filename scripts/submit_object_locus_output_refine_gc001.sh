#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
REPORT_ATTEMPT="${TASK_REPORT_ATTEMPT:?set report attempts/attemptNN path}"
RUN_ATTEMPT="${TASK_RUN_ATTEMPT:?set checkpoint attempts/attemptNN path}"
mkdir -p "$REPORT_ATTEMPT/slurm" "$RUN_ATTEMPT"
TASK_CODE_SHA="$(git -C "$REPO" rev-parse HEAD)"
JOB_BODY='set -euo pipefail
cd "$TASK_REPO"
test "$(git rev-parse HEAD)" = "$TASK_CODE_SHA"
"$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 --module scripts.smoke_object_locus_output_refine_gc001 --single
"$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 --module scripts.smoke_object_locus_output_refine_gc001 --four
"$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 --module scripts.train_object_locus_output_refine_gc001 --formal'
sbatch --parsable \
  --job-name=object-locus-r3d-gc001 \
  --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=16 --mem=64G --time=72:00:00 \
  --output="$REPORT_ATTEMPT/slurm/%x-%j.out" \
  --error="$REPORT_ATTEMPT/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$REPO",TASK_PYTHON="$PYTHON",TASK_REPORT_ATTEMPT="$REPORT_ATTEMPT",TASK_RUN_ATTEMPT="$RUN_ATTEMPT",TASK_CODE_SHA="$TASK_CODE_SHA",OMP_NUM_THREADS=4 \
  --wrap="$JOB_BODY"
