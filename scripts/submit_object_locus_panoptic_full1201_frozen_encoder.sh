#!/usr/bin/env bash
set -euo pipefail
TASK_PHASE="${1:-train}"
TASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_REPORT=/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu
mkdir -p "$TASK_REPORT/slurm"
case "$TASK_PHASE" in
 single) TASK_GPUS=1; TASK_MODULE=scripts.smoke_object_locus_panoptic_full1201_frozen_encoder; TASK_ARGS=single;;
 eight) TASK_GPUS=8; TASK_MODULE=scripts.smoke_object_locus_panoptic_full1201_frozen_encoder; TASK_ARGS=eight;;
 train) TASK_GPUS=8; TASK_MODULE=scripts.train_object_locus_panoptic_full1201_frozen_encoder; TASK_ARGS=;;
 *) exit 2;;
esac
if [[ "$TASK_PHASE" == train ]]; then
 cd "$TASK_REPO"
 "$TASK_PYTHON" - <<'PY'
import subprocess,json
from pathlib import Path
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty formal checkout')
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main','refs/heads/object-locus-panoptic-full1201-frozen-encoder'],text=True)
if not any(line.split()==[sha,'refs/heads/object-locus-panoptic-full1201-frozen-encoder'] for line in remote.splitlines()):raise RuntimeError('training SHA not pushed')
p=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu/git_provenance.json');p.write_text(json.dumps(dict(training_sha=sha,clean=True,remote_verification=remote,verification_host='submit host only'),indent=2)+'\n')
PY
fi
sbatch --job-name="panoptic-full-$TASK_PHASE" --partition=3090 --nodelist=3dimage-13 --nodes=1 --ntasks=1 --gres="gpu:$TASK_GPUS" --cpus-per-task=32 --mem=128G --time=72:00:00 --output="$TASK_REPORT/slurm/%x-%j.out" --error="$TASK_REPORT/slurm/%x-%j.err" --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_GPUS="$TASK_GPUS",TASK_MODULE="$TASK_MODULE",TASK_ARGS="$TASK_ARGS",OMP_NUM_THREADS=4 --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module "$TASK_MODULE" $TASK_ARGS'
