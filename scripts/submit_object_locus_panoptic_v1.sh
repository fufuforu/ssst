#!/usr/bin/env bash
set -euo pipefail
TASK_PHASE="${1:-train}"
TASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_LOG=/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu/slurm
mkdir -p "$TASK_LOG"
case "$TASK_PHASE" in
 single) TASK_GPUS=1; TASK_MODULE=scripts.smoke_object_locus_panoptic_v1; TASK_ARGS=--single ;;
 eight) TASK_GPUS=8; TASK_MODULE=scripts.smoke_object_locus_panoptic_v1; TASK_ARGS=--eight ;;
 train) TASK_GPUS=8; TASK_MODULE=scripts.train_object_locus_panoptic_v1; TASK_ARGS= ;;
 resume) TASK_GPUS=8; TASK_MODULE=scripts.train_object_locus_panoptic_v1; TASK_ARGS=--resume ;;
 *) exit 2 ;;
esac
if [[ "$TASK_PHASE" == train || "$TASK_PHASE" == resume ]]; then
  cd "$TASK_REPO"
  "$TASK_PYTHON" - <<'PYTHON'
import json, subprocess
from pathlib import Path
path=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu/git_provenance.json')
receipt=json.loads(path.read_text())
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main','refs/heads/object-locus-panoptic-v1-8gpu'],text=True)
if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise RuntimeError('dirty formal checkout')
if not any(line.split()==[sha,'refs/heads/object-locus-panoptic-v1-8gpu'] for line in remote.strip().splitlines()): raise RuntimeError('training SHA not pushed')
receipt.update(training_sha=sha,main_sha=next(line.split()[0] for line in remote.strip().splitlines() if line.endswith('refs/heads/main')),remote_verification=remote,clean=True,verification_host='submit host; compute node has no dependency on login-node proxy')
path.write_text(json.dumps(receipt,indent=2)+'\n')
print('Submit-host remote verification:',sha)
PYTHON
fi
sbatch --job-name="panoptic-v1-$TASK_PHASE" --partition=3090 --nodelist=3dimage-13 --nodes=1 --ntasks=1 --gres="gpu:$TASK_GPUS" \
 --cpus-per-task=32 --mem=128G --time=72:00:00 --output="$TASK_LOG/%x-%j.out" --error="$TASK_LOG/%x-%j.err" \
 --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_GPUS="$TASK_GPUS",TASK_MODULE="$TASK_MODULE",TASK_ARGS="$TASK_ARGS",OMP_NUM_THREADS=4 \
 --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module "$TASK_MODULE" $TASK_ARGS'
