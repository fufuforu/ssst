#!/usr/bin/env bash
set -euo pipefail

TASK_PHASE="${1:-contracts}"
TASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_REPORT=/space/mawb/ssst/group_plus/object_locus_region_class_v1
TASK_RUN=/space/mawb/ssst/workspace_group_plus/object_locus_region_class_v1
TASK_LOG="$TASK_REPORT/slurm"
mkdir -p "$TASK_REPORT" "$TASK_RUN" "$TASK_LOG"

case "$TASK_PHASE" in
  contracts)
    cd "$TASK_REPO"
    "$TASK_PYTHON" -m unittest tests.test_object_locus_region_class_contracts -v
    ;;
  single-smoke|two-smoke)
    if [[ "$TASK_PHASE" == single-smoke ]]; then TASK_GPUS=1; TASK_MODE=single; TASK_NAME=region-class-single-smoke
    else TASK_GPUS=2; TASK_MODE=two; TASK_NAME=region-class-two-smoke; fi
    TASK_NODELIST=3dimage-13
    sbatch --parsable --job-name="$TASK_NAME" --partition=3090 --nodelist="$TASK_NODELIST" \
      --nodes=1 --ntasks=1 --gres="gpu:$TASK_GPUS" --cpus-per-task=32 --mem=128G \
      --time=12:00:00 --output="$TASK_LOG/%x-%j.out" --error="$TASK_LOG/%x-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_MODE="$TASK_MODE",TASK_GPUS="$TASK_GPUS",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module scripts.smoke_object_locus_region_class --mode "$TASK_MODE"'
    ;;
  formal)
    cd "$TASK_REPO"
    "$TASK_PYTHON" - <<'PYTHON'
import hashlib,json,subprocess
from pathlib import Path
repo=Path.cwd(); report=Path('/space/mawb/ssst/group_plus/object_locus_region_class_v1')
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
    raise RuntimeError('formal task worktree must be clean')
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/object-locus-region-class-v1'],cwd=repo,text=True)
if not any(line.split()==[sha,'refs/heads/object-locus-region-class-v1'] for line in remote.splitlines()):
    raise RuntimeError('remote task branch SHA does not match current commit')
for name, updates in [('single_smoke.json',2),('two_card_smoke.json',40)]:
    for arm in ('control','region_class'):
        result=json.loads((report/arm/name).read_text())
        if result.get('status')!='PASS' or result.get('updates')!=updates:
            raise RuntimeError(f'{arm} {name} incomplete or failed')
        if result.get('parameter_numel') not in (572432103,572497639):
            raise RuntimeError(f'{arm} parameter count mismatch')
two={a:json.loads((report/a/'two_card_smoke.json').read_text()) for a in ('control','region_class')}
if two['control']['data_plan_sha256']!=two['region_class']['data_plan_sha256']:
    raise RuntimeError('paired two-card smoke plans differ')
if not all(two[a].get('rank_model_and_optimizer_synchronized') and two[a].get('world_size')==2 and
           two[a].get('accumulation_steps')==4 and two[a].get('optimizer_update_calls')==40 for a in two):
    raise RuntimeError('two-rank accumulated synchronization contract missing')
if two['region_class'].get('region_projection_norm',0)<=0:
    raise RuntimeError('region projection did not update in two-card smoke')
(report/'git_provenance.json').write_text(json.dumps(dict(training_sha=sha,remote_sha=sha,
    remote_verification=remote,clean=True,branch='object-locus-region-class-v1'),indent=2)+'\n')
print(json.dumps(dict(training_sha=sha,remote_sha=sha),indent=2))
PYTHON
    TASK_NODELIST=3dimage-13
    TASK_C=$(sbatch --parsable --job-name=object-locus-region-c --partition=3090 --nodelist="$TASK_NODELIST" \
      --nodes=1 --ntasks=1 --gres=gpu:2 --cpus-per-task=32 --mem=128G --time=72:00:00 \
      --output="$TASK_LOG/control-%j.out" --error="$TASK_LOG/control-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --module scripts.train_object_locus_region_class --arm control')
    echo "C_JOB_ID=$TASK_C"
    TASK_R=$(sbatch --parsable --dependency="afterok:$TASK_C" --job-name=object-locus-region-r \
      --partition=3090 --nodelist="$TASK_NODELIST" --nodes=1 --ntasks=1 --gres=gpu:2 \
      --cpus-per-task=32 --mem=128G --time=72:00:00 \
      --output="$TASK_LOG/region-class-%j.out" --error="$TASK_LOG/region-class-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --module scripts.train_object_locus_region_class --arm region_class')
    echo "R_JOB_ID=$TASK_R DEPENDENCY=afterok:$TASK_C"
    echo "Waiting only for C first-update confirmation. R remains dependency-gated."
    while true; do
      if rg -q 'Traceback|RuntimeError|FloatingPointError|CUDA out of memory' \
        "$TASK_LOG/control-$TASK_C.err" "$TASK_LOG/control-$TASK_C.out" 2>/dev/null; then
        echo "C failed before first update; see $TASK_LOG/control-$TASK_C.err" >&2
        exit 1
      fi
      if [[ -s "$TASK_REPORT/control/startup_confirmation.json" ]]; then
        if "$TASK_PYTHON" - "$TASK_REPORT/control/startup_confirmation.json" "$TASK_C" <<'PYTHON'
import json,math,sys
x=json.load(open(sys.argv[1]))
if x.get('status')=='PASS' and x.get('job_id')==sys.argv[2] and x.get('completed_update')==1 and x.get('finite_loss_and_gradient') and math.isfinite(float(x['loss'])) and math.isfinite(float(x['preclip_norm'])):
    print(json.dumps(x,indent=2));raise SystemExit(0)
raise SystemExit(1)
PYTHON
        then break; fi
      fi
      sleep 10
    done
    echo "FORMAL_C_CONFIRMED=$TASK_C"
    echo "FORMAL_R_SUBMITTED_AFTEROK=$TASK_R"
    ;;
  *) echo "usage: $0 {contracts|single-smoke|two-smoke|formal}" >&2; exit 2 ;;
esac
