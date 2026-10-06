#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-formal}"
TASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_REPORT=/space/mawb/ssst/group_plus/object_locus_mh_feedback_v1
TASK_RUN=/space/mawb/ssst/workspace_group_plus/object_locus_mh_feedback_v1
TASK_LOG="$TASK_REPORT/slurm"
mkdir -p "$TASK_REPORT" "$TASK_RUN" "$TASK_LOG"

case "$PHASE" in
  contracts)
    "$TASK_PYTHON" -m scripts.smoke_object_locus_mh_feedback --mode contracts
    ;;
  single-smoke|eight-smoke)
    if [[ "$PHASE" == single-smoke ]]; then TASK_GPUS=1; TASK_MODE=single; TASK_JOB="mhfb-single-smoke"; TASK_UPDATES=2
    else TASK_GPUS=8; TASK_MODE=eight; TASK_JOB="mhfb-eight-smoke"; TASK_UPDATES=40; fi
    sbatch --parsable --job-name="$TASK_JOB" --partition=3090 --nodelist=3dimage-11 \
      --nodes=1 --ntasks=1 --gres="gpu:$TASK_GPUS" --cpus-per-task=32 --mem=128G \
      --time=12:00:00 --output="$TASK_LOG/%x-%j.out" --error="$TASK_LOG/%x-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_MODE="$TASK_MODE",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" --module scripts.smoke_object_locus_mh_feedback --mode "$TASK_MODE" --updates '"$TASK_UPDATES"
    ;;
  formal)
    cd "$TASK_REPO"
    "$TASK_PYTHON" - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
repo=Path.cwd(); report=Path('/space/mawb/ssst/group_plus/object_locus_mh_feedback_v1')
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(): raise RuntimeError('formal checkout must be clean')
rows=subprocess.check_output(['git','ls-remote','origin','refs/heads/object-locus-mh-feedback-v1'],cwd=repo,text=True).splitlines()
if not any(x.split()==[sha,'refs/heads/object-locus-mh-feedback-v1'] for x in rows): raise RuntimeError('remote SHA mismatch')
contracts=json.loads((report/'contracts_smoke.json').read_text())
single=json.loads((report/'single_smoke.json').read_text())
eight=json.loads((report/'eight_smoke.json').read_text())
if contracts.get('status')!='PASS' or single.get('status')!='PASS' or single.get('updates')!=2: raise RuntimeError('CPU/single-card contracts are incomplete')
if eight.get('status')!='PASS' or eight.get('updates')!=40 or eight.get('world_size')!=8: raise RuntimeError('eight-card smoke is incomplete')
if not all(v>0 for v in eight.get('injection_weight_norms',{}).values()): raise RuntimeError('four-layer injection did not activate')
(report/'git_provenance.json').write_text(json.dumps(dict(training_sha=sha,remote_verification='verified via git ls-remote',clean=True),indent=2)+'\n')
PY
    TASK_JOB_ID=$(sbatch --parsable --job-name=mh-feedback-v1 --partition=3090 --nodelist=3dimage-11 \
      --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=32 --mem=128G --time=72:00:00 \
      --output="$TASK_LOG/formal-%j.out" --error="$TASK_LOG/formal-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.train_object_locus_mh_feedback')
    echo "JOB_ID=$TASK_JOB_ID"
    echo "Waiting only for first-update receipt in $TASK_REPORT/startup_confirmation.json"
    while true; do
      if [[ -s "$TASK_REPORT/startup_confirmation.json" ]]; then
        if "$TASK_PYTHON" - "$TASK_REPORT/startup_confirmation.json" "$TASK_JOB_ID" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
if x.get('status')=='PASS' and x.get('job_id')==sys.argv[2] and x.get('completed_update')==1 and x.get('finite_loss_and_gradient') and math.isfinite(float(x['loss'])) and math.isfinite(float(x['preclip_norm'])):
 print(json.dumps(x,indent=2));raise SystemExit(0)
raise SystemExit(1)
PY
        then break; fi
      fi
      sleep 15
    done
    ;;
  *) echo "usage: $0 {contracts|single-smoke|eight-smoke|formal}" >&2; exit 2 ;;
esac
