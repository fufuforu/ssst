#!/usr/bin/env bash
set -euo pipefail
TASK_PHASE="${1:-single-smoke}"
TASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_REPORT=/space/mawb/ssst/group_plus/object_locus_mask_guided_v1
TASK_LOG="$TASK_REPORT/slurm"
mkdir -p "$TASK_LOG" "$TASK_REPORT/control" "$TASK_REPORT/mask_guided"

submit_smoke() {
  local mode="$1" gpu_count="$2"
  sbatch --parsable --job-name="lmg-$mode-smoke" --partition=3090 --nodelist=3dimage-11 \
    --nodes=1 --ntasks=1 --gres="gpu:$gpu_count" --cpus-per-task=32 --mem=128G \
    --time=72:00:00 --output="$TASK_LOG/%x-%j.out" --error="$TASK_LOG/%x-%j.err" \
    --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_MODE="$mode",TASK_GPUS="$gpu_count",OMP_NUM_THREADS=4 \
    --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module scripts.smoke_object_locus_mask_guided --mode "$TASK_MODE" --arm both'
}

case "$TASK_PHASE" in
  single-smoke) submit_smoke single 1 ;;
  eight-smoke)
    sbatch --parsable --job-name=lmg-eight-smoke \
      --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres=gpu:8 \
      --cpus-per-task=32 --mem=128G --time=72:00:00 --output="$TASK_LOG/%x-%j.out" \
      --error="$TASK_LOG/%x-%j.err" --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_MODE=eight,TASK_GPUS=8,OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.smoke_object_locus_mask_guided --mode eight --arm both' ;;
  formal)
    cd "$TASK_REPO"
    "$TASK_PYTHON" - <<'PYTHON'
import json, subprocess
from pathlib import Path
import math
repo=Path.cwd()
report=Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1')
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/object-locus-mask-guided-v1'],cwd=repo,text=True)
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(): raise RuntimeError('formal checkout must be clean')
if not any(x.split()==[sha,'refs/heads/object-locus-mask-guided-v1'] for x in remote.splitlines()): raise RuntimeError('task commit not present on remote')
for mode,updates in [('single',2),('eight',40)]:
    values={arm:json.loads((report/arm/f'{mode}_smoke.json').read_text()) for arm in ('control','mask_guided')}
    if any(x.get('status')!='PASS' or x.get('updates')!=updates for x in values.values()): raise RuntimeError(f'{mode} smoke did not pass for both arms')
    if mode=='eight':
        if not all(x.get('all_parameters_trainable') for x in values.values()): raise RuntimeError('eight-GPU smoke unexpectedly froze parameters')
        if values['control'].get('data_plan_sha256')!=values['mask_guided'].get('data_plan_sha256'): raise RuntimeError('eight-GPU plan mismatch')
eight={arm:json.loads((report/arm/'eight_smoke.json').read_text()) for arm in ('control','mask_guided')}
if eight['control']['parameter_numel']!=eight['mask_guided']['parameter_numel']: raise RuntimeError('arm parameter count differs')
if not all(eight[arm].get('synchronized_parameters_and_optimizer') for arm in eight): raise RuntimeError('eight-GPU state sync contract absent')
for arm in eight:
    norms=eight[arm].get('injection_weight_norms',{})
    if set(norms)!={'L6','L8','L10','L12'} or not all(math.isfinite(float(v)) and float(v)>0 for v in norms.values()): raise RuntimeError(f'{arm} smoke injection inactive')
zero=json.loads((report/'mask_guided'/'single_smoke.json').read_text()).get('zero_injection_comparison')
if not zero or zero['control_vs_mask_guided_gaussian']>zero['gaussian_repeat_noise']+1e-6 or zero['control_vs_mask_guided_rgb']>zero['rgb_repeat_noise']+1e-6:
    raise RuntimeError('zero-injection output comparison did not pass its repeated-forward envelope')
(report/'git_provenance.json').write_text(json.dumps(dict(training_sha=sha,remote_verification=remote,clean=True),indent=2)+'\n')
print('Verified pushed task SHA',sha)
PYTHON
    TASK_C=$(sbatch --parsable --job-name=lmg-control --partition=3090 --nodelist=3dimage-11 \
      --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=32 --mem=128G --time=72:00:00 \
      --output="$TASK_LOG/control-%j.out" --error="$TASK_LOG/control-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.train_object_locus_mask_guided --arm control')
    echo "Submitted control job $TASK_C"
    "$TASK_PYTHON" - "$TASK_C" <<'PYTHON'
import json,sys,time
from pathlib import Path
p=Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1/control/progress.json')
while True:
    if p.exists():
        try:
            x=json.loads(p.read_text())
            if x.get('global_update',0)>=1 and x.get('loss') is not None and __import__('math').isfinite(float(x['loss'])):
                cp=Path('/space/mawb/ssst/workspace_group_plus/object_locus_mask_guided_v1/control/checkpoint_epoch_00.pt')
                if cp.is_file():
                    print(json.dumps(dict(job_id=sys.argv[1],epoch0_checkpoint=str(cp),first_update=x['global_update'],first_loss=x['loss'])))
                    break
        except (OSError,ValueError,json.JSONDecodeError): pass
    time.sleep(15)
PYTHON
    TASK_M=$(sbatch --parsable --dependency="afterok:$TASK_C" --job-name=lmg-mask-guided \
      --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres=gpu:8 \
      --cpus-per-task=32 --mem=128G --time=72:00:00 \
      --output="$TASK_LOG/mask-guided-%j.out" --error="$TASK_LOG/mask-guided-%j.err" \
      --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",OMP_NUM_THREADS=4 \
      --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.train_object_locus_mask_guided --arm mask_guided')
    echo "Submitted mask-guided job $TASK_M with afterok:$TASK_C"
    echo "C_JOB_ID=$TASK_C M_JOB_ID=$TASK_M"
    ;;
  *) echo "usage: $0 {single-smoke|eight-smoke|formal}" >&2; exit 2 ;;
esac
