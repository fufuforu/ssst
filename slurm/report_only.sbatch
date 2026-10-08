#!/bin/bash
#SBATCH --job-name=frozen-probe-report
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
test -n "${TASK_ATTEMPT_ROOT:-}"
test -n "${TASK_CODE_SHA:-}"
WORKTREE=/space/mawb/ssst_object_locus_frozen_representation_diagnostic_v1
cd "$WORKTREE"
mkdir -p "$TASK_ATTEMPT_ROOT/slurm"
exec >>"$TASK_ATTEMPT_ROOT/slurm/report-${SLURM_JOB_ID}.out" 2>>"$TASK_ATTEMPT_ROOT/slurm/report-${SLURM_JOB_ID}.err"
on_exit(){
  status=$?
  if [ "$status" -ne 0 ]; then
    /space/mawb/anaconda3/envs/tokengs/bin/python - "$TASK_ATTEMPT_ROOT" "$SLURM_JOB_ID" "$status" "$TASK_CODE_SHA" <<'PY'
import json,sys,datetime,pathlib
r=pathlib.Path(sys.argv[1]);p=r/'slurm'/f'report_failure_{sys.argv[2]}.json'
p.write_text(json.dumps({'stage':'report_only','job_id':sys.argv[2],'exit_status':int(sys.argv[3]),'execution_sha':sys.argv[4],
 'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
  fi
}
trap on_exit EXIT
/space/mawb/anaconda3/envs/tokengs/bin/python -m scripts.verify_execution_files_manifest --attempt "$TASK_ATTEMPT_ROOT"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(/space/mawb/anaconda3/envs/tokengs/bin/python - "$TASK_ATTEMPT_ROOT/preflight/report_preflight.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)" = PASS
TASK_ATTEMPT_ROOT="$TASK_ATTEMPT_ROOT" /space/mawb/SIU3R/.venv_gpu_v4/bin/python -m scripts.report_object_locus_frozen_probe --attempt "$TASK_ATTEMPT_ROOT"
