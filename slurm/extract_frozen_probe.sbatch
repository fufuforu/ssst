#!/bin/bash
#SBATCH --job-name=object-locus-frozen-extract
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '{"stage":"gpu_extraction","exit_status":%s,"job_id":"%s"}\n' "$status" "${SLURM_JOB_ID:-unknown}" > "$TASK_ATTEMPT_ROOT/slurm/failure_${SLURM_JOB_ID:-unknown}.json"
  fi
}
trap on_exit EXIT
WORKTREE=/space/mawb/ssst_object_locus_frozen_representation_diagnostic_v1
cd "$WORKTREE"
test -n "${TASK_CODE_SHA:-}"
test -n "${TASK_ATTEMPT_ROOT:-}"
mkdir -p "$TASK_ATTEMPT_ROOT/slurm"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/verify_execution_files_manifest.py --attempt "$TASK_ATTEMPT_ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/extract_object_locus_frozen_probe.py --attempt "$TASK_ATTEMPT_ROOT"
