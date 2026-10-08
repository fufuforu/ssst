#!/bin/bash
#SBATCH --job-name=object-locus-frozen-extract
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt00/slurm/%j.out
#SBATCH --error=/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt00/slurm/%j.err
set -euo pipefail
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '{"stage":"gpu_extraction","exit_status":%s,"job_id":"%s"}\n' "$status" "${SLURM_JOB_ID:-unknown}" > "/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt00/slurm/failure_${SLURM_JOB_ID:-unknown}.json"
  fi
}
trap on_exit EXIT
WORKTREE=/space/mawb/ssst_object_locus_frozen_representation_diagnostic_v1
cd "$WORKTREE"
test -n "${TASK_CODE_SHA:-}"
git checkout --detach "$TASK_CODE_SHA"
test "$(git rev-parse HEAD)" = "$TASK_CODE_SHA"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/extract_object_locus_frozen_probe.py
