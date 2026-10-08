#!/bin/bash
#SBATCH --job-name=frozen-probe-heads
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
on_exit(){ status=$?; if [ "$status" -ne 0 ]; then printf '{"stage":"parallel_probe_heads","exit_status":%s,"job_id":"%s"}\n' "$status" "${SLURM_JOB_ID:-unknown}" > "$TASK_ATTEMPT_ROOT/slurm/failure_heads_${SLURM_JOB_ID:-unknown}.json"; fi; }
trap on_exit EXIT
WORKTREE=/space/mawb/ssst_object_locus_frozen_representation_diagnostic_v1
cd "$WORKTREE"
test -n "${TASK_ATTEMPT_ROOT:-}"; test -n "${TASK_CODE_SHA:-}"
mkdir -p "$TASK_ATTEMPT_ROOT/slurm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/space/mawb/anaconda3/envs/tokengs/bin/python -m scripts.verify_execution_files_manifest --attempt "$TASK_ATTEMPT_ROOT"
/space/mawb/anaconda3/envs/tokengs/bin/python -m scripts.train_parallel_probe_heads --attempt "$TASK_ATTEMPT_ROOT"
