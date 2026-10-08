#!/bin/bash
#SBATCH --job-name=frozen-unified-eval
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail
on_exit(){ status=$?; if [ "$status" -ne 0 ]; then printf '{"stage":"unified_eval_report","exit_status":%s,"job_id":"%s"}\n' "$status" "${SLURM_JOB_ID:-unknown}" > "$TASK_ATTEMPT_ROOT/slurm/failure_report_${SLURM_JOB_ID:-unknown}.json"; fi; }
trap on_exit EXIT
WORKTREE=/space/mawb/ssst_object_locus_frozen_representation_diagnostic_v1
cd "$WORKTREE"
test -n "${TASK_ATTEMPT_ROOT:-}"; test -n "${TASK_CODE_SHA:-}"
mkdir -p "$TASK_ATTEMPT_ROOT/slurm"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/verify_execution_files_manifest.py --attempt "$TASK_ATTEMPT_ROOT"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(/space/mawb/anaconda3/envs/tokengs/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$TASK_ATTEMPT_ROOT/extraction_complete.json")" = PASS
test -f "$TASK_ATTEMPT_ROOT/r3d_frozen_inference_receipt.json"
test -f "$TASK_ATTEMPT_ROOT/heads_complete.json"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/eval_object_locus_frozen_probe.py --attempt "$TASK_ATTEMPT_ROOT"
TASK_OFFICIAL_PYTHON=/space/mawb/SIU3R/.venv_gpu_v4/bin/python TASK_ATTEMPT_ROOT="$TASK_ATTEMPT_ROOT" /space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/report_object_locus_frozen_probe.py --attempt "$TASK_ATTEMPT_ROOT"
