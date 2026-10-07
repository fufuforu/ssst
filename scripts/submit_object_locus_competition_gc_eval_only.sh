#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1"
EVAL_ROOT="$REPORT/four_arm_evaluation_retry01"
SMOKE_ROOT="$REPORT/four_arm_eval_interface_smoke_retry01"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
OFFICIAL_PYTHON="/space/mawb/SIU3R/.venv_gpu_v4/bin/python"

mkdir -p "$REPORT/slurm"
RECEIPT="$REPORT/evaluation_retry01_job_receipt.json"
if [[ -e "$RECEIPT" ]]; then
  echo "evaluation retry receipt already exists; inspect it before submitting" >&2
  exit 2
fi
for root in "$EVAL_ROOT" "$SMOKE_ROOT"; do
  if [[ -e "$root" ]] && find "$root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "output root is nonempty; refusing overwrite: $root" >&2
    exit 2
  fi
done
ACTIVE="$(squeue -h -u "$USER" -n competition-four-arm-eval-retry01 -o '%i' | head -n 1)"
if [[ -n "$ACTIVE" ]]; then
  echo "evaluation retry job already pending/running: $ACTIVE" >&2
  exit 2
fi

JOB="$(sbatch --parsable \
  --job-name=competition-four-arm-eval-retry01 \
  --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=48:00:00 \
  --output="$REPORT/slurm/eval-retry01-%j.out" \
  --error="$REPORT/slurm/eval-retry01-%j.err" \
  --chdir="$REPO" \
  --wrap="bash -lc 'set -euo pipefail; export OMP_NUM_THREADS=4 TASK_OFFICIAL_PYTHON=\"$OFFICIAL_PYTHON\"; \"$PYTHON\" -m scripts.eval_object_locus_gc_competition_four_arm --eval-root \"$EVAL_ROOT\" --interface-smoke --smoke-root \"$SMOKE_ROOT\"; \"$PYTHON\" -m scripts.eval_object_locus_gc_competition_four_arm --eval-root \"$EVAL_ROOT\"; \"$OFFICIAL_PYTHON\" -m scripts.summarize_object_locus_gc_competition_four_arm --eval-root \"$EVAL_ROOT\"'")"

"$PYTHON" - "$RECEIPT" "$JOB" "$REPO" "$EVAL_ROOT" "$SMOKE_ROOT" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
receipt, job, repo, eval_root, smoke_root = sys.argv[1:]
sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
record = {
    "created_unix": int(time.time()),
    "evaluation_job_id": job,
    "evaluation_code_sha": sha,
    "eval_root": eval_root,
    "interface_smoke_root": smoke_root,
    "previous_failed_eval_job_id": "58931",
    "completed_training_job_id": "58930",
    "training_state": "COMPLETED",
    "training_exit_code": "0:0",
    "endpoints": [
        "/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1/gc001/checkpoint_epoch8.pt",
        "/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1/gc010/checkpoint_epoch8.pt",
        "/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1/gc100/checkpoint_epoch8.pt",
        "/space/mawb/ssst/workspace_group_plus/object_locus_competition_gc001_v1/comp_gc001/checkpoint_epoch8.pt",
    ],
}
Path(receipt).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
PY
