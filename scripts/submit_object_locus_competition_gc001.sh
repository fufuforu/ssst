#!/usr/bin/env bash
set -euo pipefail

# Submit the registered competition arm after the verified GC1.0 job, then
# submit the unified endpoint evaluation after this arm succeeds.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
TORCHRUN="/space/mawb/anaconda3/envs/tokengs/bin/torchrun"
OFFICIAL_PYTHON="/space/mawb/SIU3R/.venv_gpu_v4/bin/python"
GC100_JOB="${1:?usage: submit_object_locus_competition_gc001.sh GC100_JOB_ID}"

mkdir -p "$REPORT/slurm"
if [[ -e "$REPORT/job_receipt.json" ]]; then
  echo "job receipt already exists; inspect it before submitting again" >&2
  exit 2
fi

TRAIN_JOB="$(sbatch --parsable \
  --job-name=competition-gc001 \
  --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 \
  --gres=gpu:8 --cpus-per-task=32 --mem=128G --time=72:00:00 \
  --exclusive --dependency="afterok:${GC100_JOB}" \
  --output="$REPORT/slurm/train-%j.out" --error="$REPORT/slurm/train-%j.err" \
  --chdir="$REPO" \
  --wrap="export OMP_NUM_THREADS=4 TASK_ARM=comp_gc001; srun \"$TORCHRUN\" --standalone --nnodes=1 --nproc_per_node=8 -m scripts.train_object_locus_competition_gc001 --arm comp_gc001")"

EVAL_JOB="$(sbatch --parsable \
  --job-name=competition-four-arm-eval \
  --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=48:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --output="$REPORT/slurm/eval-%j.out" --error="$REPORT/slurm/eval-%j.err" \
  --chdir="$REPO" \
  --wrap="export TASK_OFFICIAL_PYTHON=\"$OFFICIAL_PYTHON\"; \"$PYTHON\" -m scripts.eval_object_locus_gc_competition_four_arm && \"$OFFICIAL_PYTHON\" -m scripts.summarize_object_locus_gc_competition_four_arm")"

"$PYTHON" - "$REPORT/job_receipt.json" "$GC100_JOB" "$TRAIN_JOB" "$EVAL_JOB" "$REPO" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out, gc100, train, evaluation, repo = sys.argv[1:]
record = {
    "created_unix": int(time.time()),
    "gc100_job_id": gc100,
    "training_job_id": train,
    "training_dependency": f"afterok:{gc100}",
    "evaluation_job_id": evaluation,
    "evaluation_dependency": f"afterok:{train}",
    "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
    "training_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
}
Path(out).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
PY
