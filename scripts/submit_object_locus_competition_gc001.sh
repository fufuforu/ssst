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
RECEIPT="$REPORT/job_receipt.json"
ATTEMPT=1
PREVIOUS_RECEIPT="$("$PYTHON" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
paths=[p for p in root.glob("job_receipt*.json")]
if paths:
    paths.sort(key=lambda p: json.loads(p.read_text()).get("attempt", 1))
    print(paths[-1])
PY
)"
if [[ -n "$PREVIOUS_RECEIPT" ]]; then
  PREVIOUS_TRAIN="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["training_job_id"])' "$PREVIOUS_RECEIPT")"
  PREVIOUS_STATE="$(sacct -n -X -j "$PREVIOUS_TRAIN" --format=State | head -n 1 | xargs)"
  case "$PREVIOUS_STATE" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ;;
    *) echo "previous training attempt is not a terminal failure ($PREVIOUS_STATE); refusing duplicate" >&2; exit 2 ;;
  esac
  ATTEMPT="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("attempt",1)+1)' "$PREVIOUS_RECEIPT")"
  RECEIPT="$REPORT/job_receipt_attempt$(printf '%02d' "$ATTEMPT").json"
  if [[ -e "$RECEIPT" ]]; then
    echo "receipt for retry attempt already exists; inspect it before submitting again" >&2
    exit 2
  fi
fi
ACTIVE_TRAIN="$(squeue -h -u "$USER" -n competition-gc001 -o '%i' | head -n 1)"
if [[ -n "$ACTIVE_TRAIN" ]]; then
  echo "competition training job $ACTIVE_TRAIN is already pending or running" >&2
  exit 2
fi

# If Slurm still knows the predecessor, preserve a live afterok dependency.
# A successfully completed predecessor can be purged from slurmctld while its
# accounting record and verified endpoint remain; in that case the registered
# completion proof satisfies the same success condition and the job is eligible
# for direct submission.
TRAIN_DEP_ARGS=()
TRAIN_DEP_RECORD="afterok:${GC100_JOB}"
if GC100_CONTROLLER="$(scontrol show job "$GC100_JOB" 2>/dev/null)"; then
  [[ "$GC100_CONTROLLER" == *"JobName=gc-sweep-gc100"* ]] || { echo "GC100 controller identity mismatch" >&2; exit 3; }
  [[ "$GC100_CONTROLLER" == *"NodeList=3dimage-11"* ]] || { echo "GC100 node mismatch" >&2; exit 3; }
  [[ "$GC100_CONTROLLER" == *"WorkDir=/space/mawb/ssst_object_locus_gc_sweep_v1"* ]] || { echo "GC100 workdir mismatch" >&2; exit 3; }
  TRAIN_DEP_ARGS+=(--dependency="afterok:${GC100_JOB}")
else
  "$PYTHON" - "$REPORT/gc_predecessor_proof.json" "$GC100_JOB" <<'PY'
import json, sys
from pathlib import Path
proof = json.loads(Path(sys.argv[1]).read_text())
gc = proof.get("gc100", {})
endpoints = {r["arm"]: r for r in proof.get("completed_arms", [])}
checks = endpoints.get("gc100", {}).get("checks", {})
if (gc.get("job_id") != sys.argv[2] or gc.get("state") != "COMPLETED" or
        gc.get("exit_code") != "0:0" or gc.get("alpha") != 1.0 or
        not gc.get("sacct_verified") or
        not all(checks.get(k) for k in ("progress_complete", "progress_updates",
            "progress_exposures", "checkpoint_epoch", "checkpoint_updates",
            "checkpoint_exposures", "source_exposure", "endpoint_exposure",
            "plan_sha256", "alpha_1.0"))):
    raise SystemExit("purged GC100 job lacks a verified successful endpoint proof")
PY
  TRAIN_DEP_RECORD="already-satisfied:${GC100_JOB}:COMPLETED:0:0"
fi

TRAIN_JOB="$(sbatch --parsable \
  --job-name=competition-gc001 \
  --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 \
  --gres=gpu:8 --cpus-per-task=32 --mem=128G --time=72:00:00 \
  --exclusive "${TRAIN_DEP_ARGS[@]}" \
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

"$PYTHON" - "$RECEIPT" "$GC100_JOB" "$TRAIN_JOB" "$EVAL_JOB" "$REPO" "$TRAIN_DEP_RECORD" "$ATTEMPT" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out, gc100, train, evaluation, repo, dependency, attempt = sys.argv[1:]
record = {
    "attempt": int(attempt),
    "created_unix": int(time.time()),
    "gc100_job_id": gc100,
    "training_job_id": train,
    "training_dependency": dependency,
    "evaluation_job_id": evaluation,
    "evaluation_dependency": f"afterok:{train}",
    "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
    "training_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
}
Path(out).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
PY
