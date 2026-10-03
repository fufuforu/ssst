#!/usr/bin/env bash
# Submit exactly one formal run (or interruption-only resume) plus its fixed evaluations.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPORT_ROOT=/space/mawb/ssst/group_plus/official_source_scannet_v1
PYTHON_BIN=/space/mawb/anaconda3/envs/tokengs/bin/python
cd "$REPO_ROOT"
mkdir -p "$REPORT_ROOT"
MODE=${1:-fresh}
if [[ "$MODE" != fresh && "$MODE" != resume ]]; then exit 2; fi
"$PYTHON_BIN" - <<'PY'
import json,shutil,subprocess
from scripts.official_locusgs_recon_runtime import REPORT,REPO,source_sha
p=json.loads((REPORT/'training_authorization.json').read_text())
assert p['training_sha']==source_sha() and p['remote_verified'] and p['smoke_passed'] and p['cpu_contracts_passed']
assert shutil.disk_usage(REPO).free>=60*1024**3
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=REPO,text=True).split()[0]
assert remote==source_sha(), 'Final tested SHA must be the remote main at initial submission'
PY
if [[ "$MODE" == fresh && -e "$REPORT_ROOT/job_ids.json" ]]; then
  echo 'Existing formal job registration: no second fresh run permitted' >&2
  exit 1
fi
RESUME_ARG=
if [[ "$MODE" == resume ]]; then RESUME_ARG=--resume; fi
TRAIN_JOB=$(sbatch --parsable --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 --job-name=official-lgs-50k --chdir="$REPO_ROOT" --output="$REPORT_ROOT/train-%j.out" --error="$REPORT_ROOT/train-%j.err" --wrap="$PYTHON_BIN -u scripts/train_official_locusgs_recon.py $RESUME_ARG")
VAL32_MANIFEST=$REPORT_ROOT/val32_manifest.json
# The historical arm performs zero optimizer updates, on a separate GPU allocation.
LEGACY_JOB=$(sbatch --parsable --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 --job-name=official-lgs-legacy --chdir="$REPO_ROOT" --output="$REPORT_ROOT/legacy-%j.out" --error="$REPORT_ROOT/legacy-%j.err" --wrap="$PYTHON_BIN -u scripts/eval_official_locusgs_recon.py --kind legacy && $PYTHON_BIN -u scripts/eval_official_locusgs_recon.py --kind legacy --manifest $VAL32_MANIFEST --output $REPORT_ROOT/eval_legacy_best47500_val32")
NEW_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 --job-name=official-lgs-new-eval --chdir="$REPO_ROOT" --output="$REPORT_ROOT/new-eval-%j.out" --error="$REPORT_ROOT/new-eval-%j.err" --wrap="$PYTHON_BIN -u scripts/eval_official_locusgs_recon.py --kind official --step 50000 && $PYTHON_BIN -u scripts/eval_official_locusgs_recon.py --kind official --step 47500 --manifest $VAL32_MANIFEST --output $REPORT_ROOT/eval_new_step47500_val32 && $PYTHON_BIN -u scripts/eval_official_locusgs_recon.py --kind official --step 50000 --manifest $VAL32_MANIFEST --output $REPORT_ROOT/eval_new_step50000_val32")
PACKAGE_JOB=$(sbatch --parsable --dependency="afterok:$NEW_JOB:$LEGACY_JOB" --partition=3090 --nodelist=3dimage-11 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --job-name=official-lgs-package --chdir="$REPO_ROOT" --output="$REPORT_ROOT/package-%j.out" --error="$REPORT_ROOT/package-%j.err" --wrap="$PYTHON_BIN -u scripts/package_official_locusgs_recon.py")
"$PYTHON_BIN" - "$TRAIN_JOB" "$LEGACY_JOB" "$NEW_JOB" "$PACKAGE_JOB" "$MODE" <<'PY'
import sys
from scripts.official_locusgs_recon_runtime import REPORT,source_sha,write_json
p=dict(training_sha=source_sha(),train_job=sys.argv[1],legacy_eval_job=sys.argv[2],new_eval_job=sys.argv[3],package_job=sys.argv[4],mode=sys.argv[5])
write_json(REPORT/'job_ids.json',p)
print(p)
PY
