#!/usr/bin/env bash
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/object_locus_v1_1/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/group_plus/object_locus_v1_1/slurm-%j.err

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "train" ) ]]; then
  echo "usage: sbatch scripts/submit_object_locus_v1.sh {smoke|train}" >&2
  exit 2
fi

PHASE="$1"
REPO=/space/mawb/ssst_object_locus_v1
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
REPORTS=/space/mawb/ssst/group_plus/object_locus_v1_1
RUN_ROOT=/space/mawb/ssst/workspace_group_plus/object_locus_v1_1

cd "$REPO"
if [[ "$PHASE" == "smoke" ]]; then
  exec "$PYTHON" -u scripts/smoke_object_locus_v1.py --device cuda
fi

mkdir -p "$RUN_ROOT"
"$PYTHON" -u scripts/train_object_locus_v1.py \
  --device cuda \
  --reports "$REPORTS" \
  --run-root "$RUN_ROOT" \
  --until-step 5000 \
  --failure-capture-dir "$RUN_ROOT/failures" 2>&1 | tee -a "$RUN_ROOT/train.log"
