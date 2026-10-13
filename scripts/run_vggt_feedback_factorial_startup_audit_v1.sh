#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
EVIDENCE=${TASK_EVIDENCE:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
exec /space/mawb/anaconda3/envs/tokengs/bin/python scripts/audit_vggt_feedback_factorial_startup_v1.py \
 --evidence "$EVIDENCE" --wait-for-receipts
