#!/usr/bin/env bash
set -euo pipefail
cd "${TASK_REPO:?}"
[[ "$(git rev-parse HEAD)" == "${TASK_CODE_SHA:?}" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /space/mawb/anaconda3/envs/tokengs/bin/python scripts/audit_vggt_interaction_startup_v1.py
