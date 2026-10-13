#!/usr/bin/env bash
set -euo pipefail
cd "${TASK_REPO:?}"
[[ "$(git rev-parse HEAD)" == "${TASK_CODE_SHA:?}" ]]
exec /usr/bin/python3 scripts/summarize_vggt_bottleneck_diagnostics_v1.py \
 --mode "${TASK_MODE:?}" --root "${TASK_INPUT:?}" --output "${TASK_OUTPUT:?}"
