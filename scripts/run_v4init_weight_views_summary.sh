#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
R="$1"
trap 'code=$?; printf "{\"status\":\"FAILED\",\"stage\":\"summary\",\"exit_code\":%s}\n" "$code" > "$R/SUMMARY_FAILED.json"' ERR
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" ]]
git diff --quiet HEAD -- scripts tokengs tests
/usr/bin/python3 scripts/summarize_v4init_weight_views.py
