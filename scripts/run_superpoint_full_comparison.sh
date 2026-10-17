#!/bin/bash
set -euo pipefail
b_run="$1"
a_run="$2"
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
reference=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu
extra_args=()
if [[ -f "$reference/COMPLETE.json" ]]; then
    extra_args=(--run "unified_query_full4=$reference")
fi
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$a_run"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$b_run"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/compare_gfm_native_runs.py --run "A_superpoint_dual_full4=$a_run" --run "B_unified_direct_mask_full4=$b_run" "${extra_args[@]}" --output "$b_run/comparison_A_B"
