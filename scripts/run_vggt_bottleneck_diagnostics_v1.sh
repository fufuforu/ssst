#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
OUTPUT=${TASK_OUTPUT:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
EXTRA=()
case "${TASK_MODE:?}" in
 raw_smoke|raw)
  if [[ "$TASK_MODE" == raw_smoke ]]; then EXTRA+=(--smoke); fi
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/probe_vggt_raw_query_readout_v1.py \
   --expanded-root /space/mawb/ssst/group_plus/vggt_joint_followups_v1/attempts/d2dc0ced0bacd8e400b4d8ed1503f3bf8806804b/formal/expanded \
   --cohort /space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/cpu_review_20261013/raw_readout_cohort.json \
   --output "$OUTPUT" "${EXTRA[@]}"
  ;;
 spike_smoke)
  for ARM in A B C; do
   export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/$ARM/geometry_evidence"
   "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/diagnose_vggt_actual_gradient_spikes_v1.py \
    --arm "$ARM" --output "$OUTPUT/$ARM" --smoke
  done
  ;;
 spike)
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/diagnose_vggt_actual_gradient_spikes_v1.py \
   --arm "${TASK_ARM:?}" --output "$OUTPUT"
  ;;
 *) exit 2 ;;
esac
