#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
OUTPUT=${TASK_OUTPUT:?}
MODE=${TASK_MODE:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
[[ "$(hostname -s)" == 3dimage-17 ]]
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
if [[ "$MODE" == preflight ]]; then
  for ARM in pred_joint pred_recon gt_recon; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_vggt_overfit_diagnostic_v1.py \
      --arm "$ARM" --mode smoke --output "$OUTPUT/$ARM"
  done
else
  [[ "$MODE" == train ]]
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_vggt_overfit_diagnostic_v1.py \
    --arm "${TASK_ARM:?}" --mode train --output "$OUTPUT"
fi
