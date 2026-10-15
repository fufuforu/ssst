#!/usr/bin/env bash
set -euo pipefail
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$TASK_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GSPLAT_PRECOMPILED_SO=/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so
export POSEFREE_V2_EVIDENCE_DIR="$TASK_OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$TASK_REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nproc_per_node="$TASK_WORKERS" \
  scripts/diagnose_vggt_ownership_v1.py --root "$TASK_OUTPUT"
