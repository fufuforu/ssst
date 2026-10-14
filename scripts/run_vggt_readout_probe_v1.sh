#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
OUTPUT=${TASK_OUTPUT:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GSPLAT_PRECOMPILED_SO=/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so
export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
OFFICIAL=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
case "${TASK_MODE:?}" in
 preflight)
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/vggt_readout_probe_v1.py --mode preflight --root "$OUTPUT/preflight"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/vggt_readout_probe_v1.py --mode score --root "$OUTPUT/preflight" --smoke
  "$OFFICIAL" scripts/vggt_readout_probe_v1.py --mode reduce --root "$OUTPUT/preflight" --smoke
  ;;
 train_eval)
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${TASK_WORKERS:?}" \
    scripts/vggt_readout_probe_v1.py --mode train_export --root "$OUTPUT"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node="$TASK_WORKERS" \
    scripts/vggt_readout_probe_v1.py --mode score --root "$OUTPUT"
  "$OFFICIAL" scripts/vggt_readout_probe_v1.py --mode reduce --root "$OUTPUT"
  ;;
 summary)
  "$OFFICIAL" scripts/vggt_readout_probe_v1.py --mode summary --root "$OUTPUT"
  ;;
 *) exit 2 ;;
esac
