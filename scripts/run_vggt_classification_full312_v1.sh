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
 preflight|evaluate)
  EXTRA=()
  if [[ "$TASK_MODE" == preflight ]]; then
   EXTRA=(--smoke)
  else
   /usr/bin/python3 - "${TASK_PREFLIGHT:?}/PREFLIGHT_VERIFIED.json" "$TASK_CODE_SHA" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='FULL312_EVALUATION_PREFLIGHT_PASSED'
assert p['identity']['evaluation_execution_sha']==sys.argv[2]
assert p['native_scoring_contracts_passed'] and p['optimizer_updates']==0
PY
  fi
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/eval_vggt_classification_full312_v1.py --mode export --root "$OUTPUT" "${EXTRA[@]}"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/eval_vggt_classification_full312_v1.py --mode score --root "$OUTPUT" "${EXTRA[@]}"
  "$OFFICIAL" scripts/eval_vggt_classification_full312_v1.py --mode reduce --root "$OUTPUT" "${EXTRA[@]}"
  ;;
 reference|comparison)
  "$OFFICIAL" scripts/eval_vggt_classification_full312_v1.py --mode "$TASK_MODE" --root "$OUTPUT"
  ;;
 *) exit 2 ;;
esac
