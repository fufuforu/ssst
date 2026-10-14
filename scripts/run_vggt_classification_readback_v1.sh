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
  for STAGE in single four; do
   WORKERS=1
   if [[ "$STAGE" == four ]]; then WORKERS=4; fi
   "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
    scripts/train_vggt_classification_readback_v1.py --mode "${STAGE}_smoke" --output "$OUTPUT/$STAGE"
  done
  "$OFFICIAL" scripts/verify_vggt_classification_readback_preflight_v1.py --root "$OUTPUT"
  ;;
 train_eval)
  /usr/bin/python3 - "${TASK_PREFLIGHT:?}/PREFLIGHT_VERIFIED.json" "$TASK_CODE_SHA" <<'PY'
import json,sys
proof=json.load(open(sys.argv[1]))
assert proof['status']=='CLASSIFICATION_PREFLIGHT_VERIFIED' and proof['execution_sha']==sys.argv[2]
assert proof['actual1_and4_rank_updates_verified'] and proof['equal_C_D_capacity']
assert proof['formal_updates']==0 and len(proof['official_13_column_scores'])==6
PY
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
   scripts/train_vggt_classification_readback_v1.py --mode train --arm "${TASK_ARM:?}" --output "$OUTPUT/training"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
   scripts/eval_vggt_classification_readback_v1.py --mode export --arm "$TASK_ARM" --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 \
   scripts/eval_vggt_classification_readback_v1.py --mode score --arm "$TASK_ARM" --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  "$OFFICIAL" scripts/eval_vggt_classification_readback_v1.py --mode reduce --arm "$TASK_ARM" --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  ;;
 baseline)
  "$OFFICIAL" scripts/eval_vggt_classification_readback_v1.py --mode baseline --root "$OUTPUT"
  ;;
 summary)
  "$OFFICIAL" scripts/summarize_vggt_classification_readback_v1.py --root "$OUTPUT"
  ;;
 *) exit 2 ;;
esac
