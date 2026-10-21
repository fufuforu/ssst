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
OFFICIAL=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
case "${TASK_MODE:?}" in
 single_smoke|four_smoke)
  WORKERS=4
  if [[ "$TASK_MODE" == single_smoke ]]; then WORKERS=1; fi
  for ARM in bidirectional forward_only gated_feedback independent_feedback; do
   "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" scripts/train_vggt_interaction_controls_v1.py \
    --arm "$ARM" --mode "$TASK_MODE" --output "$OUTPUT/$ARM"
   "$OFFICIAL" scripts/eval_vggt_interaction_controls_v1.py --mode smoke_score \
    --root "$OUTPUT/$ARM/evaluation_preflight" --run-root "$OUTPUT" --training-sha "$TASK_CODE_SHA"
  done
  ;;
 train)
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_vggt_interaction_controls_v1.py \
   --arm "${TASK_ARM:?}" --mode train --output "$OUTPUT"
  ;;
 eval)
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_interaction_controls_v1.py \
   --mode export --arm "$TASK_ARM" --root "$OUTPUT" --run-root "${TASK_RUN_ROOT:?}" --training-sha "$TASK_CODE_SHA"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_interaction_controls_v1.py \
   --mode score --arm "$TASK_ARM" --root "$OUTPUT" --run-root "$TASK_RUN_ROOT" --training-sha "$TASK_CODE_SHA"
  ;;
 reduce)
  exec "$OFFICIAL" scripts/eval_vggt_interaction_controls_v1.py \
   --mode reduce --root "$OUTPUT" --run-root "$TASK_RUN_ROOT" --training-sha "$TASK_CODE_SHA"
  ;;
 *) exit 2 ;;
esac
