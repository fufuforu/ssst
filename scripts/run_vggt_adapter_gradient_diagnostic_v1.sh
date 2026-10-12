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
OFFICIAL=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
case "$MODE" in
  probe)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/probe_vggt_understanding_gradients_v1.py --output "$OUTPUT"
    "$PYTHON" scripts/summarize_vggt_understanding_gradients_v1.py --root "$OUTPUT"
    ;;
  preflight)
    for SCALE in 0.01 0; do
      "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_vggt_overfit_diagnostic_v1.py \
        --arm pred_joint --mode smoke --gradient-routing-diagnostic --adapter-understanding-scale "$SCALE" --output "$OUTPUT/scale_$SCALE"
    done
    ;;
  train)
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_vggt_overfit_diagnostic_v1.py \
      --arm pred_joint --mode train --gradient-routing-diagnostic --adapter-understanding-scale "${TASK_ADAPTER_SCALE:?}" --output "$OUTPUT"
    ;;
  eval)
    mkdir -p "$OUTPUT"
    "$PYTHON" scripts/eval_vggt_adapter_gradient_diagnostic_v1.py --root "$OUTPUT" --run-root "${TASK_RUN_ROOT:?}" \
      --training-sha "${TASK_TRAINING_SHA:?}" --validate-checkpoints
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_adapter_gradient_diagnostic_v1.py \
      --root "$OUTPUT" --run-root "$TASK_RUN_ROOT" --training-sha "$TASK_TRAINING_SHA"
    "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/score_vggt_overfit_diagnostic_v1.py \
      --root "$OUTPUT" --arms adapter_u001 adapter_u0
    "$PYTHON" scripts/summarize_vggt_adapter_gradient_diagnostic_v1.py --root "$OUTPUT"
    ;;
  *) exit 2 ;;
esac
