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
 expanded_smoke)
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_feedback_expanded_v1.py --mode smoke_export --root "$OUTPUT"
  exec "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_feedback_expanded_v1.py --mode smoke_score --root "$OUTPUT"
  ;;
 expanded)
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_feedback_expanded_v1.py --mode export --root "$OUTPUT" --arm "${TASK_ARM:?}"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_feedback_expanded_v1.py --mode score --root "$OUTPUT" --arm "$TASK_ARM"
  exec "$OFFICIAL" scripts/eval_vggt_feedback_expanded_v1.py --mode reduce --root "$OUTPUT" --arm "$TASK_ARM"
  ;;
 comparison)
  exec "$OFFICIAL" scripts/eval_vggt_feedback_expanded_v1.py --mode comparison --root "$OUTPUT"
  ;;
 gradient_smoke|gradient)
  EXTRA=()
  if [[ "$TASK_MODE" == gradient_smoke ]]; then EXTRA+=(--smoke); fi
  for ARM in A B C; do
   export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/$ARM/geometry_evidence"
   "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/diagnose_vggt_joint_gradient_spikes_v1.py --arm "$ARM" --output "$OUTPUT/$ARM" "${EXTRA[@]}"
  done
  if [[ "$TASK_MODE" == gradient ]]; then
   exec /usr/bin/python3 scripts/analyze_vggt_joint_gradient_replay_v1.py --root "$OUTPUT"
  fi
  ;;
 *) exit 2 ;;
esac
