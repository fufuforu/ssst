#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
OUTPUT=${TASK_OUTPUT:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TASK_GPU_CLASS=3090
export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
 scripts/train_vggt_understanding_probes_v1.py --arm u_fusion --mode four_smoke --output "$OUTPUT"
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/eval_vggt_understanding_probes_v1.py \
 --mode smoke_score --root "$OUTPUT/evaluation_preflight" --run-root "$OUTPUT" --training-sha "$TASK_CODE_SHA"
