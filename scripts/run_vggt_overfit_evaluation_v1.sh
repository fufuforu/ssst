#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
ROOT=${TASK_OUTPUT:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$ROOT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
case "$(hostname -s)" in
  3dimage-11|3dimage-13|3dimage-14|3dimage-17|3dimage-18) ;;
  *) exit 2 ;;
esac
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
OFFICIAL=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_vggt_overfit_diagnostic_v1.py --root "$ROOT"
"$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/score_vggt_overfit_diagnostic_v1.py --root "$ROOT"
"$PYTHON" scripts/summarize_vggt_overfit_evaluation_v1.py --root "$ROOT"
