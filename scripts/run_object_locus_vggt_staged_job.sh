#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
REPORT=/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1
RUN=/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
ATTEMPT="$REPORT/attempts/$SLURM_JOB_ID"
export HF_HUB_CACHE="$ASSET/hf_cache/hub"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$ATTEMPT"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
case "$(hostname -s)" in
  3dimage-11|3dimage-13|3dimage-14|3dimage-17|3dimage-18) ;;
  *) echo 'node outside the authorized five-node pool' >&2; exit 2 ;;
esac
mkdir -p "$ATTEMPT"
printf 'job=%s mode=%s sha=%s time=%s\n' "$SLURM_JOB_ID" "$TASK_MODE" "$TASK_CODE_SHA" "$(date -Is)"
COMMON=(--run-training --staged-vggt-adapt --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$REPO/vggt_artifact_manifest.json")
if [[ "$TASK_MODE" == single_smoke ]]; then
  exec "$PYTHON" scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --staged-mode single_smoke --run-dir "$ATTEMPT/single_smoke"
fi
[[ "$TASK_MODE" == train ]]
"$PYTHON" - "$TASK_SINGLE_REPORT" <<'PY'
import json,sys
from scripts.object_locus_frozen_vggt_posefree_runtime import STAGED_RECIPE,staged_training_configuration
d=json.load(open(sys.argv[1]))
assert d['status']=='GPU_SMOKE_COMPLETED' and d['mode']=='single_smoke' and d['world_size']==1
assert d['recipe']==STAGED_RECIPE
# Reuse the completed node13 proof, comparing every scientific configuration
# field exactly while allowing the newly authorized hardware pool metadata.
previous=dict(d['training_configuration']);current=staged_training_configuration()
for field in ('node','gpu_model'):
    previous.pop(field);current.pop(field)
assert previous==current
PY
printf 'stage=eight_smoke time=%s\n' "$(date -Is)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --staged-mode eight_smoke --run-dir "$ATTEMPT/eight_smoke"
EXTRA=()
if [[ -f "$RUN/checkpoint_latest.pt" ]]; then EXTRA+=(--resume); fi
printf 'stage=formal_adapt2_then_joint4 time=%s\n' "$(date -Is)"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --run-dir "$RUN" "${EXTRA[@]}"
