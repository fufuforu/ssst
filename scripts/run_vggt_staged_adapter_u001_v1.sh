#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
REPORT=/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1
RUN=/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
ATTEMPT="$REPORT/attempts/$SLURM_JOB_ID"
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$ATTEMPT"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
[[ "$(hostname -s)" == 3dimage-13 ]]
mkdir -p "$ATTEMPT"
COMMON=(--run-training --staged-vggt-adapt --staged-world-size 4 --staged-adapter-understanding-scale 0.01 --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$REPO/vggt_artifact_manifest.json")
printf 'job=%s mode=%s sha=%s time=%s\n' "$SLURM_JOB_ID" "$TASK_MODE" "$TASK_CODE_SHA" "$(date -Is)"
if [[ "$TASK_MODE" == single_smoke ]]; then
  exec "$PYTHON" scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --staged-mode single_smoke --run-dir "$ATTEMPT/single_smoke"
fi
[[ "$TASK_MODE" == train ]]
"$PYTHON" - "$TASK_SINGLE_REPORT" "$TASK_CODE_SHA" <<'PY'
import json,sys
from scripts.object_locus_frozen_vggt_posefree_runtime import staged_training_configuration
d=json.load(open(sys.argv[1]))
assert d['status']=='GPU_SMOKE_COMPLETED' and d['mode']=='single_smoke' and d['world_size']==1
assert d['formal_updates']==0 and d['execution_git_sha']==sys.argv[2]
assert d['training_configuration']==staged_training_configuration(4,.01)
PY
printf 'stage=fresh_four_card_smoke time=%s\n' "$(date -Is)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --staged-mode four_smoke --run-dir "$ATTEMPT/four_smoke"
"$PYTHON" - "$ATTEMPT/four_smoke/smoke_report.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['status']=='GPU_SMOKE_COMPLETED' and d['formal_updates']==0 and d['world_size']==4
assert d['exposures_per_phase']==16 and len(d['rank_rows'])==4
assert all(row['adapter_understanding_scale']==.01 for update in d['rank_rows'] for row in update)
assert any(6923 in row['window_indices'] for update in d['rank_rows'][:2] for row in update)
assert all(row['vggt_probe_change_norm']==0 for update in d['rank_rows'][2:] for row in update)
PY
EXTRA=(--staged-boundary-checkpoint /space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_reconstruction_adaptation_epoch_02.pt)
if [[ -f "$RUN/checkpoint_latest.pt" ]]; then EXTRA=(--resume); fi
printf 'stage=formal_frozen_joint4_from_exact_adapt2_boundary time=%s\n' "$(date -Is)"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --run-dir "$RUN" "${EXTRA[@]}"
