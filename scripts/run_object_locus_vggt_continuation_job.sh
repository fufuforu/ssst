#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
BRANCH=${TASK_ARM:?}
WORLD=${TASK_WORLD_SIZE:-4}
NODE=${TASK_NODE:-3dimage-13}
REPORT=/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_v1
RUN=/space/mawb/ssst/workspace_group_plus/object_locus_vggt_joint_continuation_ab_v1/$BRANCH
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
ATTEMPT="$REPORT/$BRANCH/attempts/$SLURM_JOB_ID"
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$ATTEMPT"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
[[ "$(hostname -s)" == "$NODE" ]]
[[ "$WORLD" == 4 || "$WORLD" == 8 ]]
[[ "$BRANCH" == frozen || "$BRANCH" == joint_aa ]]
mkdir -p "$ATTEMPT"
COMMON=(--run-training --joint-continuation-branch "$BRANCH" --joint-continuation-world-size "$WORLD" --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$REPO/vggt_artifact_manifest.json")
SMOKE=four_smoke
if [[ "$WORLD" == 8 ]]; then SMOKE=eight_smoke; fi
printf 'stage=%s arm=%s job=%s sha=%s time=%s\n' "$SMOKE" "$BRANCH" "$SLURM_JOB_ID" "$TASK_CODE_SHA" "$(date -Is)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --joint-continuation-mode "$SMOKE" --run-dir "$ATTEMPT/$SMOKE"
EXTRA=()
if [[ -f "$RUN/checkpoint_latest.pt" ]]; then EXTRA+=(--resume); fi
printf 'stage=formal_continuation2 arm=%s time=%s\n' "$BRANCH" "$(date -Is)"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" scripts/train_object_locus_frozen_vggt_posefree_v1.py "${COMMON[@]}" --run-dir "$RUN" "${EXTRA[@]}"
