#!/usr/bin/env bash
#SBATCH --job-name=vggt-pf-8x3090
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
set -euo pipefail

REPO=${TASK_REPO:?submit wrapper must provide TASK_REPO}
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
ASSET_ROOT=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
MANIFEST=/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json
SOURCE=/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt
RUN_DIR=/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1
REPORT=/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1
ARTIFACT="$REPO/vggt_artifact_manifest.json"
JOB_SMOKE="$REPORT/smoke/${SLURM_JOB_ID}"
SINGLE_REPORT="$JOB_SMOKE/single/smoke_report.json"
EIGHT_REPORT="$JOB_SMOKE/eight/smoke_report.json"

mkdir -p "$JOB_SMOKE"
export HF_HUB_CACHE="$ASSET_ROOT/hf_cache/hub"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET_ROOT/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"

printf 'job=%s node=%s visible_gpus=%s code_sha=%s\n' \
  "$SLURM_JOB_ID" "$(hostname -s)" "$CUDA_VISIBLE_DEVICES" "$TASK_CODE_SHA"
[[ "$(hostname -s)" == 3dimage-13 ]]
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
IFS=',' read -r -a allocated_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#allocated_gpus[@]}" -eq 8 ]]
free -h
[[ "$(/space/mawb/anaconda3/envs/tokengs/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["hf_revision"])' "$ARTIFACT")" == 860abec7937da0a4c03c41d3c269c366e82abdf9 ]]

single_gpu=${CUDA_VISIBLE_DEVICES%%,*}
printf 'stage=single_smoke_start gpu=%s time=%s\n' "$single_gpu" "$(date -Is)"
env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
  CUDA_VISIBLE_DEVICES="$single_gpu" HF_HUB_CACHE="$HF_HUB_CACHE" HF_HUB_OFFLINE=1 \
  PYTHONPATH="$PYTHONPATH" OMP_NUM_THREADS=4 "$PYTHON" -u \
  scripts/smoke_object_locus_frozen_vggt_posefree_v1.py --single-card-real \
  --manifest "$MANIFEST" --checkpoint "$SOURCE" --output-dir "$JOB_SMOKE/single" \
  --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$ARTIFACT"
printf 'stage=single_smoke_pass time=%s\n' "$(date -Is)"

printf 'stage=eight_smoke_start time=%s\n' "$(date -Is)"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/smoke_object_locus_frozen_vggt_posefree_v1.py --eight-card-real \
  --manifest "$MANIFEST" --checkpoint "$SOURCE" --output-dir "$JOB_SMOKE/eight" \
  --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$ARTIFACT"
printf 'stage=eight_smoke_pass time=%s\n' "$(date -Is)"
export POSEFREE_SINGLE_SMOKE_REPORT="$SINGLE_REPORT"
export POSEFREE_EIGHT_SMOKE_REPORT="$EIGHT_REPORT"

TRAIN_ARGS=(--run-training --manifest "$MANIFEST" --checkpoint "$SOURCE" --run-dir "$RUN_DIR" \
  --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$ARTIFACT")
if [[ "${TASK_RESUME:-0}" == 1 ]]; then TRAIN_ARGS+=(--resume); fi
printf 'stage=formal_training_start resume=%s time=%s\n' "${TASK_RESUME:-0}" "$(date -Is)"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_object_locus_frozen_vggt_posefree_v1.py "${TRAIN_ARGS[@]}"
