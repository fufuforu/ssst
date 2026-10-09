#!/usr/bin/env bash
#SBATCH --job-name=vggt-pf-monitor-8x3090
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
RUN_DIR=/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor
REPORT=/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2_monitor
ARTIFACT="$REPO/vggt_artifact_manifest.json"
ATTEMPT="$REPORT/attempts/${SLURM_JOB_ID}"
JOB_SMOKE="$ATTEMPT/smoke"
HISTORICAL=/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2/attempts/59658
WINDOW_REPORT="$HISTORICAL/window4253"
SINGLE_REPORT="$HISTORICAL/smoke/single/smoke_report.json"
EIGHT_REPORT="$JOB_SMOKE/eight/smoke_report.json"

mkdir -p "$ATTEMPT" "$REPORT/slurm" "$JOB_SMOKE"
export POSEFREE_V2_EVIDENCE_DIR="$ATTEMPT"
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

ATTEMPT_ROOT="$ATTEMPT" TASK_CODE_SHA="$TASK_CODE_SHA" TASK_JOB_ID="$SLURM_JOB_ID" \
  "$PYTHON" - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ['ATTEMPT_ROOT'])
record={'slurm_job_id':os.environ['TASK_JOB_ID'],'execution_git_sha':os.environ['TASK_CODE_SHA'],
        'calibration_protocol':'shared_context_depth_sim3_v2','geometry_quality_policy':'monitor_v1',
        'single_smoke_reused':True,'single_smoke_execution_sha':'d4107c881b0c5ce4e0bb187f620d0607e9e41454','node':'3dimage-13',
        'gpu':'RTX3090','world_size':8,'microbatch_per_rank':1,'accumulation':1}
(root/'attempt_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
jobs=root.parent/'jobs.json';rows=json.loads(jobs.read_text()) if jobs.is_file() else []
rows.append(record);tmp=jobs.with_suffix('.json.tmp');tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(jobs)
PY

# Reuse the strict-policy 59658 single smoke; no computation/precision/gradient change.
# The historical --window-4253-calibration and single smoke evidence remain immutable.
printf 'stage=single_smoke_reused job=59658 sha=d4107c881b0c5ce4e0bb187f620d0607e9e41454 policy=strict_v2_historical\n'
[[ -f "$SINGLE_REPORT" && -f "$WINDOW_REPORT/window4253_context_sim3_v2.json" ]]

printf 'stage=eight_smoke_start time=%s\n' "$(date -Is)"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/smoke_object_locus_frozen_vggt_posefree_v1.py --eight-card-real \
  --manifest "$MANIFEST" --checkpoint "$SOURCE" --output-dir "$JOB_SMOKE/eight" \
  --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$ARTIFACT"
printf 'stage=eight_smoke_pass time=%s\n' "$(date -Is)"
export POSEFREE_SINGLE_SMOKE_REPORT="$SINGLE_REPORT"
export POSEFREE_EIGHT_SMOKE_REPORT="$EIGHT_REPORT"

TRAIN_ARGS=(--run-training --manifest "$MANIFEST" --checkpoint "$SOURCE" --run-dir "$RUN_DIR" \
  --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest "$ARTIFACT" \
  --calibration-report "$WINDOW_REPORT/window4253_context_sim3_v2.json" \
  --single-smoke-report "$SINGLE_REPORT" --eight-smoke-report "$EIGHT_REPORT")
if [[ "${TASK_RESUME:-0}" == 1 ]]; then TRAIN_ARGS+=(--resume); fi
printf 'stage=formal_training_start resume=%s time=%s\n' "${TASK_RESUME:-0}" "$(date -Is)"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_object_locus_frozen_vggt_posefree_v1.py "${TRAIN_ARGS[@]}"
