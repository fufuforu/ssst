#!/usr/bin/env bash
#SBATCH --job-name=vggt-pf-official-eval
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=12:00:00
set -euo pipefail
REPO=${TASK_REPO:?}
ROOT=${EVAL_ROOT:?}
SHARDS=${EVAL_SHARDS:-7}
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
OFFICIAL_PYTHON=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$ROOT/evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
mkdir -p "$ROOT"
EXTRA=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then EXTRA+=(--limit "$EVAL_LIMIT"); fi
if [[ "${EVAL_SCORE_ONLY:-0}" != 1 ]]; then
printf 'stage=export job=%s sha=%s shards=%s time=%s\n' "$SLURM_JOB_ID" "$TASK_CODE_SHA" "$SHARDS" "$(date -Is)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$SHARDS" scripts/eval_object_locus_frozen_vggt_posefree_v1.py \
 --checkpoint /space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor/checkpoint_epoch_08.pt \
 --manifest /space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json \
 --cohort full_validation --output-root "$ROOT" --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 \
 --artifact-manifest "$REPO/vggt_artifact_manifest.json" --official-png-export --shards "$SHARDS" --resume-export "${EXTRA[@]}"
fi
SCORE_EXTRA=()
if [[ "${EVAL_GPU_SEGMENTATION:-0}" == 1 ]]; then SCORE_EXTRA+=(--segmentation-device cuda); fi
printf 'stage=score time=%s\n' "$(date -Is)"
"$OFFICIAL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$SHARDS" scripts/score_object_locus_posefree_official.py --root "$ROOT" --shards "$SHARDS" "${SCORE_EXTRA[@]}"
EXTRA=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then EXTRA+=(--partial); fi
printf 'stage=reduce time=%s\n' "$(date -Is)"
"$OFFICIAL_PYTHON" scripts/score_object_locus_posefree_official.py --root "$ROOT" --shards "$SHARDS" --reduce "${EXTRA[@]}" "${SCORE_EXTRA[@]}"
printf 'stage=complete time=%s\n' "$(date -Is)"
