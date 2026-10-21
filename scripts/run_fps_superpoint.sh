#!/bin/bash
set -euo pipefail
R="$1"
MODE="$2"
FULL="$3"
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset RG_IOUHEAD RG_ADAPT RG_ISO RG_EXTRA RG_DEPTH RG_TSEM
export RG_SUPERVISED=6,12
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
stage=preflight
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$R/JOB_FAILED.json"' ERR
[[ "$MODE" == smoke || "$MODE" == full ]]
EXPECTED=$(cat "$R/code.sha")
[[ "$(git rev-parse HEAD)" == "$EXPECTED" ]]
git diff --quiet HEAD -- tokengs scripts tests
if [[ "$MODE" == full ]]; then
  "$PY" scripts/submit_fps_superpoint_after_smoke.py --check-gate "$FULL"
fi
NGPU=$("$PY" -c 'import torch;print(torch.cuda.device_count())')
[[ "$NGPU" -eq 8 ]]
EXTRA=(--epoch-probes)
if [[ "$MODE" == smoke ]]; then EXTRA=(--max-updates 2); fi
stage=train
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_unified_query_v1.py --architecture dual_fps1024_superpoint --output-dir "$R" \
  --epochs 16 --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 \
  --checkpoint-every 50 --no-epoch-checkpoints "${EXTRA[@]}" \
  --initial-checkpoint "$R/checkpoint_untrained_initial.pt" \
  --initial-sha256 "$(cat "$R/untrained_initial.sha256")" --initialize-fps-superpoint-from-legacy-zero
if [[ "$MODE" == smoke ]]; then
  stage=export_smoke
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    scripts/export_unified_query_checkpoint.py --checkpoint "$R/checkpoint_latest.pt" --output "$R/export_smoke" --smoke
  stage=verify_smoke
  "$PY" scripts/verify_fps_superpoint_smoke.py "$R"
  stage=submit_full_after_pass
  "$PY" scripts/submit_fps_superpoint_after_smoke.py "$FULL"
fi
