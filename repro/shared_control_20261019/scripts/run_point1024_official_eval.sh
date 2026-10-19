#!/bin/bash
set -euo pipefail
[[ $# == 3 ]] || exit 2
run_root="$1"
training_code="$2"
control_code="$3"
cd "$training_code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=evaluation_preflight
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/EVALUATION_FAILED.json"' ERR
/usr/bin/python3 "$control_code/scripts/point1024_eval_flexible_dispatch.py" --runtime-eval-check --run "$run_root" --snapshot "$training_code"
stage=original_snapshot_export128
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  "$training_code/scripts/export_unified_query_checkpoint.py" --checkpoint "$run_root/checkpoint_latest.pt" --output "$run_root/evaluation"
stage=unchanged_official_score11
/space/mawb/SIU3R/.venv_gpu_v4/bin/python "$control_code/scripts/score_dual_stream_v1.py" --run-root "$run_root"
stage=validate_128_and_11_metrics
/space/mawb/anaconda3/envs/tokengs/bin/python "$control_code/scripts/dual_stream_evaluation_checks.py" --check-complete "$run_root"
