#!/bin/bash
set -euo pipefail
R="$1"
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
unset RG_IOUHEAD RG_EXTRA RG_ADAPT RG_ISO RG_DEPTH RG_TSEM
export RG_SUPERVISED=6,12
stage=preflight
trap 'code=$?; printf "{\"status\":\"FAILED\",\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$R/JOB_FAILED.json"' ERR
trap 'printf "{\"status\":\"FAILED\",\"stage\":\"scheduler_signal\",\"exit_code\":143}\n" > "$R/JOB_FAILED.json"; exit 143' TERM
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" ]]
git diff --quiet HEAD -- tokengs scripts tests
stage=train
# The wrapper itself imports only stdlib; no retry after STARTED or any update.
/usr/bin/python3 scripts/run_with_io_bootstrap_retry.py --run-root "$R" -- \
 /space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
 scripts/train_unified_query_v1.py --architecture dual_point1024_isoadapter --output-dir "$R" \
 --config "$R/model_config.json" --epochs 4 --learning-rate .0001 \
 --understanding-weight .05 --understanding-warmup 1043 --checkpoint-every 50 --epoch-probes \
 --training-window-manifest /space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_training2ctx4novel_20261021_v1/training_windows.json \
 --initial-checkpoint "$R/checkpoint_untrained_initial.pt" --initial-sha256 "$(cat "$R/untrained_initial.sha256")" \
 --initialize-point1024-isoadapter-from-legacy-zero
