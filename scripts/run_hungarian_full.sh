#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"hungarian_full4\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_unified_query_v1.py --architecture dual_superpoint --output-dir "$run_root" \
  --epochs 4 --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 \
  --checkpoint-every 50 --epoch-probes --config "$run_root/hungarian_config.json" \
  --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" \
  --initial-sha256 "$(cat "$run_root/untrained_initial.sha256")"
