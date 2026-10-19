#!/bin/bash
set -euo pipefail
[[ $# == 1 ]] || exit 2
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?isolated code snapshot required}"
[[ "${SLURMD_NODENAME:-}" == 3dimage-17 ]] || exit 2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=source_preflight
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/submit_concat_shared_full.py --verify-only --run "$run_root"
stage=concat_shared_full4_training
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_unified_query_v1.py \
  --architecture concat_shared --output-dir "$run_root" --epochs 4 \
  --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 \
  --checkpoint-every 50 --epoch-probes --config "$run_root/model_config.json" \
  --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" \
  --initial-sha256 "$(cat "$run_root/untrained_initial.sha256")"
