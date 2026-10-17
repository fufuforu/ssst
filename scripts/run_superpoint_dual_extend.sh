#!/bin/bash
set -euo pipefail
run_root="$1"
variant="${2:-dual_superpoint}"
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"superpoint_epoch4_to8\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_unified_query_v1.py --architecture "$variant" --output-dir "$run_root" --epochs 8 --understanding-weight .05 --understanding-warmup 1043 --learning-rate .00003 --checkpoint-every 50 --epoch-probes --config "$run_root/model_config.json" --continue-checkpoint "$run_root/checkpoint_source_epoch4.pt" --continue-sha256 "$(cat "$run_root/source.sha256")"
