#!/bin/bash
set -euo pipefail
architecture="$1"
run_root="$2"
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=short_training
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_unified_query_v1.py --architecture "$architecture" --output-dir "$run_root" --understanding-weight .05 --short-validation --checkpoint-every 32
stage=short_summary
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_query_short.py --run "$run_root"
