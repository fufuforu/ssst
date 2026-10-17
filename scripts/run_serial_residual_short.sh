#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=short_training
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_unified_query_v1.py --architecture serial_residual --output-dir "$run_root" --understanding-weight .05 --short-validation --checkpoint-every 32 --config "$run_root/model_config.json" --initial-checkpoint "$run_root/checkpoint_prepared_zero.pt" --initial-sha256 "$(cat "$run_root/prepared.sha256")"
stage=short_summary
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_query_short.py --run "$run_root"
stage=native_fixed_weight_diagnosis
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/diagnose_serial_residual_short.py --run-root "$run_root"
stage=native_comparison_summary
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_serial_residual_short.py --run-root "$run_root"
