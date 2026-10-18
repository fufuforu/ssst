#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?independent snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=fixed_weight_gpu_export_and_mechanism
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/diagnose_fixed_instance_readout.py --run-root "$run_root"
stage=cpu_hdbscan_evaluation_and_summary
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/evaluate_fixed_instance_clustering.py --run-root "$run_root"
