#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?independent snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=direct_attribute_full4_training
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
initial_sha=$(sha256sum "$run_root/checkpoint_untrained_initial.pt")
initial_sha="${initial_sha%% *}"
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_unified_query_v1.py --architecture unified_direct_instance_attributes --output-dir "$run_root" --epochs 4 --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 --checkpoint-every 50 --epoch-probes --config "$run_root/model_config.json" --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" --initial-sha256 "$initial_sha"
stage=full4_training_integrity_audit
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/direct_instance_full.py audit
