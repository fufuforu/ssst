#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?independent execution snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=direct_attribute_cuda_gate
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/smoke_direct_instance_attributes.py --run-root "$run_root"
stage=validate_smoke_receipt
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/direct_instance_short_policy.py --run-root "$run_root"
stage=fresh_seed42_256_update_short_training
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_unified_query_v1.py --architecture unified_direct_instance_attributes --output-dir "$run_root" --understanding-weight .05 --short-validation --checkpoint-every 32 --config "$run_root/model_config.json"
for readout_stage in initial final; do
    stage="fixed_${readout_stage}_prototype_and_hdbscan"
    if [[ "$readout_stage" == initial ]]; then checkpoint_path="$run_root/checkpoint_initial.pt"; else checkpoint_path="$run_root/checkpoint_latest.pt"; fi
    checkpoint_sha=$(sha256sum "$checkpoint_path")
    checkpoint_sha="${checkpoint_sha%% *}"
    /space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/evaluate_direct_instance_attributes.py --checkpoint "$checkpoint_path" --checkpoint-sha256 "$checkpoint_sha" --output "$run_root/readouts_$readout_stage" --readout both --evaluate
done
stage=training_and_paired_readout_audit
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_direct_instance_short.py --run-root "$run_root"
