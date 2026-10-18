#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=cuda_smoke
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/smoke_unified_instance_grounding.py --run-root "$run_root"
stage=validate_smoke_gate
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/validate_unified_instance_smoke.py --run-root "$run_root"
stage=fresh_short_training
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_unified_query_v1.py --architecture "${2:-unified_instance_grounding}" --output-dir "$run_root" --understanding-weight .05 --short-validation --checkpoint-every 32 --config "$run_root/model_config.json"
stage=short_summary
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_query_short.py --run "$run_root"
stage=instance_feature_audit
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_instance_short.py --run-root "$run_root"
if [[ "${2:-unified_instance_grounding}" == "unified_gaussian_consistency" ]]; then
    stage=direct_gaussian_audit
    /space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_gaussian_consistency_short.py --run-root "$run_root"
elif [[ "${2:-unified_instance_grounding}" != "unified_instance_grounding" ]]; then
    stage=coupling_audit
    /space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_coupling_short.py --run-root "$run_root"
fi
