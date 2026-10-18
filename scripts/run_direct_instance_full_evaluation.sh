#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?independent snapshot required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=prototype_official128_export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/EVALUATION_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/export_unified_query_checkpoint.py --checkpoint "$run_root/checkpoint_latest.pt" --output "$run_root/evaluation"
stage=official128_scoring
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$run_root"
stage=auxiliary_fixed8_prototype_and_hdbscan
checkpoint_sha=$(sha256sum "$run_root/checkpoint_latest.pt")
checkpoint_sha="${checkpoint_sha%% *}"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/evaluate_direct_instance_attributes.py --checkpoint "$run_root/checkpoint_latest.pt" --checkpoint-sha256 "$checkpoint_sha" --output "$run_root/readouts_final" --readout both --evaluate
stage=equal_budget_baseline_comparison
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/direct_instance_full.py compare
