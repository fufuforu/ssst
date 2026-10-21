#!/bin/bash
set -euo pipefail
run_root="$1"
cd "${SLURM_SUBMIT_DIR:?snapshot submit directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=unified_export128
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/EVALUATION_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/export_unified_query_checkpoint.py --checkpoint "$run_root/checkpoint_latest.pt" --output "$run_root/evaluation"
stage=official_score11
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$run_root"
stage=compare_completed_controls
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/compare_gfm_native_runs.py --run "unified_query_full4=$run_root" --run v4_isolated=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt12_geometric_sam_isolated_node13_8gpu --run v4_weight005=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt13_geometric_sam_weight005_node18_8gpu --output "$run_root/comparison_to_completed_controls"
