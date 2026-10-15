#!/bin/bash
# Pass run root; submit with an afterok dependency on the corresponding training.
set -euo pipefail
run_root="$1"
cd /space/mawb/ssst_gfm_semantic_pool_v3_staged_code_20261015
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# The already-running direct4 job scores inline. Validate its artifacts and
# avoid repeating inference/scoring when those are already complete.
if test -f "$run_root/COMPLETE.json"; then
    /space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$run_root"
    exit 0
fi
stage=final_checkpoint_export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/EVALUATION_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/export_dual_stream_checkpoint.py --run-root "$run_root"
stage=official_score
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$run_root"
