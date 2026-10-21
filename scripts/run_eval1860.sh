#!/bin/bash
set -euo pipefail
run_root="$1"
ckpt="$2"
cohort=/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"eval1860\"}\n" "$code" > "$run_root/EVALUATION_FAILED.json"' ERR
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
echo "=== export 1860 pairs on ${SLURM_NTASKS:-?} GPUs ==="
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC:-8}" \
  scripts/export_unified_query_checkpoint.py --checkpoint "$ckpt" --output "$run_root/evaluation" --cohort "$cohort"
echo "=== score ==="
"$PY" scripts/score_dual_stream_v1.py --run-root "$run_root"
echo "=== done ==="
