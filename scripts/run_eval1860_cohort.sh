#!/bin/bash
set -euo pipefail
out="$1"; ckpt="$2"; cohort="$3"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"eval1860\"}\n" "$code" > "$out/EVALUATION_FAILED.json"' ERR
echo "=== export 1860 on 8 GPUs ==="
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/export_unified_query_checkpoint.py --checkpoint "$ckpt" --output "$out/evaluation" --cohort "$cohort"
echo "=== score ==="
"$PY" scripts/score_dual_stream_v1.py --run-root "$out"
echo "=== done ==="
wc -l "$out/evaluation/gt_reference_manifest.jsonl" 2>/dev/null || true
