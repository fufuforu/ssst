#!/bin/bash
set -euo pipefail
out="$1"; ckpt="$2"; cohort="$3"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf "$out"; mkdir -p "$out"
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/export_unified_query_checkpoint.py --checkpoint "$ckpt" --output "$out" --cohort "$cohort"
echo "SMOKE_EXPORT_OK"
ls "$out/all" | head -5
wc -l "$out/gt_reference_manifest.jsonl"
