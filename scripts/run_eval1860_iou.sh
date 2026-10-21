#!/bin/bash
set -euo pipefail
out="$1"; ckpt="$2"; cohort="$3"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RG_IOUHEAD=1 RG_SUPERVISED=6,12
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"eval1860_iou\"}\n" "$code" > "$out/EVALUATION_FAILED.json"' ERR
echo "=== export 1860 with IoU-head ranking ==="
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/export_unified_query_checkpoint.py --checkpoint "$ckpt" --output "$out/evaluation" --cohort "$cohort"
echo "=== score (official evaluator venv) ==="
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$out" || \
  { echo "Official scoring failed"; exit 1; }
echo "=== eval done ==="
"$PY" - "$out" <<'VERIFY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);m=json.loads((r/'evaluation/metrics.json').read_text())
assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860
assert json.loads((r/'COMPLETE.json').read_text())['windows']==1860
assert len((r/'evaluation/gt_reference_manifest.jsonl').read_text().splitlines())==1860
VERIFY
