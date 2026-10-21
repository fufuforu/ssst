#!/bin/bash
set -euo pipefail
out="$1"; ckpt="$2"; resolution="$3"; layers="$4"; cohort="$5"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset RG_IOUHEAD RG_ADAPT RG_ISO RG_EXTRA RG_DEPTH RG_TSEM
export RG_MASK_RESOLUTION="$resolution" RG_SUPERVISED="$layers"
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"A8_eval1860\"}\n" "$code" > "$out/EVALUATION_FAILED.json"' ERR
NGPU=$("$PY" -c 'import torch;print(torch.cuda.device_count())')
[ "$NGPU" -eq 4 ] || { echo 'Expected four allocated GPUs'; exit 1; }
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" \
  scripts/export_unified_query_checkpoint.py --checkpoint "$ckpt" --checkpoint-epoch 8 \
  --output "$out/evaluation" --cohort "$cohort"
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$out"
"$PY" - "$out" <<'VERIFY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);m=json.loads((r/'evaluation/metrics.json').read_text());c=json.loads((r/'configuration.json').read_text())
assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860
assert json.loads((r/'COMPLETE.json').read_text())['windows']==1860
identity=json.loads((r/'evaluation/checkpoint_identity.json').read_text())
assert identity['evaluated_epoch']==8 and identity['mask_resolution']==c['mask_resolution']
assert identity['checkpoint_sha256']==c['checkpoint_sha256'] and not identity['smoke_only']
assert len((r/'evaluation/gt_reference_manifest.jsonl').read_text().splitlines())==1860
VERIFY
