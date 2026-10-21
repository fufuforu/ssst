#!/bin/bash
set -euo pipefail
run="$1"; mode="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
unset RG_IOUHEAD RG_ADAPT RG_ISO RG_EXTRA RG_DEPTH RG_TSEM
export RG_MASK_RESOLUTION=128 RG_SUPERVISED=6,8,10,12
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
out="$run/$mode"; mkdir -p "$out"
trap 'status=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\",\"training\":false,\"parameter_updates\":0}\n" "$status" "$mode" > "$out/JOB_FAILED.json"' ERR
if [[ "$mode" == smoke ]]; then
  "$PY" scripts/diagnose_anchor_child_boundary.py --config "$run/configuration.json" --output "$out" --mode smoke
else
  test "$mode" = full
  test -f "$run/smoke/SMOKE_PASSED.json"
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/diagnose_anchor_child_boundary.py --config "$run/configuration.json" --output "$out" --mode full
fi
"$PY" scripts/summarize_anchor_child_boundary.py --config "$run/configuration.json" --output "$out" --mode "$mode"
