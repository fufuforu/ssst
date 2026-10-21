#!/bin/bash
set -euo pipefail
run_root="$1"; epochs="$2"; mode="${3:-full}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RG_SUPERVISED=6,12 RG_IOUHEAD=1 RG_IOU_W=0.5
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
NGPU=$("$PY" -c "import torch;print(torch.cuda.device_count())")
[ "$NGPU" -eq 8 ] || { echo 'Original zero checkpoint requires 8 ranks'; exit 1; }
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"iouhead\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
extra=(--epoch-probes --no-epoch-checkpoints)
if [ "$mode" = smoke ]; then
  extra=(--max-updates 2 --no-epoch-checkpoints)
else
  "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASSED"' "$4/SMOKE_PASSED.json"
fi
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" \
  scripts/train_unified_query_v1.py --architecture dual_superpoint --output-dir "$run_root" \
  --epochs "$epochs" --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 \
  --checkpoint-every 50 "${extra[@]}" \
  --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" \
  --initial-sha256 "$(cat "$run_root/untrained_initial.sha256")"
if [ "$mode" = smoke ]; then
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    scripts/export_unified_query_checkpoint.py --checkpoint "$run_root/checkpoint_latest.pt" \
    --output "$run_root/export_smoke" --smoke
  "$PY" scripts/verify_iouhead_smoke.py "$run_root"
fi
