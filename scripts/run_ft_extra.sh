#!/bin/bash
set -euo pipefail
R=/space/mawb/ssst/group_plus/semantic_resolution_v1/finetune_extra_v1
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"finetune_extra\"}\n" "$code" > "$R/JOB_FAILED.json"' ERR
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
run_one () {
  name="$1"; shift
  out="$R/$name"
  if [ -f "$out/TRAINING_COMPLETE.json" ]; then echo "skip $name (complete)"; return 0; fi
  echo "===== $name"
  env "$@" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_unified_query_v1.py --architecture dual_superpoint --output-dir "$out" \
    --epochs 1 --understanding-weight .05 --understanding-warmup 1043 --learning-rate 1e-5 \
    --checkpoint-every 250 --epoch-probes \
    --initial-checkpoint "$out/checkpoint_untrained_initial.pt" \
    --initial-sha256 "$(cat "$out/untrained_initial.sha256")"
  [ -f "$out/TRAINING_COMPLETE.json" ] || { echo "missing complete $name" >&2; exit 3; }
}
run_one C RG_EXTRA=0
run_one R1 RG_EXTRA=1 RG_DEPTH=1
run_one U1 RG_EXTRA=1 RG_TSEM=1
