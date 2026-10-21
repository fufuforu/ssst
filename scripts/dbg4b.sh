#!/bin/bash
set -uo pipefail
W=/space/mawb/ssst/group_plus/semantic_resolution_v1/finetune_dbg4
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
cd /space/mawb/ssst_sem_res256_code_20261019
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf "$W"/C "$W"/R1 "$W"/U1
run_one () {
  name="$1"; shift
  out="$W/$name"; mkdir -p "$out"
  cp "$W/${name}_ckpt.pt" "$out/checkpoint_untrained_initial.pt"
  ( cd "$out" && sha256sum checkpoint_untrained_initial.pt | cut -d' ' -f1 > untrained_initial.sha256 )
  echo "===== $name"
  env "$@" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_unified_query_v1.py --architecture dual_superpoint --output-dir "$out" \
    --epochs 1 --max-updates 10 --understanding-weight .05 --understanding-warmup 1043 \
    --learning-rate 1e-5 --checkpoint-every 10 \
    --initial-checkpoint "$out/checkpoint_untrained_initial.pt" \
    --initial-sha256 "$(cat "$out/untrained_initial.sha256")" > "$out/run.log" 2>&1
  echo "exit=$?"
  grep -m3 -E "Error|error|Traceback|raise" "$out/run.log" | head -5
  tail -3 "$out/run.log"
}
run_one C RG_EXTRA=0
run_one R1 RG_EXTRA=1 RG_DEPTH=1
run_one U1 RG_EXTRA=1 RG_TSEM=1
echo DBG_DONE
