#!/bin/bash
set -euo pipefail
W=/space/mawb/ssst/group_plus/semantic_resolution_v1/finetune_dbg4
mkdir -p "$W"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
cd /space/mawb/ssst_sem_res256_code_20261019
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$PY - "$W" <<'EOF'
import hashlib, sys, torch
from pathlib import Path
w = Path(sys.argv[1])
src = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_latest.pt')
ck = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
for name in ('C', 'R1', 'U1'):
    d = w / name
    d.mkdir(parents=True, exist_ok=True)
    blob = dict(model=ck['model'], optimizer=dict(state={}, param_groups=[]),
                progress=dict(epoch=0, update_in_epoch=0, full_updates=0),
                rank_rng=ck['rank_rng'][:4], meta=ck['meta'])
    p = d / 'checkpoint_untrained_initial.pt'
    torch.save(blob, p)
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    (d / 'untrained_initial.sha256').write_text(h.hexdigest() + '\n')
    print(name, 'ok', h.hexdigest()[:12])
EOF
run_one () {
  name="$1"; shift
  out="$W/$name"
  echo "===== $name"
  env "$@" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_unified_query_v1.py --architecture dual_superpoint --output-dir "$out" \
    --epochs 1 --max-updates 10 --understanding-weight .05 --understanding-warmup 1043 \
    --learning-rate 1e-5 --checkpoint-every 10 \
    --initial-checkpoint "$out/checkpoint_untrained_initial.pt" \
    --initial-sha256 "$(cat "$out/untrained_initial.sha256")" 2>&1 | tail -6
  echo "---- $name exit=$?"
  ls "$out" | head -6
}
run_one C RG_EXTRA=0
run_one R1 RG_EXTRA=1 RG_DEPTH=1
run_one U1 RG_EXTRA=1 RG_TSEM=1
echo DBG_DONE
