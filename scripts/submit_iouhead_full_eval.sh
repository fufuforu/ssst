#!/bin/bash
set -euo pipefail
cd /space/mawb/ssst_iouhead_code_20261020_v1
B=/space/mawb/ssst/group_plus/semantic_resolution_v1/iouhead_20261020_v1
COH=/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
SMOKE_JOB="$1"
/space/mawb/anaconda3/envs/tokengs/bin/python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASSED"' "$B/smoke/SMOKE_PASSED.json"
[ ! -e "$B/SUBMITTED.json" ] || { echo 'Refusing duplicate submission'; exit 1; }
FULL=$(sbatch --parsable --job-name iou16_v1 --partition=3090 --gres=gpu:3090:8 --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --mem=192G --time=30:00:00 --dependency="afterok:$SMOKE_JOB" --kill-on-invalid-dep=yes \
  --output="$B/train16/slurm-%j.out" scripts/run_iouhead.sh "$B/train16" 16 full "$B/smoke")
printf '%s\n' "$FULL" > "$B/FULL_JOB_ID"
EVAL=$(sbatch --parsable --job-name eiou1860_v1 --partition=3090,4090 --gres=gpu:8 --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --mem=192G --time=08:00:00 --dependency="afterok:$FULL" --kill-on-invalid-dep=yes \
  --output="$B/train16/eval1860/slurm-%j.out" scripts/run_eval1860_iou.sh "$B/train16/eval1860" "$B/train16/checkpoint_latest.pt" "$COH")
printf '%s\n' "$EVAL" > "$B/EVAL_JOB_ID"
/space/mawb/anaconda3/envs/tokengs/bin/python - "$B" "$SMOKE_JOB" "$FULL" "$EVAL" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);record=dict(smoke=sys.argv[2],training=sys.argv[3],evaluation=sys.argv[4],dependencies=['smoke PASSED + afterok:smoke -> training','afterok:training -> eval1860'],training_gpus=8,training_partition='3090',eval_gpus=8,eval_partition='3090,4090')
(r/'SUBMITTED.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
PY
