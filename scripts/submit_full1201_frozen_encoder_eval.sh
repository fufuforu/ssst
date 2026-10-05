#!/usr/bin/env bash
set -euo pipefail
TASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK_REPORT=/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu
TASK_EVAL="$TASK_REPORT/evaluation"
TASK_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_SIU_PYTHON=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
TASK_NODE=3dimage-11
TASK_GPUS=5
TASK_BRANCH=object-locus-panoptic-full1201-frozen-encoder-eval

cd "$TASK_REPO"
TASK_SHA="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain)" ]] || { echo "evaluation worktree must be clean"; exit 2; }
TASK_REMOTE="$(git ls-remote origin "refs/heads/$TASK_BRANCH" | awk '{print $1}')"
[[ "$TASK_REMOTE" == "$TASK_SHA" ]] || { echo "local evaluation SHA is not the verified pushed branch SHA"; exit 2; }
mkdir -p "$TASK_EVAL/slurm"

TASK_SELECT="$(sbatch --parsable --job-name=freeze-dev8 --partition=3090 --nodelist="$TASK_NODE" \
  --nodes=1 --ntasks=1 --gres="gpu:3090:$TASK_GPUS" --cpus-per-task=32 --mem=128G --time=48:00:00 \
  --output="$TASK_EVAL/slurm/%x-%j.out" --error="$TASK_EVAL/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_GPUS="$TASK_GPUS",TASK_EVAL="$TASK_EVAL",OMP_NUM_THREADS=4 \
  --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module scripts.eval_full1201_frozen_encoder --profile dev8 --output "$TASK_EVAL/selection"')"
TASK_SELECT="$(cut -d';' -f1 <<<"$TASK_SELECT")"

TASK_DEV_AGG="$(sbatch --parsable --dependency="afterok:$TASK_SELECT" --job-name=freeze-dev8-aggregate \
  --partition=3090 --nodelist=3dimage-13 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=24:00:00 \
  --output="$TASK_EVAL/slurm/%x-%j.out" --error="$TASK_EVAL/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$TASK_REPO",TASK_SIU_PYTHON="$TASK_SIU_PYTHON",TASK_EVAL="$TASK_EVAL",OMP_NUM_THREADS=4 \
  --wrap='cd "$TASK_REPO" && "$TASK_SIU_PYTHON" scripts/aggregate_full1201_frozen_encoder_eval.py --mode dev8 --worker-root "$TASK_EVAL/selection"')"
TASK_DEV_AGG="$(cut -d';' -f1 <<<"$TASK_DEV_AGG")"

TASK_BEST="$(sbatch --parsable --dependency="afterok:$TASK_DEV_AGG" --job-name=freeze-best-export \
  --partition=3090 --nodelist="$TASK_NODE" --nodes=1 --ntasks=1 --gres="gpu:3090:$TASK_GPUS" --cpus-per-task=32 --mem=128G --time=48:00:00 \
  --output="$TASK_EVAL/slurm/%x-%j.out" --error="$TASK_EVAL/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_GPUS="$TASK_GPUS",TASK_EVAL="$TASK_EVAL",OMP_NUM_THREADS=4 \
  --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module scripts.eval_full1201_frozen_encoder --profile best --epoch 0 --reuse-dev8 --output "$TASK_EVAL/best"')"
TASK_BEST="$(cut -d';' -f1 <<<"$TASK_BEST")"

TASK_BASELINE="$(sbatch --parsable --dependency="afterok:$TASK_DEV_AGG" --job-name=unfrozen-depth-export \
  --partition=3090 --nodelist="$TASK_NODE" --nodes=1 --ntasks=1 --gres="gpu:3090:$TASK_GPUS" --cpus-per-task=32 --mem=128G --time=48:00:00 \
  --output="$TASK_EVAL/slurm/%x-%j.out" --error="$TASK_EVAL/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$TASK_REPO",TASK_PYTHON="$TASK_PYTHON",TASK_GPUS="$TASK_GPUS",TASK_EVAL="$TASK_EVAL",OMP_NUM_THREADS=4 \
  --wrap='cd "$TASK_REPO" && "$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_GPUS" --module scripts.eval_full1201_frozen_encoder --profile baseline_depth --output "$TASK_EVAL/unfrozen_depth"')"
TASK_BASELINE="$(cut -d';' -f1 <<<"$TASK_BASELINE")"

TASK_FINAL="$(sbatch --parsable --dependency="afterok:$TASK_BEST:$TASK_BASELINE" --job-name=freeze-final-report \
  --partition=3090 --nodelist=3dimage-13 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=48:00:00 \
  --output="$TASK_EVAL/slurm/%x-%j.out" --error="$TASK_EVAL/slurm/%x-%j.err" \
  --export=ALL,TASK_REPO="$TASK_REPO",TASK_SIU_PYTHON="$TASK_SIU_PYTHON",TASK_EVAL="$TASK_EVAL",OMP_NUM_THREADS=4 \
  --wrap='cd "$TASK_REPO" && "$TASK_SIU_PYTHON" scripts/aggregate_full1201_frozen_encoder_eval.py --mode best --epoch 0 --worker-root "$TASK_EVAL/best" --dev8-root "$TASK_EVAL/selection" && "$TASK_SIU_PYTHON" scripts/aggregate_full1201_frozen_encoder_eval.py --mode baseline_depth --worker-root "$TASK_EVAL/unfrozen_depth" && "$TASK_SIU_PYTHON" scripts/finalize_full1201_frozen_encoder_eval.py')"
TASK_FINAL="$(cut -d';' -f1 <<<"$TASK_FINAL")"

"$TASK_PYTHON" - "$TASK_SHA" "$TASK_NODE" "$TASK_GPUS" "$TASK_SELECT" "$TASK_DEV_AGG" "$TASK_BEST" "$TASK_BASELINE" "$TASK_FINAL" <<'PY'
import json,sys
from pathlib import Path
sha,node,gpus,select,devagg,best,baseline,final=sys.argv[1:]
p=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu/evaluation/submitted_jobs.json')
p.write_text(json.dumps(dict(eval_git_sha=sha,gpu_node=node,gpus_per_shard=int(gpus),
 jobs=dict(dev8_selection=select,dev8_aggregate_and_select=devagg,best_frozen_exports=best,
           unfrozen_epoch06_depth_only=baseline,final_aggregate_report_zip=final),
 chain='afterok; no polling or monitor job'),indent=2)+'\n')
print(p.read_text())
PY
