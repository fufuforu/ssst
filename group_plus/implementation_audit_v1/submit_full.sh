#!/bin/bash
#SBATCH --job-name=recipe-full
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/full-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/full-%j.err
#
# FULL 50000-step run.  Refuses to start unless gate.json says the candidate
# passed all five development gates.  NOT started by this round.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
python - <<'PY'
import json, sys
gate = json.load(open("group_plus/implementation_audit_v1/gate.json"))
if gate.get("blocked") or gate.get("candidate") in (None, "PENDING_GATE", "BLOCKED"):
    sys.exit("gate.json is BLOCKED or pending: refusing to start the full run")
print("gate ok:", gate["candidate"])
PY
python -u scripts/train_group_locusgs.py \
  --arm g0 --recipe \
  --recipe-head-mode legacy_prefix \
  --out-dir workspace_group_plus/implementation_audit_v1/recipe_full_50000/run \
  --plan group_plus/implementation_audit_v1/plan_full_50000.json \
  --split group_plus/implementation_audit_v1/full_split.json \
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \
  --instance-outer-weight 0.05 --assign-coef 0.2 --assign-every 1 \
  --lr 2e-05 --warmup 2000 \
  --steps 50000 --save-steps 0 25000 50000 \
  --eval-every 2500
