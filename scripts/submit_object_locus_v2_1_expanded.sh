#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
NODE="3dimage-13"
PHASE="${1:-train}"
if [[ "$PHASE" != "train" && "$PHASE" != "smoke" ]]; then
  echo "usage: $0 [smoke|train]" >&2
  exit 2
fi
REPORTS="/space/mawb/ssst/group_plus/object_locus_v2_1_expanded/slurm"
mkdir -p "$REPORTS"
STATE="$(sinfo -N -h -n "$NODE" -o '%T %G' | head -n1 || true)"
if [[ -z "$STATE" || "${STATE,,}" == *down* || "${STATE,,}" == *drain* || "$STATE" != *3090* ]]; then
  echo "Registered node $NODE is unavailable or not 3090: $STATE" >&2
  exit 2
fi
if [[ "$PHASE" == "smoke" ]]; then
  SCRIPT="scripts/smoke_object_locus_v2_1_expanded.py"
  JOB="object-locus-v2-1-expanded-smoke"
else
  SCRIPT="scripts/train_object_locus_v2_1_expanded.py"
  JOB="object-locus-v2-1-expanded"
fi
sbatch --job-name="$JOB" --partition=3090 --nodelist="$NODE" --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=24:00:00 --export=ALL,REPO="$REPO",PYTHON="$PYTHON" \
  --output="$REPORTS/%x-%j.out" \
  --wrap="cd \"\$REPO\" && \"\$PYTHON\" -u \"$SCRIPT\""
