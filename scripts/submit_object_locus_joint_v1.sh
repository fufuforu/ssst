#!/usr/bin/env bash
set -euo pipefail
TASK_PHASE="${1:-validation}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/space/mawb/anaconda3/envs/tokengs/bin/python"
REPORTS="/space/mawb/ssst/group_plus/object_locus_joint_v1/slurm"
mkdir -p "$REPORTS"
case "$TASK_PHASE" in
 validation) SCRIPT="scripts/smoke_object_locus_joint_v1.py"; ARGS="--phase all" ;;
 diagnose) SCRIPT="scripts/smoke_object_locus_joint_v1.py"; ARGS="--phase diagnose" ;;
 m1) SCRIPT="scripts/smoke_object_locus_joint_v1.py"; ARGS="--phase m1" ;;
 formal_pair) SCRIPT="scripts/train_object_locus_joint_v1.py"; ARGS="formal_pair" ;;
 control|joint) SCRIPT="scripts/train_object_locus_joint_v1.py"; ARGS="--arm $TASK_PHASE" ;;
 *) echo "Unknown phase: $TASK_PHASE" >&2; exit 2 ;;
esac
sbatch --job-name="object-joint-v1-$TASK_PHASE" --partition=3090 --nodelist=3dimage-11 \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --export=ALL,REPO="$REPO",PYTHON="$PYTHON",SCRIPT="$SCRIPT",ARGS="$ARGS" \
  --output="$REPORTS/%x-%j.out" \
  --wrap='set -e; cd "$REPO"; if [ "$ARGS" = "formal_pair" ]; then "$PYTHON" -u "$SCRIPT" --arm control; "$PYTHON" -u "$SCRIPT" --arm joint; else "$PYTHON" -u "$SCRIPT" $ARGS; fi' 
