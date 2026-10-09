#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=object-locus-frozen-vggt-posefree-v1
RUN_DIR=/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor
REPORT=/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2_monitor
SLURM_DIR="$REPORT/slurm"
mkdir -p "$SLURM_DIR"
cd "$REPO"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || { echo "wrong branch" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "formal launch requires clean pushed worktree" >&2; exit 2; }
CODE_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')
[[ "$CODE_SHA" == "$REMOTE_SHA" ]] || { echo "code SHA is not the remote task branch head" >&2; exit 2; }
if squeue -h -u "$USER" -o "%j" | rg -q "^vggt-pf-(v2|monitor)-8x3090$"; then
  echo "a pose-free 8x3090 job is already queued or running; refusing duplicate submission" >&2
  exit 2
fi
TASK_RESUME=0
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | rg -q .; then
  if [[ -f "$RUN_DIR/checkpoint_latest.pt" ]]; then
    TASK_RESUME=1
  else
    FAILED_ROOT="$REPORT/failed_formal_roots/$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$(dirname "$FAILED_ROOT")"
    mv "$RUN_DIR" "$FAILED_ROOT"
    echo "preserved checkpoint-free failed root at $FAILED_ROOT; fresh restart"
  fi
fi
# Prepare the immutable code snapshot on the submit host, never on a compute node.
SNAPSHOT="$REPORT/snapshots/$CODE_SHA"
if [[ ! -d "$SNAPSHOT" ]]; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  git clone --quiet --shared --no-checkout "$REPO" "$SNAPSHOT"
  git -C "$SNAPSHOT" checkout --quiet --detach "$CODE_SHA"
fi
[[ "$(git -C "$SNAPSHOT" rev-parse HEAD)" == "$CODE_SHA" ]]
TASK_REPO="$SNAPSHOT" TASK_CODE_SHA="$CODE_SHA" TASK_RESUME="$TASK_RESUME" \
  sbatch --partition=3090 --nodelist=3dimage-13 --nodes=1 --ntasks=1 --gres=gpu:8 \
    --cpus-per-task=32 --mem=128G --time=48:00:00 \
    --job-name=vggt-pf-monitor-8x3090 \
    --output="$SLURM_DIR/%x-%j.out" --error="$SLURM_DIR/%x-%j.err" \
    --export=ALL,TASK_REPO="$SNAPSHOT",TASK_CODE_SHA="$CODE_SHA",TASK_RESUME="$TASK_RESUME",OMP_NUM_THREADS=4 \
    "$SNAPSHOT/scripts/run_object_locus_frozen_vggt_posefree_v1_job.sh"
