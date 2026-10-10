#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=object-locus-vggt-recon-adapt-freeze-v1
REPORT=/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1
MODE=${1:-train}
[[ "$MODE" == train || "$MODE" == single_smoke ]]
cd "$REPO"
[[ "$(git branch --show-current)" == "$BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
SHA=$(git rev-parse HEAD)
[[ "$SHA" == "$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')" ]]
NAME=vggt-adapt2-joint4
GPUS=8
CPUS=32
MEM=256G
DEPENDENCY=()
if [[ "$MODE" == single_smoke ]]; then
  NAME=vggt-adapt-single
  GPUS=1
  CPUS=8
  MEM=128G
else
  [[ -n "${TASK_SINGLE_REPORT:-}" ]]
  if [[ -n "${TASK_SINGLE_JOB:-}" ]]; then DEPENDENCY=(--dependency="afterok:$TASK_SINGLE_JOB"); fi
fi
if squeue -h -u "$USER" -o '%j' | rg -qx "$NAME"; then
  echo "existing $NAME job; refusing duplicate submission" >&2
  exit 2
fi
SNAPSHOT="$REPORT/snapshots/$SHA"
mkdir -p "$REPORT/slurm"
if [[ ! -d "$SNAPSHOT" ]]; then
  PREPARING="$SNAPSHOT.preparing.$$"
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --shared --no-checkout "$REPO" "$PREPARING"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$PREPARING" checkout --quiet --detach "$SHA"
  mv "$PREPARING" "$SNAPSHOT"
fi
[[ "$(git -C "$SNAPSHOT" rev-parse HEAD)" == "$SHA" ]]
[[ -z "$(git -C "$SNAPSHOT" status --porcelain --untracked-files=no)" ]]
# A nodelist requires every listed node; use the partition union minus node12
# so the scheduler chooses ONE of 11,13,14,17,18 for the unchanged eight ranks.
sbatch --parsable --partition=3090,4090 --exclude=3dimage-12 --nodes=1 --ntasks=1 \
  --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time=48:00:00 \
  --job-name="$NAME" --output="$REPORT/slurm/%x-%j.out" --error="$REPORT/slurm/%x-%j.err" \
  "${DEPENDENCY[@]}" --export="ALL,TASK_REPO=$SNAPSHOT,TASK_CODE_SHA=$SHA,TASK_MODE=$MODE" \
  "$SNAPSHOT/scripts/run_object_locus_vggt_staged_job.sh"
