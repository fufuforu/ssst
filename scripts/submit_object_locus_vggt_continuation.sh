#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=object-locus-vggt-joint-continuation-ab-v1
REPORT=/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_v1
ARM=${1:?choose frozen or joint_aa}
[[ "$ARM" == frozen || "$ARM" == joint_aa ]]
cd "$REPO"
[[ "$(git branch --show-current)" == "$BRANCH" && -z "$(git status --porcelain)" ]]
SHA=$(git rev-parse HEAD)
[[ "$SHA" == "$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')" ]]
NAME="vggt-cont2-$ARM"
if squeue -h -u "$USER" -o '%j' | rg -qx "$NAME"; then
  echo "existing $NAME job; refusing duplicate submission" >&2; exit 2
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
# 2*240G fits node13's 512000MB; 2*256G would prevent simultaneous allocation.
sbatch --parsable --partition=3090 --nodelist=3dimage-13 --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=16 --mem=240G --time=48:00:00 \
  --job-name="$NAME" --output="$REPORT/slurm/%x-%j.out" --error="$REPORT/slurm/%x-%j.err" \
  --export="ALL,TASK_REPO=$SNAPSHOT,TASK_CODE_SHA=$SHA,TASK_ARM=$ARM" \
  "$SNAPSHOT/scripts/run_object_locus_vggt_continuation_job.sh"
