#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=object-locus-vggt-joint-continuation-ab-v1
ARM=${1:?choose frozen or joint_aa}
NODE=${2:-3dimage-13}
WORLD=${3:-4}
GLOBAL_BATCH=${4:-8}
EXPERIMENT=object_locus_vggt_joint_continuation_ab_v1
if [[ "$GLOBAL_BATCH" == 16 ]]; then EXPERIMENT=object_locus_vggt_joint_continuation_ab_batch16_v1; fi
REPORT=/space/mawb/ssst/group_plus/$EXPERIMENT
[[ "$ARM" == frozen || "$ARM" == joint_aa ]]
[[ "$NODE" == 3dimage-11 || "$NODE" == 3dimage-13 || ( "$NODE" == 3dimage-18 && "$ARM" == frozen && "$WORLD" == 8 && "$GLOBAL_BATCH" == 16 ) ]]
[[ "$WORLD" == 4 || "$WORLD" == 8 ]]
[[ "$WORLD" == 8 || "$NODE" == 3dimage-13 ]]
[[ "$GLOBAL_BATCH" == 8 || ( "$GLOBAL_BATCH" == 16 && "$WORLD" == 8 ) ]]
cd "$REPO"
[[ "$(git branch --show-current)" == "$BRANCH" && -z "$(git status --porcelain)" ]]
SHA=$(git rev-parse HEAD)
[[ "$SHA" == "$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')" ]]
NAME="vggt-cont2-$ARM"
if [[ "$GLOBAL_BATCH" == 16 ]]; then NAME="vggt-cont2-b16-$ARM"; fi
for EXISTING in "vggt-cont2-$ARM" "vggt-cont2-b16-$ARM"; do
  if squeue -h -u "$USER" -o '%j' | rg -qx "$EXISTING"; then
    echo "existing $EXISTING job; refusing duplicate submission" >&2; exit 2
  fi
done
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
MEM=240G
if [[ "$WORLD" == 8 ]]; then MEM=480G; fi
PARTITION=3090
GPU_MODEL=RTX3090
if [[ "$NODE" == 3dimage-18 ]]; then PARTITION=4090; GPU_MODEL=RTX4090; fi
sbatch --parsable --partition="$PARTITION" --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gres="gpu:$WORLD" --cpus-per-task="$((WORLD*4))" --mem="$MEM" --time=48:00:00 \
  --job-name="$NAME" --output="$REPORT/slurm/%x-%j.out" --error="$REPORT/slurm/%x-%j.err" \
  --export="ALL,TASK_REPO=$SNAPSHOT,TASK_CODE_SHA=$SHA,TASK_ARM=$ARM,TASK_WORLD_SIZE=$WORLD,TASK_NODE=$NODE,TASK_GLOBAL_BATCH=$GLOBAL_BATCH,TASK_GPU_MODEL=$GPU_MODEL" \
  "$SNAPSHOT/scripts/run_object_locus_vggt_continuation_job.sh"
