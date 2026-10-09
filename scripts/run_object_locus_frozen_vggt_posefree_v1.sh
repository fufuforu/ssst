#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
MANIFEST=/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json
SOURCE=/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt
RUN_DIR=/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1
ARTIFACT="$REPO/vggt_artifact_manifest.json"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {single-smoke|eight-smoke|train|resume} [smoke-output-dir]" >&2
  exit 2
fi
MODE=$1
REVISION=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["hf_revision"])' "$ARTIFACT")
export HF_HUB_CACHE="$ASSET_ROOT/hf_cache/hub"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET_ROOT/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

case "$MODE" in
  single-smoke)
    [[ $# -eq 2 ]] || { echo "single-smoke requires a fresh output directory" >&2; exit 2; }
    exec "$PYTHON" scripts/smoke_object_locus_frozen_vggt_posefree_v1.py \
      --single-card-real --manifest "$MANIFEST" --checkpoint "$SOURCE" \
      --output-dir "$2" --vggt-revision "$REVISION" --artifact-manifest "$ARTIFACT"
    ;;
  eight-smoke)
    [[ $# -eq 2 ]] || { echo "eight-smoke requires a fresh output directory" >&2; exit 2; }
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/smoke_object_locus_frozen_vggt_posefree_v1.py \
      --eight-card-real --manifest "$MANIFEST" --checkpoint "$SOURCE" \
      --output-dir "$2" --vggt-revision "$REVISION" --artifact-manifest "$ARTIFACT"
    ;;
  train)
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_object_locus_frozen_vggt_posefree_v1.py \
      --run-training --manifest "$MANIFEST" --checkpoint "$SOURCE" --run-dir "$RUN_DIR" \
      --vggt-revision "$REVISION" --artifact-manifest "$ARTIFACT"
    ;;
  resume)
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_object_locus_frozen_vggt_posefree_v1.py \
      --run-training --resume --manifest "$MANIFEST" --checkpoint "$SOURCE" --run-dir "$RUN_DIR" \
      --vggt-revision "$REVISION" --artifact-manifest "$ARTIFACT"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
