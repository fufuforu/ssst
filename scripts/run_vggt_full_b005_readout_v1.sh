#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
OUTPUT=${TASK_OUTPUT:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
CHECKPOINT=/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/checkpoint_frozen_joint_epoch_04.pt
OFFICIAL=/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/evaluation/final_adapt2_joint4_official
COHORT=/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/probe_vggt_raw_query_readout_v1.py \
 --full-checkpoint "$CHECKPOINT" --official-root "$OFFICIAL" --cohort "$COHORT" \
 --output "$OUTPUT/preflight" --smoke
"$PYTHON" - "$OUTPUT/preflight" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for rank in range(4):
    proof=json.loads((root/f'rank{rank:02d}/smoke_report.json').read_text())
    assert proof['status']=='RAW_READOUT_SMOKE_PASSED' and len(proof['records'])==1
    assert proof['new_optimizer_updates']==0 and proof['model_parameters_unchanged']
    record=proof['records'][0]
    assert record['arm']=='full' and record['endpoint_pngs_reproduced'] and record['provider_rgb_reproduced']
    assert record['source']['new_joint_updates']==4172
    assert record['source']['adapter_understanding_scale']==.01 and record['source']['feedback_peak']==.05
print('FULL_B005_ALL_FOUR_RANK_PREFLIGHT_PASSED',flush=True)
PY
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/probe_vggt_raw_query_readout_v1.py \
 --full-checkpoint "$CHECKPOINT" --official-root "$OFFICIAL" --cohort "$COHORT" \
 --output "$OUTPUT/formal"
"$PYTHON" scripts/summarize_vggt_bottleneck_diagnostics_v1.py --mode raw --arms full \
 --root "$OUTPUT/formal" --output "$OUTPUT/formal/analysis"
