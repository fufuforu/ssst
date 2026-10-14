#!/usr/bin/env bash
set -euo pipefail
REPO=${TASK_REPO:?}
OUTPUT=${TASK_OUTPUT:?}
ASSET=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets
export HF_HUB_CACHE="$ASSET/hf_cache/hub" HF_HUB_OFFLINE=1
export PYTHONPATH="$ASSET/vggt_source_checkout:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GSPLAT_PRECOMPILED_SO=/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so
export POSEFREE_V2_EVIDENCE_DIR="$OUTPUT/geometry_evidence"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$TASK_CODE_SHA" ]]
PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
OFFICIAL=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
case "${TASK_MODE:?}" in
 preflight)
  for STAGE in single eight; do
   WORKERS=1
   if [[ "$STAGE" == eight ]]; then WORKERS=8; fi
   for ARM in baseline object_local_read anchor_visual reliable_depth; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
     scripts/train_vggt_reading_controls_v1.py --mode "${STAGE}_smoke" --arm "$ARM" --output "$OUTPUT/$STAGE/$ARM"
    # One full-strength real-window official13 witness per arm/stage.
    "$OFFICIAL" scripts/invoke_siu3r_official_evaluator.py --eval-path "$OUTPUT/$STAGE/$ARM/evaluation_rank00/all" \
     --output "$OUTPUT/$STAGE/$ARM/official_witness.json"
   done
  done
  /usr/bin/python3 - "$OUTPUT" "$TASK_CODE_SHA" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]);arms=('baseline','object_local_read','anchor_visual','reliable_depth')
capacity=None
for stage,world in [('single',1),('eight',8)]:
 for arm in arms:
  p=root/stage/arm
  complete=json.load(open(p/'COMPLETE.json'));proof=json.load(open(p/'smoke_update_proof.json'))
  assert complete['status']=='READING_CONTROL_SMOKE_PASSED' and complete['formal_updates']==0
  assert proof['status']=='GPU_STEP_VERIFIED' and len(proof['rank_rows'])==world
  zero=json.load(open(p/'zero_reproduction.json'));assert zero['status']=='NATIVE_C_ZERO_START_VERIFIED'
  assert all(all(r['max_abs_differences'][k]<=r['tolerances'][k] and r['native_repeat_max_abs_differences'][k]<=r['tolerances'][k] for k in r['tolerances']) for r in zero['rows'])
  initial=json.load(open(p/'run_manifest.json'))['rank_initialization'][0]
  if capacity is not None: assert initial==capacity
  capacity=initial
  witness=json.load(open(p/'official_witness.json'));assert witness['official_evaluator_used']
(root/'PREFLIGHT_VERIFIED.json').write_text(json.dumps(dict(status='READING_PREFLIGHT_VERIFIED',execution_sha=sys.argv[2],formal_updates=0,
 actual1_and8_rank_updates_verified=True,zero_start_native_C_verified=True,exact_upstream_query_and_Gaussians=True,equal_capacity=True),indent=2)+'\n')
PY
  ;;
 startup|train_eval)
  /usr/bin/python3 - "${TASK_PREFLIGHT:?}/PREFLIGHT_VERIFIED.json" "$TASK_CODE_SHA" <<'PY'
import json,sys,os
p=json.load(open(sys.argv[1]));assert p['status']=='READING_PREFLIGHT_VERIFIED'
if p['execution_sha']!=sys.argv[2]:
 from scripts.vggt_reading_numerical_contract_v1 import contract
 reuse=json.load(open(os.environ['TASK_PREFLIGHT_REUSE_PROOF']))
 assert reuse['status'] in ('SERIALIZER_ONLY_RECOVERY_WITH_IDENTICAL_TRAINING_MATH','ENGINEERING_ONLY_RECOVERY_WITH_IDENTICAL_TRAINING_MATH')
 assert reuse['preflight_execution_sha']==p['execution_sha'] and reuse['execution_sha']==sys.argv[2]
 assert contract(os.environ['TASK_REPO'])==reuse['numerical_contract']
 assert reuse['lossless_parameter_optimizer_RNG_counts_roundtrip_verified']
else:assert p['execution_sha']==sys.argv[2]
PY
  if [[ "$TASK_MODE" == startup ]]; then
   exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_vggt_reading_controls_v1.py \
    --mode train --stop-after 20 --arm "${TASK_ARM:?}" --output "$OUTPUT/training"
  fi
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_vggt_reading_controls_v1.py \
   --mode train --resume --arm "${TASK_ARM:?}" --output "$OUTPUT/training"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/eval_vggt_reading_controls_v1.py \
   --mode export --arm "$TASK_ARM" --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/eval_vggt_reading_controls_v1.py \
   --mode score --arm "$TASK_ARM" --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  "$OFFICIAL" scripts/eval_vggt_reading_controls_v1.py --mode reduce --arm "$TASK_ARM" \
   --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  ;;
 startup_gate)
  /usr/bin/python3 - "$OUTPUT" "$TASK_CODE_SHA" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]);rows={};initial=None
for arm in ('baseline','object_local_read','anchor_visual','reliable_depth'):
 p=root/arm/'training';proof=json.load(open(p/'startup_confirmation.json'))
 assert proof['status']=='TRAINING_CONFIRMED' and proof['formal_updates']==20 and proof['formal_exposures']==160
 assert proof['execution_sha']==sys.argv[2] and len(proof['rank_rows'])==8
 assert all(r['finite'] and r['window_counts']==20 for r in proof['rank_rows'])
 init=json.load(open(p/'run_manifest.json'))['rank_initialization'][0]
 if initial is not None:assert init==initial
 initial=init
 ck=json.load(open(p/'checkpoint_latest.json'));assert ck['completed_updates']==20
 rows[arm]=proof
(root/'ALL_STARTUPS_VERIFIED.json').write_text(json.dumps(dict(status='ALL_FOUR_TRAINING_STARTUPS_VERIFIED',execution_sha=sys.argv[2],
 formal_updates_per_arm=20,formal_exposures_per_arm=160,rows=rows),indent=2)+'\n')
PY
  ;;
 score_existing)
  "$OFFICIAL" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/eval_vggt_reading_controls_v1.py \
   --mode score --arm "${TASK_ARM:?}" --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  exec "$OFFICIAL" scripts/eval_vggt_reading_controls_v1.py --mode reduce --arm "$TASK_ARM" \
   --root "$OUTPUT/evaluation" --run-root "$OUTPUT/training"
  ;;
 summary)
  exec "$OFFICIAL" scripts/eval_vggt_reading_controls_v1.py --mode summary --root "$OUTPUT"
  ;;
 *) exit 2 ;;
esac
