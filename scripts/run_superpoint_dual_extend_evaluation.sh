#!/bin/bash
set -euo pipefail
run_root="$1"
epoch="$2"
eval_root="$run_root/official_epoch$epoch"
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$eval_root/EVALUATION_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python - "$run_root" "$eval_root" "$epoch" <<'PY'
import json,sys
from pathlib import Path
root,out,epoch=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
training=json.loads((root/'TRAINING_COMPLETE.json').read_text())
assert training['progress']==dict(epoch=8,update_in_epoch=0,full_updates=8344)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'COMPLETE.json').exists()
(out/'TRAINING_COMPLETE.json').write_text(json.dumps(training,indent=2)+'\n')
configuration=json.loads((root/'configuration.json').read_text());configuration['evaluation_epoch']=epoch
(out/'configuration.json').write_text(json.dumps(configuration,indent=2)+'\n')
PY
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/export_unified_query_checkpoint.py --checkpoint "$run_root/checkpoint_epoch$epoch.pt" --output "$eval_root/evaluation" --evaluation-epoch "$epoch"
stage=official_score11
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$eval_root"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$eval_root"
