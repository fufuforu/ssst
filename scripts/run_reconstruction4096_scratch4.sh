#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
run="$1";mode="$2"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
export GSPLAT_PRECOMPILED_SO=/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so
while IFS= read -r task_rg_name; do unset "$task_rg_name"; done < <(compgen -A variable RG_)
export RG_SUPERVISED=6,12
[[ "$(git rev-parse HEAD)" == "$(cat "$run/code.sha")" ]]
git diff --quiet HEAD -- tokengs scripts tests
trap 'rc=$?; /usr/bin/python3 -c '\''import json,sys;json.dump(dict(status="FAILED",mode=sys.argv[2],exit_code=int(sys.argv[3])),open(sys.argv[1],"w"))'\'' "$run/JOB_FAILED_${mode}.json" "$mode" "$rc"' ERR
args=()
case "$mode" in
 smoke) args+=(--smoke);;
 train) ;;
 *) exit 2;;
esac
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -- scripts/train_reconstruction4096_scratch4.py --run-root "$run" "${args[@]}"
