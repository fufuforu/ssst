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
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
[[ "$(git rev-parse HEAD)" == "$(cat "$run/code.sha")" ]]
git diff --quiet HEAD -- tokengs scripts tests
trap 'rc=$?; /usr/bin/python3 -c '\''import json,sys;json.dump(dict(status="FAILED",mode=sys.argv[2],exit_code=int(sys.argv[3])),open(sys.argv[1],"w"))'\'' "$run/JOB_FAILED_${mode}.json" "$mode" "$rc"' ERR
if [[ "$mode" == smoke || "$mode" == train ]]; then
 args=();if [[ "$mode" == smoke ]]; then args+=(--smoke); fi
 "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -- scripts/train_reconstruction_abcd.py --run-root "$run" "${args[@]}"
elif [[ "$mode" == export ]]; then
 "$PY" scripts/reconstruction_abcd_pipeline.py "$run" freeze
 "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -- scripts/export_unified_query_checkpoint.py --checkpoint "$run/eval/source_checkpoint.pt" --output "$run/eval/evaluation" --cohort /space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
 /usr/bin/python3 scripts/reconstruction_abcd_pipeline.py "$run" novel
elif [[ "$mode" == score ]]; then
 /space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run/eval"
 /space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/invoke_siu3r_official_evaluator.py --eval-path "$run/eval/novel" --output "$run/eval/novel_metrics.json" --recon-only --device cuda:0
 /usr/bin/python3 scripts/reconstruction_abcd_pipeline.py "$run" finish
else
 exit 2
fi
