#!/bin/bash
set -euo pipefail
R="$1"
phase="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
while IFS= read -r task_rg_name; do unset "$task_rg_name"; done < <(compgen -A variable RG_)
export RG_SUPERVISED=6,12
export GSPLAT_PRECOMPILED_SO=/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so
trap 'code=$?; printf "{\"status\":\"FAILED\",\"phase\":\"%s\",\"exit_code\":%s}\n" "$phase" "$code" > "$R/${phase}_FAILED.json"' ERR
trap 'printf "{\"status\":\"FAILED\",\"exit_code\":143}\n" > "$R/${phase}_FAILED.json"; exit 143' TERM
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" && -z "$(git status --porcelain)" ]]
[[ ! -e "$R/${phase}_STARTED.json" ]]
sha256sum --check "$R/source_checkpoint.sha256"
printf '{"status":"STARTED"}\n' > "$R/${phase}_STARTED.json"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
SP=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
case "$phase" in
 cache|export)
  [[ "${SLURM_GPUS_ON_NODE:-}" == 8 ]]
  output="$R/cache"
  cohort="$R/cache_cohort.json"
  script=cache_gs_quality
  if [[ "$phase" == export ]]; then
   output="$R/eval"
   cohort=/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
   script=export_gs_quality
  fi
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "scripts/$script.py" --checkpoint "$R/source_checkpoint.pt" --output "$output" --cohort "$cohort"
  ;;
 fit) "$PY" scripts/fit_gs_quality.py --run-root "$R" ;;
 score)
  [[ "${SLURM_GPUS_ON_NODE:-}" == 5 ]]
  "$SP" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=5 scripts/score_gs8_quality.py --run-root "$R"
  ;;
 *) exit 2 ;;
esac
sha256sum --check "$R/source_checkpoint.sha256"
