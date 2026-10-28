#!/bin/bash
set -euo pipefail
R="$1"
mode="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
while IFS= read -r task_rg_name; do unset "$task_rg_name"; done < <(compgen -A variable RG_)
export RG_SUPERVISED=6,12
trap 'code=$?; printf "{\"status\":\"FAILED\",\"stage\":\"%s\",\"exit_code\":%s}\n" "$mode" "$code" > "$R/${mode}_FAILED.json"' ERR
trap 'printf "{\"status\":\"FAILED\",\"stage\":\"scheduler_signal\",\"exit_code\":143}\n" > "$R/${mode}_FAILED.json"; exit 143' TERM
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" ]]
git diff --quiet HEAD -- tokengs scripts tests
sha256sum --check "$R/source_checkpoint.sha256"
if [[ "$mode" == export ]]; then
 [[ ! -e "$R/export_complete.json" && ! -d "$R/E0" ]]
 # PROCESS_COUNT_BEGIN
 nproc="${SLURM_GPUS_ON_NODE:-}"
 [[ "$nproc" =~ ^[1-9][0-9]*$ ]] && (( nproc <= 10 )) || { echo "Invalid Slurm GPU count: $nproc" >&2; exit 2; }
 # PROCESS_COUNT_END
 /space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$nproc" \
  scripts/export_fixed_query_diagnostics.py --checkpoint "$R/source_checkpoint.pt" --output "$R" \
  --cohort /space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
elif [[ "$mode" == score ]]; then
 [[ -f "$R/export_complete.json" && ! -e "$R/COMPLETE.json" ]]
 /space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_fixed_query_diagnostics.py --run-root "$R"
else
 exit 2
fi
sha256sum --check "$R/source_checkpoint.sha256"
