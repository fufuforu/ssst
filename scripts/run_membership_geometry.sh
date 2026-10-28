#!/bin/bash
set -euo pipefail
R="$1"
mode="$2"
arm="${3:-}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
while IFS= read -r task_rg_name; do unset "$task_rg_name"; done < <(compgen -A variable RG_)
export RG_SUPERVISED=6,12
trap 'code=$?; printf "{\"status\":\"FAILED\",\"stage\":\"%s\",\"arm\":\"%s\",\"exit_code\":%s}\n" "$mode" "$arm" "$code" > "$R/${mode}${arm}_FAILED.json"' ERR
trap 'printf "{\"status\":\"FAILED\",\"stage\":\"scheduler_signal\",\"exit_code\":143}\n" > "$R/${mode}${arm}_FAILED.json"; exit 143' TERM
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" ]]
[[ -z "$(git status --porcelain)" ]]
sha256sum --check "$R/source_checkpoint.sha256"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
if [[ "$mode" == score ]]; then
 [[ -f "$R/G1/evaluation/PAIRING_AUDIT.json" && -f "$R/G2/evaluation/PAIRING_AUDIT.json" && ! -e "$R/COMPLETE.json" ]]
 /space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_membership_geometry.py --run-root "$R"
else
 [[ "${SLURM_GPUS_ON_NODE:-}" == 4 ]]
 if [[ "$mode" == cache ]]; then
  [[ ! -e "$R/teacher/STARTED.json" ]]
  mkdir -p "$R/teacher"
  printf '{"stage":"cache"}\n' > "$R/teacher/STARTED.json"
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/cache_membership_geometry.py \
   --checkpoint "$R/source_checkpoint.pt" --output "$R/teacher" --cohort "$R/training_cohort.json"
 elif [[ "$mode" == train || "$mode" == export ]]; then
  [[ "$arm" == G1 || "$arm" == G2 ]]
  [[ -f "$R/teacher/export_complete.json" && ! -e "$R/${mode}${arm}_STARTED.json" ]]
  printf '{"stage":"%s","arm":"%s"}\n' "$mode" "$arm" > "$R/${mode}${arm}_STARTED.json"
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 "scripts/${mode}_membership_geometry.py" --run-root "$R" --arm "$arm"
 else
  exit 2
 fi
fi
sha256sum --check "$R/source_checkpoint.sha256"
