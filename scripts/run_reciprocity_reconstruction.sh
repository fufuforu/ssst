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
export GSPLAT_PRECOMPILED_SO=/space/mawb/tokengs_siu3r_joint_v1/.torch_extensions_v4/gsplat_cuda/gsplat_cuda.so
trap 'code=$?; printf "{\"status\":\"FAILED\",\"stage\":\"%s\",\"arm\":\"%s\",\"exit_code\":%s}\n" "$mode" "$arm" "$code" > "$R/${mode}${arm}_FAILED.json"' ERR
trap 'printf "{\"status\":\"FAILED\",\"stage\":\"scheduler_signal\",\"exit_code\":143}\n" > "$R/${mode}${arm}_FAILED.json"; exit 143' TERM
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" && -z "$(git status --porcelain)" ]]
sha256sum --check "$R/source_checkpoint.sha256"
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
SCOREPY=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
[[ ! -e "$R/${mode}${arm}_STARTED.json" ]]
printf '{"stage":"%s","arm":"%s"}\n' "$mode" "$arm" > "$R/${mode}${arm}_STARTED.json"
case "$mode" in
 membership_gate)
  [[ "${SLURM_GPUS_ON_NODE:-}" == 4 ]]
  for gate_arm in G1 G2; do
   "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_membership_geometry.py --run-root "$R" --arm "$gate_arm" --gate
  done
  ;;
 membership_train|membership_export)
  [[ "${SLURM_GPUS_ON_NODE:-}" == 4 ]]
  task_script=train_membership_geometry
  [[ "$mode" != membership_export ]] || task_script=export_membership_geometry
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 "scripts/$task_script.py" --run-root "$R" --arm "$arm"
  ;;
 membership_score) "$SCOREPY" scripts/score_membership_geometry.py --run-root "$R" ;;
 gs_cache|gs_export)
  [[ "${SLURM_GPUS_ON_NODE:-}" == 4 ]]
  task_script=cache_gs_quality
  task_output="$R/cache"
  task_cohort="$R/cache_cohort.json"
  if [[ "$mode" == gs_export ]]; then
   task_script=export_gs_quality
   task_output="$R/eval"
   task_cohort=/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
  fi
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 "scripts/$task_script.py" --checkpoint "$R/source_checkpoint.pt" --output "$task_output" --cohort "$task_cohort"
  ;;
 gs_fit) "$PY" scripts/fit_gs_quality.py --run-root "$R" ;;
 gs_score) "$SCOREPY" scripts/score_gs_quality.py --run-root "$R" ;;
 recon_export)
  [[ "${SLURM_GPUS_ON_NODE:-}" == 4 ]]
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/diagnose_reconstruction_gap.py --run-root "$R"
  ;;
 recon_score) "$SCOREPY" scripts/score_reconstruction_gap.py --run-root "$R" ;;
 *) exit 2 ;;
esac
sha256sum --check "$R/source_checkpoint.sha256"
