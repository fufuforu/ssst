#!/bin/bash
set -euo pipefail
R="$1"; mode="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="/space/mawb/.tb_deps${PYTHONPATH:+:$PYTHONPATH}"
while IFS= read -r task_rg_name; do unset "$task_rg_name"; done < <(compgen -A variable RG_)
export RG_SUPERVISED=6,12
PY=/space/mawb/anaconda3/envs/tokengs/bin/python
stage="$mode"
trap 'code=$?; if [[ ! -f "$R/JOB_FAILED.json" ]]; then printf "{\"status\":\"FAILED\",\"stage\":\"%s\",\"exit_code\":%s}\n" "$stage" "$code" > "$R/JOB_FAILED.json"; fi' ERR
trap 'printf "{\"status\":\"FAILED\",\"stage\":\"scheduler_signal\",\"exit_code\":143}\n" > "$R/JOB_FAILED.json"; exit 143' TERM
[[ "$(git rev-parse HEAD)" == "$(cat "$R/code.sha")" ]]
git diff --quiet HEAD -- tokengs scripts tests
if [[ "$mode" == smoke ]]; then
 "$PY" scripts/smoke_v4init_isolated.py --output-dir "$R"
elif [[ "$mode" == train ]]; then
 /usr/bin/python3 scripts/v4init_isolated_pipeline.py "$R" train_gate
 model_key=$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["model_key"])' "$R/configuration.json")
 /usr/bin/python3 scripts/run_with_io_bootstrap_retry.py --run-root "$R" -- \
 "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
 scripts/train_unified_query_v1.py --architecture "$model_key" --output-dir "$R" \
 --config "$R/model_config.json" --epochs 4 --learning-rate .0001 \
 --understanding-weight 1.0 --understanding-warmup 1043 --checkpoint-every 50 --epoch-probes \
 --initial-checkpoint "$R/checkpoint_untrained_initial.pt" --initial-sha256 "$(cat "$R/untrained_initial.sha256")" \
 --initialize-v4init-isolated-zero
elif [[ "$mode" == eval ]]; then
 /usr/bin/python3 scripts/v4init_isolated_pipeline.py "$R" eval_gate
 stage=export1860
 "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
 scripts/export_unified_query_checkpoint.py --checkpoint "$R/source_checkpoint.pt" --output "$R/evaluation" \
 --cohort /space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json
 stage=official_score
 /space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$R"
 stage=verify_compare
 /usr/bin/python3 scripts/v4init_isolated_pipeline.py "$R" finish
else
 exit 2
fi
