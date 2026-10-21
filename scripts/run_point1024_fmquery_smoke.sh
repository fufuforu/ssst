#!/bin/bash
set -euo pipefail
R="$1"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset RG_IOUHEAD RG_EXTRA RG_ADAPT RG_ISO RG_DEPTH RG_TSEM
export RG_SUPERVISED=6,12
trap 'code=$?; if [[ ! -f "$R/JOB_FAILED.json" ]]; then printf "{\"status\":\"FAILED\",\"stage\":\"shell\",\"exit_code\":%s}\n" "$code" > "$R/JOB_FAILED.json"; fi' ERR
trap 'printf "{\"status\":\"FAILED\",\"stage\":\"scheduler_signal\",\"exit_code\":143}\n" > "$R/JOB_FAILED.json"; exit 143' TERM
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/smoke_point1024_fmquery_isoadapter.py --output-dir "$R"
