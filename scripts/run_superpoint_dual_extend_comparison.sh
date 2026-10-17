#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?immutable submit directory required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/compare_superpoint_dual_extend.py --run-root "$1"
