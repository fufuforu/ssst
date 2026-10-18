#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_coupling_short.py --group "$1"
