#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/gs_evidence_full.py compare
