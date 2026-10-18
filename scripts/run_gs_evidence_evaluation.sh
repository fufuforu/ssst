#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
case "$1" in
 A) run_root=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt18_gs_evidence_full_node12_8gpu ;;
 B) run_root=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt19_gs_evidence_coverage_full_node13_8gpu ;;
 *) exit 2 ;;
esac
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage=official128_export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/EVALUATION_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/export_unified_query_checkpoint.py --checkpoint "$run_root/checkpoint_latest.pt" --output "$run_root/evaluation"
stage=official128_scoring
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/dual_stream_evaluation_checks.py --check-complete "$run_root"
