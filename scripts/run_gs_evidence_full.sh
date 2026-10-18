#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?immutable snapshot required}"
group="$1"
case "$group" in
 A) run_root=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt18_gs_evidence_full_node12_8gpu; architecture=unified_gs_evidence ;;
 B) run_root=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt19_gs_evidence_coverage_full_node13_8gpu; architecture=unified_gs_evidence_coverage ;;
 *) exit 2 ;;
esac
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=cuda_evidence_and_two_discarded_ddp_updates
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/smoke_gs_evidence_full.py --group "$group"
stage=fresh_full4_training
initial_sha=$(sha256sum "$run_root/checkpoint_untrained_initial.pt")
initial_sha="${initial_sha%% *}"
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_unified_query_v1.py --architecture "$architecture" --output-dir "$run_root" --epochs 4 --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 --checkpoint-every 50 --epoch-probes --config "$run_root/model_config.json" --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" --initial-sha256 "$initial_sha"
stage=full_training_integrity_audit
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/gs_evidence_full.py audit --group "$group"
