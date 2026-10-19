#!/bin/bash
set -euo pipefail
[[ $# == 3 ]] || exit 2
run_root="$1"
training_code="$2"
control_code="$3"
case "${SLURMD_NODENAME:-}" in
  3dimage-13|3dimage-14|3dimage-17|3dimage-18) ;;
  *) exit 2 ;;
esac
cd "$training_code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=source_preflight
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/usr/bin/python3 "$control_code/scripts/point1024_eval_flexible_dispatch.py" --runtime-training-check --run "$run_root" --snapshot "$training_code"
/space/mawb/anaconda3/envs/tokengs/bin/python "$training_code/scripts/submit_recon_then_object_full.py" --verify-only --run "$run_root"
stage=recon_then_object_full4_training
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  "$training_code/scripts/train_unified_query_v1.py" --architecture recon_then_object --output-dir "$run_root" --epochs 4 \
  --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 \
  --checkpoint-every 50 --epoch-probes --config "$run_root/model_config.json" \
  --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" \
  --initial-sha256 "$(cat "$run_root/untrained_initial.sha256")"
