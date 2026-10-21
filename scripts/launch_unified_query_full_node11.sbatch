#!/bin/bash
#SBATCH --job-name=unified_esam_many_full4
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu
cd /space/mawb/ssst_unified_esam_full_code_20261017
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"unified_many_full4\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_unified_query_v1.py --output-dir "$run_root" --epochs 4 --understanding-weight .05 --understanding-warmup 1043 --learning-rate .0001 --checkpoint-every 50 --epoch-probes --initial-checkpoint "$run_root/checkpoint_untrained_initial.pt" --initial-sha256 "$(cat "$run_root/untrained_initial.sha256")"
