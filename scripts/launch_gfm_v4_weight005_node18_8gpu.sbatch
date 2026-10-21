#!/bin/bash
#SBATCH --job-name=gfm_v4_weight005_joint4
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-18
#SBATCH --gres=gpu:4090:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt13_geometric_sam_weight005_node18_8gpu/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt13_geometric_sam_weight005_node18_8gpu
cd /space/mawb/ssst_gfm_v4_weight005_code_20261016
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"v4_weight005_joint4\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_dual_stream_v1.py --output-dir "$run_root" --architecture gfm_v4 --v4-direct-joint --understanding-loss-scale 0.05 --untrained-initial-from "$run_root/checkpoint_untrained_initial.pt" --untrained-initial-digest "$(cat "$run_root/untrained_initial.sha256")" --gate-policy learning_evidence_v2 --defer-export --cache-root /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache
