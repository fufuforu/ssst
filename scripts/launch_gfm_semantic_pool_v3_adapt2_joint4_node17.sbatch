#!/bin/bash
#SBATCH --job-name=gfm_sem_adapt2_joint4
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-17
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt06_adapt2_joint4_node17/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt06_adapt2_joint4_node17
source_checkpoint="$run_root/checkpoint_shared_short0306.pt"
cd /space/mawb/ssst_gfm_semantic_pool_v3_staged_code_20261015
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=one_window_export_smoke
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/export_dual_stream_checkpoint.py --run-root "$run_root" --checkpoint "$source_checkpoint" --smoke
stage=one_window_official_score_smoke
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_protocol_smoke.py --run-root "$run_root"
stage=adapt2_then_joint4
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_dual_stream_v1.py --output-dir "$run_root" --architecture gfm_v3 --adapt-epochs 2 --staged-from "$source_checkpoint" --staged-source-digest 56d4b7f8f0d680fc0367a97fc10ae7e4d3ebe2f441a01d777861f674352b093f --gate-policy learning_evidence_v2 --defer-export --cache-root /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache
