#!/bin/bash
#SBATCH --job-name=gfm_sem_pool_full4
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt05_full_node11/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt05_full_node11
cd /space/mawb/ssst_gfm_semantic_pool_v3_full_code_20261015
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=resume306_full_train_export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_dual_stream_v1.py --output-dir "$run_root" --architecture gfm_v3 --resume --resume-source-sha f5a6134260c66a4012d7850e2e6acf5c08a478a0 --gate-policy learning_evidence_v2 --cache-root /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache
stage=official_score
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
