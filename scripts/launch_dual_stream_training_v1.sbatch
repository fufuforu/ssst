#!/bin/bash
#SBATCH --job-name=omega_sam_learn_full4
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01
code_root=/space/mawb/ssst_dual_stream_training_v1_code_20261015
cd "$code_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=train_and_export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/train_dual_stream_v1.py --output-dir "$run_root"
stage=official_score
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
