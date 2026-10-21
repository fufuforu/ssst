#!/bin/bash
#SBATCH --job-name=gfm_two_adapter_v2
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-18
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt03_gfm_v2/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt03_gfm_v2
cd /space/mawb/ssst_gfm_dual_stream_v2_code_20261015
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
stage=transition_short_train_export
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"%s\"}\n" "$code" "$stage" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_dual_stream_v1.py --output-dir "$run_root" --architecture gfm_v2 --transition-from /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt02_node18/migration_source.pt --cache-root /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache
stage=official_score
/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/score_dual_stream_v1.py --run-root "$run_root"
