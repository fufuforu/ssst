#!/bin/bash
#SBATCH --job-name=unified_esam_many_short
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-18
#SBATCH --gres=gpu:4090:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu
cd /space/mawb/ssst_unified_esam_many_code_20261017
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"unified_many_short\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_unified_query_v1.py --output-dir "$run_root" --understanding-weight .05 --short-validation --checkpoint-every 32
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_unified_query_short.py --run "$run_root"
