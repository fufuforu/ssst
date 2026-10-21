#!/bin/bash
#SBATCH --job-name=gfm_direct4_cachefix
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt07_direct4_cachefix_node11/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt07_direct4_cachefix_node11
cd /space/mawb/ssst_gfm_cache_publication_fix_code_20261015
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"resume_full250\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/train_dual_stream_v1.py --output-dir "$run_root" --architecture gfm_v3 --resume --resume-source-sha b403e7c5e56e0952b518df423c3d405206dc84cc --resume-reason 'Engineering cache publication fix; model/loss/data/schedule unchanged' --gate-policy learning_evidence_v2 --defer-export --cache-root /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache
