#!/bin/bash
#SBATCH --job-name=gfm_geo_sam_frozen_u4
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-18
#SBATCH --gres=gpu:4090:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt10_geometric_sam_frozen_node18_8gpu/slurm-%j.out
set -euo pipefail
run_root=/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt10_geometric_sam_frozen_node18_8gpu
cd /space/mawb/ssst_gfm_geometric_frozen_code_20261016
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"geometric_frozen_understanding4\"}\n" "$code" > "$run_root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train_dual_stream_v1.py --output-dir "$run_root" --architecture gfm_v4 --adapt-epochs 2 --joint-warmup --freeze-reconstruction --geometric-from "$run_root/checkpoint_source_adaptation2.pt" --geometric-source-digest 2b07922b1b2c3e8f5dfa6c1591951e043bc418874f2cea379588003ddde6f3e3 --gate-policy learning_evidence_v2 --defer-export --cache-root /space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache
