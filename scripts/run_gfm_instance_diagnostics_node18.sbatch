#!/bin/bash
#SBATCH --job-name=gfm_instance_diagnostic
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-18
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt01/slurm-%j.out
set -euo pipefail
cd /space/mawb/ssst_gfm_v3_isolated_diagnostics_code_20261016
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt01
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$root/JOB_FAILED.json"' ERR
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/diagnose_gfm_instances.py --output "$root/smoke" --smoke
/space/mawb/anaconda3/envs/tokengs/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/diagnose_gfm_instances.py --output "$root/full"
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/summarize_gfm_instance_diagnostics.py --output "$root/full"
