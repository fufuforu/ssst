#!/bin/bash -l
#SBATCH --job-name=obj-locus-text-refer-smoke
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/space/mawb/ssst/group_plus/object_locus_text_refer_v1/slurm/smoke-%j.out
#SBATCH --error=/space/mawb/ssst/group_plus/object_locus_text_refer_v1/slurm/smoke-%j.err
set -eo pipefail
source /space/mawb/anaconda3/etc/profile.d/conda.sh
conda activate tokengs
cd /space/mawb/ssst_object_locus_text_refer_v1
export PYTHONPATH=/space/mawb/ssst_object_locus_text_refer_v1
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
python scripts/smoke_object_locus_text_refer.py
