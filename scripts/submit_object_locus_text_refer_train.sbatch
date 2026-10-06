#!/bin/bash -l
#SBATCH --job-name=object-locus-text-refer
#SBATCH --partition=3090
#SBATCH --nodelist=3dimage-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/object_locus_text_refer_v1_train/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/object_locus_text_refer_v1_train/slurm-%j.err
set -euo pipefail
source /space/mawb/anaconda3/etc/profile.d/conda.sh
conda activate tokengs
cd /space/mawb/ssst_object_locus_text_refer_v1
export PYTHONPATH=/space/mawb/ssst_object_locus_text_refer_v1
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
python -u scripts/train_object_locus_text_refer_formal.py \
  --visual-checkpoint /space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt \
  --data-root /space/mawb/SIU3R/data/scannet \
  --output-dir /space/mawb/ssst/group_plus/object_locus_text_refer_v1_train \
  --checkpoint-dir /space/mawb/ssst/workspace_group_plus/object_locus_text_refer_v1_train
