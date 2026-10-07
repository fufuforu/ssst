#!/usr/bin/env bash
#SBATCH --job-name=prob-lift-paired
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-17
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1/logs/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1/logs/slurm-%j.err
set -euo pipefail
repo=/space/mawb/ssst_probability_lift_eval_v1
report=/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1
python=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
mkdir -p "$report/logs"
cd "$repo"
"$python" scripts/check_object_locus_probability_lift_v1.py > "$report/contracts.json"
"$python" scripts/eval_object_locus_probability_lift_v1.py --smoke
"$python" scripts/eval_object_locus_probability_lift_v1.py
