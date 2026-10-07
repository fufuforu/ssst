#!/usr/bin/env bash
#SBATCH --job-name=prob-lift-paired
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-17
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1/attempts/retry01/logs/slurm-%j.out
#SBATCH --error=/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1/attempts/retry01/logs/slurm-%j.err
set -euo pipefail
repo=/space/mawb/ssst_probability_lift_eval_v1
report=/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1/attempts/retry01
TASK_MODEL_PYTHON=/space/mawb/anaconda3/envs/tokengs/bin/python
TASK_OFFICIAL_PYTHON=/space/mawb/SIU3R/.venv_gpu_v4/bin/python
mkdir -p "$report/logs"
if find "$report" -mindepth 1 -maxdepth 1 ! -name logs -print -quit | grep -q .; then
  echo "retry01 output conflict: $report contains pre-existing non-log content" >&2
  exit 2
fi
cd "$repo"
export TASK_MODEL_PYTHON TASK_OFFICIAL_PYTHON TASK_REPORT_ROOT="$report"
export TASK_WORKTREE="$repo"
export TORCH_EXTENSIONS_DIR="$report/torch_extensions"
"$TASK_MODEL_PYTHON" - <<'PY'
import importlib.metadata as md, importlib.util, json, os, pathlib, platform, socket, sys, traceback
root=pathlib.Path(os.environ['TASK_REPORT_ROOT']); info={'role':'model','executable':sys.executable,'versions':{}}
info['python_version']=sys.version;info['hostname']=socket.gethostname()
for name in ('torch','torchvision','torchmetrics','numpy','gsplat'):
 try: info['versions'][name]=md.version(name)
 except Exception as e: info['versions'][name]=f'{type(e).__name__}: {e}'
info['lpips_importable']=bool(importlib.util.find_spec('lpips'))
try:
 import torch
 info['cuda_available']=torch.cuda.is_available()
 info['cuda_device_name']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
 info['cuda_runtime']=torch.version.cuda
 info['cuda_driver_version']=torch._C._cuda_getDriverVersion() if hasattr(torch._C,'_cuda_getDriverVersion') else None
 if info['hostname']!='3dimage-17' or not info['cuda_available'] or info['cuda_device_name']!='NVIDIA GeForce RTX 4090':
  raise RuntimeError('model environment hardware contract mismatch')
 sys.path.insert(0,os.environ['TASK_WORKTREE'])
 from scripts import object_locus_panoptic_v1_runtime
 from tokengs.models.object_locus_probability_lift_eval_v1 import LocusGSObjectLocusProbabilityLiftEvalV1
 info['gsplat_extension_path']=getattr(sys.modules.get('gsplat_cuda'),'__file__',None)
 info['gsplat_extension_status']='PRELOADED' if info['gsplat_extension_path'] else 'JIT_ON_FIRST_USE'
 info['runtime_import']='PASS'
except Exception:
 info['runtime_import']='FAIL';info['traceback']=traceback.format_exc()
(root/'model_environment.json').write_text(json.dumps(info,indent=2)+'\n')
if info['runtime_import']!='PASS': raise SystemExit(info['traceback'])
PY
"$TASK_OFFICIAL_PYTHON" - <<'PY'
import importlib.metadata as md, importlib.util, json, os, pathlib, platform, sys, traceback
root=pathlib.Path(os.environ['TASK_REPORT_ROOT']); info={'role':'official','executable':sys.executable,'versions':{}}
info['python_version']=sys.version
info['lpips_importable']=bool(importlib.util.find_spec('lpips'))
for name in ('torch','torchvision','torchmetrics','numpy','gsplat'):
 try: info['versions'][name]=md.version(name)
 except Exception as e: info['versions'][name]=f'{type(e).__name__}: {e}'
try:
 sys.path.insert(0,'/space/mawb/SIU3R')
 from src.evaluator import Evaluator
 from torchmetrics.detection import MeanAveragePrecision
 import pycocotools.mask
 info['siu3r_import']='PASS'
except Exception:
 info['siu3r_import']='FAIL';info['traceback']=traceback.format_exc()
(root/'official_environment.json').write_text(json.dumps(info,indent=2)+'\n')
if info['siu3r_import']!='PASS': raise SystemExit(info['traceback'])
PY
"$TASK_MODEL_PYTHON" scripts/check_object_locus_probability_lift_v1.py > "$report/contracts.json"
"$TASK_MODEL_PYTHON" scripts/eval_object_locus_probability_lift_v1.py --report-root "$report" --smoke
"$TASK_MODEL_PYTHON" scripts/eval_object_locus_probability_lift_v1.py --report-root "$report"
