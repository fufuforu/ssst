"""Refuse short training unless the same-snapshot CUDA smoke passed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,subprocess
from scripts.instance_grounding_smoke_policy import require_smoke


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    root=a.run_root;receipt=json.loads((root/'SMOKE_PASSED.json').read_text())
    require_smoke(receipt,hashlib.sha256((root/'preflight.json').read_bytes()).hexdigest(),subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    preflight=json.loads((root/'preflight.json').read_text())
    if 'instance_coupling_recipe' in preflight['extra_checkpoint_metadata']:
        from scripts.instance_coupling_smoke_policy import validate_coupling_smoke
        validate_coupling_smoke(receipt,preflight)
    if 'gaussian_consistency_recipe' in preflight['extra_checkpoint_metadata']:
        from scripts.gaussian_consistency_smoke_policy import validate_direct_smoke
        validate_direct_smoke(receipt)
    if any((root/name).exists() for name in ('STARTED.json','checkpoint_latest.pt')):raise ValueError('Short training already exists')
    print('Smoke passed; start fresh 16-scene/256-update training. Smoke parameters and Adam are discarded.')


if __name__=='__main__':main()
