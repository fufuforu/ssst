"""One-scene end-to-end evaluator check; never an official heldout128 score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_REPO,SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize
from scripts.format_object_locus_posefree_siu3r_table import table_values

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    root=args.run_root/'protocol_smoke'
    manifest=json.loads((root/'export_complete.json').read_text())
    if manifest['status']!='SMOKE_EXPORTED' or len(manifest['records'])!=1:raise ValueError('Expected isolated one-window smoke')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT:raise ValueError('Evaluator version changed')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Evaluator source modified')
    raw=itemize(evaluate(root/'all',device='cuda:0'))
    report=dict(scopes={'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}|{'mIoU_t':None},'context':{'mIoU_t':None}},official_segmentation={'all':raw})
    values=table_values(report)
    (root/'SMOKE_PASSED.json').write_text(json.dumps(dict(status='ONE_SCENE_PROTOCOL_SMOKE_PASSED',official_heldout128_score=False,parameter_updates=0,values=values,official_commit=SIU3R_COMMIT),indent=2)+'\n')
