"""Score completed dual-stream exports with the pinned, unchanged SIU3R evaluator."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT,SIU3R_REPO
from scripts.format_object_locus_posefree_siu3r_table import FIELDS,table_values,markdown_table


def itemize(x):
    if hasattr(x,'detach'):return x.detach().cpu().tolist()
    if isinstance(x,dict):return {k:itemize(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [itemize(v) for v in x]
    return x


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    root=args.run_root/'evaluation'
    exported=json.loads((root/'export_complete.json').read_text())
    if exported['status']!='EXPORTED' or len(exported['records'])!=128:raise ValueError('Export incomplete')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT:raise ValueError('Official evaluator commit changed')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Official evaluator src is modified')
    raw=itemize(evaluate(root/'all',device='cuda:0'))
    report=dict(status='HELDOUT128_OFFICIALLY_SCORED',windows=128,official_commit=SIU3R_COMMIT,
        scopes={'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}|{'mIoU_t':None},'context':{'mIoU_t':None}},
        official_segmentation={'all':raw},official_native_result=raw,parameter_updates=0,
        depth_units=exported['depth_units'],prediction_protocol=exported['prediction_protocol'])
    (root/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (root/'siu3r_table1.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerow(table_values(report))
    (root/'siu3r_table1.md').write_text('# Omega + SAM dual stream — held-out 128 scenes\n\n'+markdown_table(report)+'\n\nText mIoU columns remain blank. Target uses official context + four extra frames.\n')
    reference=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
    with reference.open() as stream:
        reader=csv.reader(stream);fields=next(reader);old=next(reader)
    if fields!=list(FIELDS):raise ValueError('Reference metric columns differ')
    current=table_values(report)
    differences=[None if b is None or a=='' else b-float(a) for a,b in zip(old,current)]
    with (root/'comparison_to_old_C.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS));writer.writerow(['old_VGGT_MASt3R_C']+old)
        writer.writerow(['Omega_SAM_dual_stream_epoch4']+current);writer.writerow(['new_minus_old']+differences)
    (args.run_root/'COMPLETE.json').write_text(json.dumps(dict(status='TRAINING_AND_HELDOUT128_OFFICIAL_EVAL_COMPLETE',training=json.loads((args.run_root/'TRAINING_COMPLETE.json').read_text()),metrics=str(root/'metrics.json')),indent=2)+'\n')

if __name__=='__main__':main()
