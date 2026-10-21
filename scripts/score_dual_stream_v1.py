"""Score completed dual-stream exports with the pinned, unchanged SIU3R evaluator."""
import argparse
import csv
import json
import math
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


def native_table_values(raw):
    result=[raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')]
    for view in ('context','target'):result.extend([raw[view+'_miou'],raw[view+'_map']['map'],raw[view+'_pq'],None])
    return result


def json_safe(value,path='$',invalid=None):
    if invalid is None:invalid=[]
    if isinstance(value,float) and not math.isfinite(value):invalid.append(path);return None
    if isinstance(value,dict):return {k:json_safe(v,path+'.'+k,invalid) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v,path+f'[{i}]',invalid) for i,v in enumerate(value)]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    root=args.run_root/'evaluation'
    exported=json.loads((root/'export_complete.json').read_text())
    EXPECTED_WINDOWS=len(exported['records'])
    if exported['status']!='EXPORTED' or EXPECTED_WINDOWS<1:raise ValueError('Export incomplete')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT:raise ValueError('Official evaluator commit changed')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Official evaluator src is modified')
    configuration=json.loads((args.run_root/'configuration.json').read_text())
    raw=itemize(evaluate(root/'all',device='cuda:0'))
    report=dict(status='OFFICIAL_VAL_SCORED',windows=EXPECTED_WINDOWS,official_commit=SIU3R_COMMIT,model_recipe=configuration['recipe'],
        scopes={'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}|{'mIoU_t':None},'context':{'mIoU_t':None}},
        official_segmentation={'all':raw},official_native_result=raw,parameter_updates=0,
        depth_units=exported['depth_units'],prediction_protocol=exported['prediction_protocol'])
    values=native_table_values(raw)
    invalid_metrics=[name for name,value in zip(FIELDS,values) if value is not None and not math.isfinite(value)]
    invalid_paths=[];safe=json_safe(report,invalid=invalid_paths);safe['nonfinite_native_paths']=invalid_paths
    if invalid_metrics:
        safe.update(status='HELDOUT128_NATIVE_METRICS_INVALID',invalid_required_metrics=invalid_metrics,native_result_path=str(root/'all/results.json'))
        (root/'metrics_invalid.json').write_text(json.dumps(safe,indent=2,allow_nan=False)+'\n')
        with (root/'siu3r_table1_invalid.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerow(['' if v is None else ('NaN' if not math.isfinite(v) else v) for v in values])
        raise FloatingPointError('Official metrics remain invalid (not filtered/repaired): '+', '.join(invalid_metrics))
    (root/'metrics.json').write_text(json.dumps(safe,indent=2,allow_nan=False)+'\n')
    with (root/'siu3r_table1.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerow(table_values(report))
    (root/'siu3r_table1.md').write_text('# '+configuration['recipe']+f' — official {EXPECTED_WINDOWS} pairs\n\n'+markdown_table(report)+'\n\nText mIoU columns remain blank. Target uses official context + four extra frames.\n')
    if EXPECTED_WINDOWS==128:
        reference=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
        with reference.open() as stream:
            reader=csv.reader(stream);fields=next(reader);old=next(reader)
        if fields!=list(FIELDS):raise ValueError('Reference metric columns differ')
        current=table_values(report)
        differences=[None if b is None or a=='' else b-float(a) for a,b in zip(old,current)]
        with (root/'comparison_to_old_C.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS));writer.writerow(['old_VGGT_MASt3R_C']+old)
            writer.writerow([configuration['recipe']+'_epoch4']+current);writer.writerow(['new_minus_old']+differences)
    training_receipt=args.run_root/'TRAINING_COMPLETE.json'
    (args.run_root/'COMPLETE.json').write_text(json.dumps(dict(status='OFFICIAL_EVAL_COMPLETE',windows=EXPECTED_WINDOWS,
        training=json.loads(training_receipt.read_text()) if training_receipt.exists() else None,
        metrics=str(root/'metrics.json')),indent=2)+'\n')

if __name__=='__main__':main()
