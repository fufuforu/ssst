"""Official scores for visual-freeze/feedback factorial, with preserved baseline reuse."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_feedback_factorial_v1 import ARMS,COHORT,EVAL_STEPS,SOURCE_SHA,check_clock,digest
from scripts.eval_vggt_joint_short_controls_v1 import export_window,write,smoke_score


def export(a):
    from scripts.train_vggt_feedback_factorial_v1 import build
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('four evaluation workers required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local):raise ValueError('all formal probe exports remain on4090 for matched inference hardware')
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha!=a.training_sha or sha!=os.environ['TASK_CODE_SHA']:raise ValueError('evaluation code identity differs')
    windows=json.loads(COHORT.read_text())['heldout']
    if len(windows)!=64 or len({w['scene'] for w in windows})!=64:raise ValueError('heldout coverage differs')
    complete=json.loads((a.run_root/'COMPLETE.json').read_text())
    if (complete['status'],complete['arm'],complete['completed_updates'],complete['completed_exposures'])!=('INTERACTION_TRAINING_COMPLETED',a.arm,500,4000):
        raise ValueError('training incomplete')
    model,opt,optimizer,_=build(a.arm,device);del optimizer
    model.eval().requires_grad_(False)
    for step in EVAL_STEPS[rank::world]:
        root=a.root/f'update{step:04d}'
        if root.exists():raise ValueError('preserve existing evaluation attempt')
        path=a.run_root/f'checkpoint_update_{step:04d}.pt';receipt=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=receipt['sha256']:raise ValueError('checkpoint SHA differs')
        b=torch.load(path,map_location='cpu',mmap=True,weights_only=False);check_clock(b,a.arm,step,a.training_sha)
        native.restore_model_state_strict(model,b['model']);model.understanding_step=step*8
        config=b['config'];del b
        versions={n:p._version for n,p in model.named_parameters()}
        records=[]
        with torch.no_grad():
            for index,w in enumerate(windows):
                records.append(export_window(model,opt,w,root,device))
                write(root/'export_progress.json',dict(arm=a.arm,step=step,completed=index+1,total=64))
        if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('evaluation mutated parameters')
        write(root/'export_complete.json',dict(status='EXPORTED',arm=a.arm,step=step,records=records,
            optimizer_updates=0,checkpoint_sha256=receipt['sha256'],config=config,execution_git_sha=sha))
        print('EXPORTED',a.arm,step,flush=True)


def score(a):
    from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT
    from scripts.posefree_official_state_helpers import itemize
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS,table_values,markdown_table
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('four scoring workers required')
    torch.cuda.set_device(local);torch.set_num_threads(4)
    conditions=[(step,a.root/f'update{step:04d}') for step in EVAL_STEPS[rank::world]]
    for step,root in conditions:
        e=json.loads((root/'export_complete.json').read_text())
        if e['status']!='EXPORTED' or len(e['records'])!=64 or e['optimizer_updates']!=0:raise ValueError('export incomplete')
        if (root/'metrics.json').exists():raise ValueError('preserve prior score')
        # One complete condition per worker: global AP/PQ, never per-scene averages.
        raw=itemize(evaluate(root/'all',device=f'cuda:{local}'))
        report=dict(status='SMALL_HELDOUT_EVALUATED',diagnostic_only=True,windows=64,unique_scenes=64,
            official_commit=SIU3R_COMMIT,checkpoint_sha256=e['checkpoint_sha256'],arm=a.arm,step=step,
            scopes={'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}|{'mIoU_t':None},'context':{'mIoU_t':None}},
            official_segmentation={'all':raw},official_native_result=raw,optimizer_updates=0)
        if 'intervention' in e:report['intervention']=e['intervention']
        values=table_values(report)
        if len(values)!=13 or values[8] is not None or values[12] is not None:raise ValueError('13-column contract differs')
        if not all(v is None or __import__('math').isfinite(v) for v in values):raise ValueError('nonfinite official score')
        write(root/'metrics.json',report)
        with (root/'siu3r_table1.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerow(values)
        (root/'siu3r_table1.md').write_text('64 held-out scenes; short diagnostic, not full validation.\n\n'+markdown_table(report)+'\n')
        write(root/'score_complete.json',dict(status='SCORED',arm=a.arm,step=step,optimizer_updates=0))
        print('SCORED',a.arm,step,flush=True)


def reduce(a):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS,HEADERS,PRECISION,table_values
    from scripts.eval_vggt_feedback_beta_scan_v1 import OLD_REPORT
    conditions={};reference={};initial_rgb={};provenance={}
    for arm in ARMS:
        for step in EVAL_STEPS:
            root=(OLD_REPORT if arm=='joint_full' else a.root/arm)/f'update{step:04d}'
            e=json.loads((root/'export_complete.json').read_text());s=json.loads((root/'score_complete.json').read_text())
            metrics=json.loads((root/'metrics.json').read_text())
            if s['status']!='SCORED' or e['optimizer_updates']!=0 or s['optimizer_updates']!=0:raise ValueError('incomplete evaluation')
            if e['checkpoint_sha256']!=metrics['checkpoint_sha256']:raise ValueError('score provenance differs')
            for row in e['records']:
                name=row['name'];pair=root/'all'/name
                hashes={str(f.relative_to(pair)):digest(f) for sub in ('rgb_gt','depth_gt','context_seg_gt','target_seg_gt') for f in (pair/sub).glob('*') if f.is_file()}
                if name in reference and hashes!=reference[name]:raise ValueError('heldout GT changed')
                reference[name]=hashes
                if step==0:
                    rgb={f.name:digest(f) for f in (pair/'rgb').glob('*.png')}
                    if name in initial_rgb and rgb!=initial_rgb[name]:raise ValueError('initial beta0 RGB differs')
                    initial_rgb[name]=rgb
            key=f'{arm}/{step}';conditions[key]=table_values(metrics)
            provenance[key]=dict(reused=arm=='joint_full',artifact_root=str(root),checkpoint_sha256=e['checkpoint_sha256'],execution_git_sha=e['execution_git_sha'])
    if len(reference)!=64:raise ValueError('heldout coverage differs')
    out=a.root/'analysis'
    if out.exists():raise ValueError('preserve prior analysis')
    out.mkdir(parents=True)
    write(out/'results.json',dict(status='FEEDBACK_FACTORIAL_EVALUATED',columns=FIELDS,conditions=conditions,
        provenance=provenance,small_feedback_peak=a.small_peak,heldout_gt_identical=True,initial_rgb_identical=True,
        source_checkpoint_sha256=SOURCE_SHA,diagnostic_only=True,optimizer_updates=0))
    lines=['# Visual feature / feedback factorial','', 'Full pool4000 unique windows;64 held-out scenes. Original joint baseline reused.', '',
           '| Control / updates | '+' | '.join(HEADERS)+' |','|---|'+'---:|'*13]
    with (out/'siu3r_table1_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['control/updates',*FIELDS])
        for name,values in conditions.items():
            w.writerow([name,*values]);cells=['' if v is None else f'{v:.{p}f}' for v,p in zip(values,PRECISION)]
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(a.root/'COMPLETE.json',dict(status='FEEDBACK_FACTORIAL_EVALUATED',conditions=16,
        unique_heldout_scenes=64,optimizer_updates=0,small_feedback_peak=a.small_peak,reused_baseline_conditions=4))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('export','score','reduce','smoke_score'),required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--training-sha',required=True)
    p.add_argument('--small-peak',type=float,choices=(.025,.05),required=True)
    a=p.parse_args()
    from scripts import train_vggt_feedback_factorial_v1 as trainer
    trainer.SMALL_PEAK=a.small_peak
    globals()[a.mode](a)
