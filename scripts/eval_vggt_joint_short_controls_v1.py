"""Read-only held-out evaluation with the pinned, unmodified SIU3R evaluator."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.train_vggt_joint_short_controls_v1 import ARMS,COHORT,SOURCE_SHA,AA_PREFIXES,check_clock,digest
CONDITIONS=tuple((arm,step) for arm in ARMS for step in (0,50,100))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def export_window(model,opt,w,root,device,*,observer=None):
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render,_official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.evaluate_ssst_validation import save_rgb,save_depth
    batch=build_batch(opt,w,device)
    output=generate_and_render(model,batch['images_input'],batch['images_all'])
    rendered=output['render'];view=_official_export_view(rendered)
    if any(not torch.isfinite(rendered[k]).all() for k in ('images_pred','depths_pred')):raise ValueError('nonfinite render')
    row=write_official_pair(view,batch,w,root/'all',target_frames='all')
    name=w['scene']+'_context'+'_'.join(map(str,w['context']));pair=root/'all'/name
    frames=batch['frame_ids'][0].cpu().tolist()
    if len(frames)!=6:raise ValueError('heldout provider views differ')
    for sub in ('rgb','rgb_gt','depth','depth_gt'):(pair/sub).mkdir(exist_ok=True)
    for i,f in enumerate(frames):
        filename=f"{w['scene']}_{f}.png"
        save_rgb(pair/'rgb'/filename,rendered['images_pred'][0,i]);save_rgb(pair/'rgb_gt'/filename,batch['images_all'][0,i])
        save_depth(pair/'depth'/filename,rendered['depths_pred'][0,i]/.15);save_depth(pair/'depth_gt'/filename,batch['depth_gt_m_all'][0,i])
    record=dict(name=name,source_index=w['source_index'],frame_ids=frames,
        input_rgb_sha256=__import__('hashlib').sha256(batch['images_all'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),official_pair=row)
    if observer is not None:record['diagnostics']=observer(output,batch)
    return record


def export(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.train_vggt_joint_short_controls_v1 import build
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render,_official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.evaluate_ssst_validation import save_rgb,save_depth
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('requires four evaluation workers')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha!=a.training_sha or sha!=os.environ['TASK_CODE_SHA']:raise ValueError('exact-code evaluator mismatch')
    windows=json.loads(COHORT.read_text())['heldout']
    if len(windows)!=16 or len({w['scene'] for w in windows})!=16:raise ValueError('wrong heldout count')
    model,opt,optimizer,_=build('frozen_warm',device);del optimizer
    model.frozen_vggt.set_reconstruction_adaptation(False,keep_fp32=True)
    model.eval().requires_grad_(False)
    for arm,step in CONDITIONS[rank::world]:
        root=a.root/arm/f'update{step:04d}'
        if root.exists():raise ValueError('preserve prior export attempt')
        complete=json.loads((a.run_root/arm/'COMPLETE.json').read_text())
        if (complete['status'],complete['completed_updates'],complete['completed_exposures'],complete['execution_git_sha'])!=('JOINT_SHORT_TRAINING_COMPLETED',100,1600,a.training_sha):raise ValueError('training incomplete')
        path=a.run_root/arm/f'checkpoint_update_{step:04d}.pt'
        receipt=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=receipt['sha256']:raise ValueError('checkpoint hash mismatch')
        b=torch.load(path,map_location='cpu',mmap=True,weights_only=False);check_clock(b,arm,step,a.training_sha)
        native.restore_model_state_strict(model,b['model'])
        native.restore_adapted_vggt(model,native.CONTINUATION_SOURCE.parent/'adapted_vggt.pt',native.CONTINUATION_VGGT_SHA)
        target=model.frozen_vggt.state_dict()
        expected={k for k in target if k.startswith(AA_PREFIXES)} if arm=='aa_warm' else set()
        if set(b['vggt_aa'])!=expected:raise ValueError('AA snapshot keys differ')
        with torch.no_grad():
            for key,value in b['vggt_aa'].items():target[key].copy_(value)
        model.understanding_step=50064+step*16
        meta=dict(arm=arm,step=step,checkpoint_sha256=receipt['sha256'],source_checkpoint_sha256=SOURCE_SHA,
            optimizer_updates=0,diagnostic_only=True,heldout_windows=16,heldout_scenes=16,
            execution_git_sha=sha,config=b['config'],precision='FP32 AA/BF16 DINO; same as training',records=[])
        del b,target
        versions={n:p._version for n,p in model.named_parameters()}
        with torch.no_grad():
            for index,w in enumerate(windows):
                meta['records'].append(export_window(model,opt,w,root,device))
                write(root/'export_progress.json',dict(completed=index+1,total=16,arm=arm,step=step))
        if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('evaluation changed parameters')
        write(root/'export_complete.json',dict(status='EXPORTED',**meta))
        print('EXPORTED',arm,step,flush=True)


def score(a):
    from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT
    from scripts.posefree_official_state_helpers import itemize
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS,markdown_table
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('four scoring workers required')
    torch.cuda.set_device(local);torch.set_num_threads(4)
    for arm,step in CONDITIONS[rank::world]:
        root=a.root/arm/f'update{step:04d}'
        e=json.loads((root/'export_complete.json').read_text())
        if e['status']!='EXPORTED' or len(e['records'])!=16 or e['optimizer_updates']!=0:raise ValueError('export incomplete')
        if (root/'metrics.json').exists():raise ValueError('preserve old score')
        # Each worker scores a complete 16-scene condition. mAP/PQ are computed
        # globally by the original evaluator, never averaged across windows.
        raw=itemize(evaluate(root/'all',device=f'cuda:{local}'))
        report=dict(status='SMALL_HELDOUT_EVALUATED',diagnostic_only=True,windows=16,unique_scenes=16,
            official_commit=SIU3R_COMMIT,checkpoint_sha256=e['checkpoint_sha256'],arm=arm,step=step,
            scopes={'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}|{'mIoU_t':None},'context':{'mIoU_t':None}},
            official_segmentation={'all':raw},official_native_result=raw,optimizer_updates=0)
        write(root/'metrics.json',report)
        from scripts.format_object_locus_posefree_siu3r_table import table_values
        values=table_values(report)
        with (root/'siu3r_table1.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerow(values)
        (root/'siu3r_table1.md').write_text('# Small held-out control: '+arm+' / '+str(step)+' updates\n\n16 held-out scenes / 16 windows; diagnostic only, not full SIU3R validation. Text mIoU columns are blank.\n\n'+markdown_table(report)+'\n')
        write(root/'score_complete.json',dict(status='SCORED',arm=arm,step=step,official_commit=SIU3R_COMMIT,optimizer_updates=0))
        print('SCORED',arm,step,flush=True)


def smoke_score(a):
    from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT
    from scripts.posefree_official_state_helpers import itemize
    from scripts.format_object_locus_posefree_siu3r_table import table_values
    root=a.root
    e=json.loads((root/'export_complete.json').read_text())
    if e['status']!='PREFLIGHT_EXPORTED' or e['optimizer_updates']!=0:raise ValueError('invalid real-window export proof')
    raw=itemize(evaluate(root/'all',device='cuda'))
    report=dict(scopes={'context':{'mIoU_t':None},'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}|{'mIoU_t':None}},official_segmentation={'all':raw})
    values=table_values(report)
    write(root/'score_complete.json',dict(status='OFFICIAL_13_COLUMN_PREFLIGHT_PASSED',values=values,
        official_commit=SIU3R_COMMIT,official_native_result=raw,optimizer_updates=0,diagnostic_only=True))


def reduce(a):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS,HEADERS,PRECISION,table_values
    conditions={};reference={};initial_rgb={}
    for arm,step in CONDITIONS:
        root=a.root/arm/f'update{step:04d}'
        e=json.loads((root/'export_complete.json').read_text());s=json.loads((root/'score_complete.json').read_text())
        if s['status']!='SCORED' or e['optimizer_updates']!=0 or s['optimizer_updates']!=0:raise ValueError('incomplete read-only evaluation')
        for row in e['records']:
            name=row['name'];p=root/'all'/name
            hashes={str(f.relative_to(p)):digest(f) for sub in ('rgb_gt','depth_gt','context_seg_gt','target_seg_gt') for f in (p/sub).glob('*') if f.is_file()}
            if name in reference and hashes!=reference[name]:raise ValueError('heldout GT differs across arms')
            reference[name]=hashes
            if arm in ('frozen_warm','aa_warm') and step==0:
                rgb={f.name:digest(f) for f in (p/'rgb').glob('*.png')}
                if name in initial_rgb and initial_rgb[name]!=rgb:raise ValueError('initial AA/frozen render differs')
                initial_rgb[name]=rgb
        conditions[f'{arm}/{step}']=table_values(json.loads((root/'metrics.json').read_text()))
    if len(reference)!=16:raise ValueError('heldout scene coverage mismatch')
    out=a.root/'analysis'
    if out.exists():raise ValueError('preserve previous analysis')
    out.mkdir()
    write(out/'results.json',dict(status='SMALL_HELDOUT_CONTROLS_EVALUATED',diagnostic_only=True,
        source_checkpoint_sha256=SOURCE_SHA,training_sha=a.training_sha,columns=FIELDS,conditions=conditions,
        heldout_gt_identical=True,aa_and_frozen_initial_rgb_identical=True,optimizer_updates=0))
    lines=['# AA / decoder small joint-task controls','',
        '16 held-out scenes, not the full validation set. Original official metrics; both text columns blank.','',
        '| Control / updates | '+' | '.join(HEADERS)+' |','|---|'+'---:|'*13]
    with (out/'siu3r_table1_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['control/updates',*FIELDS])
        for name,values in conditions.items():
            w.writerow([name,*values]);cells=['' if v is None else f'{v:.{p}f}' for v,p in zip(values,PRECISION)]
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(a.root/'COMPLETE.json',dict(status='SMALL_HELDOUT_CONTROLS_EVALUATED',conditions=9,
        windows=144,images=864,unique_heldout_scenes=16,optimizer_updates=0,training_sha=a.training_sha))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('export','score','reduce','smoke_score'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--training-sha',required=True)
    a=p.parse_args();globals()[a.mode](a)
