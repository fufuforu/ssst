#!/usr/bin/env python3
"""Deferred fixed epoch-64 evaluation; do not call until the user requests it."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from scripts import object_locus_mh_feedback_runtime as runtime
from scripts.eval_object_locus_panoptic_v1 import evaluate_windows


def evaluate_mh(device):
    from scripts.object_locus_v3_set_runtime import write_json
    from scripts.object_locus_v3_set_runtime import capture_rng, restore_rng
    import scripts.eval_object_locus_panoptic_v1 as panoptic_eval
    import scripts.export_object_locus_v3_set_official as official_export
    from scripts.eval_object_locus_v3_set import _run
    from tokengs.utils.metrics import MetricsCalculator

    manifest, _ = runtime.manifest_and_plan()
    model, opt = runtime.build_model(device, report=False)
    path = runtime.RUN_ROOT/'checkpoint_epoch_64.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if (blob['completed_updates'],blob['exposures'],blob['epoch']) != (448,3584,64):
        raise RuntimeError('only the fixed epoch-64 endpoint can be evaluated')
    model.load_state_dict(blob['model'], strict=True)
    del blob
    model.understanding_step = 3584
    model.eval()
    # Export packed predictions and calculate all local/image-quality metrics in
    # the same _run call. The official evaluator then reads those exact files.
    results={}
    for split in ('train_all56','same_scene_holdout8','dev8','val32'):
        metrics=MetricsCalculator(device=device,lpips_net='vgg')
        values={s:{k:[] for k in ('psnr','ssim','lpips')} for s in ('context','target_all','novel')}
        scenes={}
        step_root=runtime.REPORT_ROOT/'official/step_0448'/split
        from scripts.export_object_locus_v3_set_official import write_official_pair
        original_run=panoptic_eval._run
        original_export=official_export.export_windows
        def cached_run(m,opt0,window,builder,dev):
            batch,out=original_run(m,opt0,window,builder,dev)
            write_official_pair(out,batch,window,step_root/'all',target_frames='all')
            write_official_pair(out,batch,window,step_root/'novel',target_frames='novel')
            ids=[int(x) for x in batch['frame_ids'][0].cpu().tolist()]
            scopes={'context':[0,1],'target_all':list(range(len(ids))),
                'novel':[i for i,x in enumerate(ids) if x in set(map(int,window['novel']))]}
            scene=scenes.setdefault(window['scene'],{s:{k:[] for k in ('psnr','ssim','lpips')} for s in scopes})
            for scope,indices in scopes.items():
                if not indices:continue
                row=metrics.calculate_all_metrics(out['render']['images_pred'][:,indices],
                    batch['images_all'][:,indices],reduction='none')
                for key,val in row.items():
                    xs=val.detach().cpu().reshape(-1).tolist()
                    values[scope][key].extend(xs);scene[scope][key].extend(xs)
            return batch,out
        def exports_from_cache(*args,**kwargs):
            return {'records':[],'target_set':kwargs.get('target_frames','all'),'source':'same-pass prediction cache'}
        panoptic_eval._run=cached_run
        official_export.export_windows=exports_from_cache
        try:
            result,per_gt,queries=evaluate_windows(model,opt,manifest[split],448,split,
                runtime.REPORT_ROOT,device,runtime.build_batch,official=True,panels=True)
        finally:
            panoptic_eval._run=original_run
            official_export.export_windows=original_export
        quality={'aggregate':{s:{k:(sum(v)/len(v) if v else None) for k,v in x.items()} for s,x in values.items()},
            'by_scene':{scene:{s:{k:(sum(v)/len(v) if v else None) for k,v in scopes.items()} for s,scopes in row.items()} for scene,row in scenes.items()}}
        result['image_quality']=quality
        write_json(runtime.REPORT_ROOT/f'eval_{split}_step0448.json',result)
        report=dict(split=split,epoch=64,updates=448,exposures=3584,scopes=result,
            scope_mapping={'context':'official.all.context','target_all':'official.all.target','true_novel':'official.novel.target'},
            camera_poses='ground-truth posed setting')
        write_json(runtime.REPORT_ROOT/f'eval_{split}_epoch64.json',report)
        write_json(runtime.REPORT_ROOT/f'per_gt_epoch64_{split}.json',per_gt)
        write_json(runtime.REPORT_ROOT/f'queries_epoch64_{split}.json',queries)
        results[split]=report
    final=dict(arm='mask_guided_multi_head',checkpoint=str(path),epoch=64,updates=448,exposures=3584,
        splits=results,camera_poses='ground-truth posed setting',evaluation_protocol='deferred fixed endpoint')
    write_json(runtime.REPORT_ROOT/'eval_epoch64.json',final)
    compare_with_control(final,model=None,opt=opt,manifest=manifest,device=device)
    return final


def compare_with_control(mh,model,opt,manifest,device):
    """Reuse C's saved metrics and regenerate only its paired holdout bootstrap inputs."""
    from scripts.object_locus_v3_set_runtime import write_json
    from scripts.export_object_locus_v3_set_official import export_windows
    c_path=runtime.C_REPORT/'eval_epoch64.json'
    c=json.loads(c_path.read_text())
    if c.get('epoch')!=64 or c.get('updates')!=448 or c.get('exposures')!=3584:
        raise RuntimeError('fixed C epoch64 result is missing or has the wrong endpoint')

    # The scoped cleanup removed C's bulk prediction PNGs. Recreate only the
    # val32 true-novel packed pairs needed by the registered paired bootstrap;
    # this does not recompute or replace any saved C metrics.
    c_model,c_opt=runtime.locked.build_model('control',device,report=False)
    c_checkpoint=runtime.C_RUN/'model_only/control_epoch_64.pt'
    c_blob=torch.load(c_checkpoint,map_location='cpu',mmap=True,weights_only=False)
    c_model.load_state_dict(c_blob['model'],strict=True);del c_blob
    c_model.understanding_step=3584
    bootstrap_c=runtime.REPORT_ROOT/'bootstrap_inputs/control/val32/true_novel'
    export_windows(c_model,c_opt,manifest['val32'],bootstrap_c,device=device,
        batch_builder=runtime.build_batch,target_frames='novel')
    del c_model,c_opt

    bootstrap_path=runtime.REPORT_ROOT/'paired_scene_bootstrap.json'
    paired_scene_bootstrap_val32(bootstrap_c,
        runtime.REPORT_ROOT/'official/step_0448/val32/novel',bootstrap_path)
    bootstrap=json.loads(bootstrap_path.read_text())
    rows=[]
    for split in ('train_all56','same_scene_holdout8','dev8','val32'):
        c_split=c['splits'][split];m_split=mh['splits'][split]['scopes']
        for scope in ('context','target_all','novel'):
            cs=c_split['local'][scope];ms=m_split['local'][scope]
            official_scope='novel' if scope=='novel' else 'all'
            official_view='context' if scope=='context' else 'target'
            co=c_split['official'][official_scope];mo=m_split['official'][official_scope]
            def get_official(blob,metric):
                if metric in ('map','map_50'):
                    return blob.get(official_view+'_map',{}).get(metric)
                return blob.get(official_view+'_'+metric)
            cq=c_split.get('image_quality',{}).get('aggregate',{}).get(scope,{})
            mq=mh['splits'][split].get('image_quality',{}).get('aggregate',{}).get(scope,{})
            for metric,keyc,keym in (
                ('official_mIoU','miou','miou'),('official_PQ','pq','pq'),
                ('official_mAP','map','map'),('official_AP50','map_50','map_50'),
                ('candidate_mAP','candidate_ap.map','candidate_ap.map'),
                ('candidate_AP50','candidate_ap.map_50','candidate_ap.map_50'),
                ('PSNR','psnr','psnr'),('SSIM','ssim','ssim'),('LPIPS','lpips','lpips')):
                def nested(x,path):
                    for k in path.split('.'):x=x[k]
                    return float(x)
                if metric.startswith('official_'):
                    cv=get_official(co,keyc);mv=get_official(mo,keym)
                elif metric in ('PSNR','SSIM','LPIPS'):
                    cv=cq.get(keyc);mv=mq.get(keym)
                else:
                    cv=nested(cs,keyc);mv=nested(ms,keym)
                rows.append(dict(split=split,scope=scope,metric=metric,control=cv,mh=mv,
                    mh_minus_control=(float(mv)-float(cv)) if cv is not None and mv is not None else None))
            for metric,path in (
                ('candidate_CA_precision','candidate_ca.precision'),('candidate_CA_recall','candidate_ca.recall'),
                ('candidate_CW_precision','candidate_cw.precision'),('candidate_CW_recall','candidate_cw.recall'),
                ('raw_best_mask_coverage','raw_best_iou_ge_0_5_fraction'),
                ('matched_class_accuracy','matched_19_class_accuracy'),
                ('panoptic_CA_precision','panoptic_ca.precision'),('panoptic_CA_recall','panoptic_ca.recall'),
                ('panoptic_CW_precision','panoptic_cw.precision'),('panoptic_CW_recall','panoptic_cw.recall')):
                cv=nested(cs,path);mv=nested(ms,path)
                rows.append(dict(split=split,scope=scope,metric=metric,control=cv,mh=mv,mh_minus_control=mv-cv))
    primary=next(row for row in rows if row['split']=='val32' and row['scope']=='novel' and row['metric']=='official_mAP')
    def delta(split,scope,metric):
        return next(row['mh_minus_control'] for row in rows if row['split']==split and row['scope']==scope and row['metric']==metric)
    ci_low=(bootstrap.get('map_difference_ci95') or [None,None])[0]
    gates={
        'val32_true_novel_official_mAP_ci_lower_gt_zero':ci_low is not None and float(ci_low)>0,
        'val32_true_novel_AP50_not_lower_than_minus_0_01':delta('val32','novel','official_AP50')>=-0.01,
        'val32_true_novel_PQ_not_lower_than_minus_0_01':delta('val32','novel','official_PQ')>=-0.01,
        'train_all56_context_AP50_not_lower_than_minus_0_02':delta('train_all56','context','official_AP50')>=-0.02,
        'four_split_context_and_novel_PSNR_not_lower_than_minus_0_5db':all(
            delta(split,scope,'PSNR')>=-0.5 for split in ('train_all56','same_scene_holdout8','dev8','val32') for scope in ('context','novel'))}
    improves=all(gates.values())
    write_json(runtime.REPORT_ROOT/'paired_endpoint_comparison.json',dict(
        control_endpoint=str(runtime.C_RUN/'model_only/control_epoch_64.pt'),mh_endpoint=str(runtime.RUN_ROOT/'checkpoint_epoch_64.pt'),
        updates=448,exposures=3584,primary_metric='val32 true-novel official packed mAP',
        primary_delta=primary['mh_minus_control'],paired_scene_bootstrap=bootstrap,rows=rows,
        scope_mapping={'context':'official.all.context','target_all':'official.all.target','true_novel':'official.novel.target'},
        control_metrics_reused=True,control_prediction_scope_recreated='val32 true-novel bootstrap input only',
        gates=gates,conclusion=('预注册收益判据全部通过。' if improves else '未达到预注册收益判据，点估计与其他指标方向混合或不确定；不自动调参或追加训练。'),
        camera_poses='ground-truth posed setting',cross_scene_claim='dev8/val32 independently reported; posed with ground-truth camera poses'))


def paired_scene_bootstrap_val32(control_root,mh_root,output):
    """Paired scene bootstrap, recomputing global official packed AP each draw."""
    siu3r=Path('/space/mawb/SIU3R')
    pinned='8ea80166be76854f938e90521f1a5b688b755c87'
    actual=subprocess.check_output(['git','-C',str(siu3r),'rev-parse','HEAD'],text=True).strip()
    if actual!=pinned:raise RuntimeError('pinned official evaluator revision mismatch')
    code=r'''import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/space/mawb/SIU3R')
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
from torchmetrics.detection import MeanAveragePrecision
roots=[Path(sys.argv[1]),Path(sys.argv[2])];output=Path(sys.argv[3])
if any(not p.is_dir() for p in roots):raise FileNotFoundError('both paired val32 prediction roots are required')
cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,
 eval_target_miou=False,eval_target_pq=False,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
 id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(roots[0]))
ev=Evaluator(cfg);ev.setup();dirs=sorted(p.name for p in roots[0].iterdir() if p.is_dir())
if len(dirs)!=32:raise RuntimeError(f'expected 32 val32 scenes/windows groups, got {len(dirs)}')
cache=[{},{}]
for arm,root in enumerate(roots):
 for name in dirs:
  if not (root/name/'target_seg_pred').is_dir():raise RuntimeError(f'missing paired novel target export {root/name}')
  scene=name.split('_context')[0]
  data=ev.process_segmentation(root/name/'target_seg_pred',root/name/'target_seg_gt')
  cache[arm].setdefault(scene,[]).append((data['map_pred'],data['map_gt']))
if set(cache[0])!=set(cache[1]):raise RuntimeError('paired bootstrap scene IDs differ')
scenes=sorted(cache[0]);rng=np.random.default_rng(2026);draws=[]
for it in range(2000):
 sampled=rng.choice(scenes,size=len(scenes),replace=True);values=[]
 for arm in range(2):
  metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
  pairs=[item for scene in sampled for item in cache[arm][scene]]
  metric.update([p for p,g in pairs],[g for p,g in pairs]);v=metric.compute()
  values.append((float(v['map']),float(v['map_50'])))
 draws.append([values[1][j]-values[0][j] for j in range(2)])
ci=np.percentile(np.asarray(draws),[2.5,97.5],axis=0)
output.write_text(json.dumps(dict(status='AVAILABLE',unit='val32 scene',seed=2026,resamples=2000,
 scenes=scenes,scope='true-novel official packed panoptic',map_difference_ci95=ci[:,0].tolist(),
 ap50_difference_ci95=ci[:,1].tolist(),differences=draws,
 method='paired scene resampling; recompute global MeanAveragePrecision on each resample; MH-C'),indent=2)+'\n')
'''
    subprocess.run(['/space/mawb/SIU3R/.venv_gpu_v4/bin/python','-c',code,
        str(control_root),str(mh_root),str(output)],check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',action='store_true',help='explicitly run only after the user confirms training has completed')
    args=ap.parse_args()
    if not args.run:
        raise SystemExit('Evaluation is deferred. Reinvoke with --run only after explicit user notice.')
    if not torch.cuda.is_available():raise RuntimeError('registered endpoint evaluator requires CUDA')
    evaluate_mh(torch.device('cuda:0'))


if __name__=='__main__':main()
