"""Deferred fixed-epoch-64 evaluation entry point; never called by training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from scripts.object_locus_mask_guided_runtime import (
    arm_dirs, build_model, build_batch, manifest_and_plan, write_json,
)
from scripts.eval_object_locus_panoptic_v1 import evaluate_windows


def evaluate_arm(arm, device):
    reports, run = arm_dirs(arm)
    manifest, _ = manifest_and_plan()
    checkpoint = run / 'checkpoint_epoch_64.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'endpoint checkpoint not present: {checkpoint}')
    model, opt = build_model(arm, device, report=False)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if blob['completed_updates'] != 448 or blob['exposures'] != 3584:
        raise RuntimeError('evaluation requires the registered 448-update endpoint')
    model.load_state_dict(blob['model'], strict=True)
    del blob
    model.understanding_step=3584
    model.eval()
    splits = ('train_all56', 'same_scene_holdout8', 'dev8', 'val32')
    results = {}
    for split in splits:
        result, per_gt, queries = evaluate_windows(model, opt, manifest[split], 448,
            split, reports, device, build_batch, official=True, panels=True)
        result['image_quality'] = image_quality_windows(model,opt,manifest[split],device)
        write_json(reports/f'eval_{split}_step0448.json',result)
        results[split] = result
        write_json(reports / f'per_gt_epoch64_{split}.json', per_gt)
        write_json(reports / f'queries_epoch64_{split}.json', queries)
    output = dict(arm=arm, checkpoint=str(checkpoint), epoch=64, updates=448,
        exposures=3584, splits=results,
        scope_protocol={'context':'official.all.context', 'target_all':'official.all.target',
                        'true_novel':'official.novel.target'},
        camera_poses='ground-truth posed setting', paired_scene_bootstrap='computed by compare mode')
    write_json(reports / 'eval_epoch64.json', output)
    return output


def image_quality_windows(model,opt,windows,device):
    """Report per-image PSNR, SSIM and LPIPS for the registered scopes."""
    from scripts.eval_object_locus_v3_set import _run
    from scripts.object_locus_v3_set_runtime import capture_rng,restore_rng
    from tokengs.utils.metrics import MetricsCalculator
    metrics=MetricsCalculator(device=device,lpips_net='vgg')
    values={scope:{key:[] for key in ('psnr','ssim','lpips')} for scope in ('context','target_all','novel')}
    scenes={}
    was=model.training;rng=capture_rng();model.eval()
    try:
        with torch.no_grad():
            for window in windows:
                batch,out=_run(model,opt,window,build_batch,device)
                frame_ids=[int(v) for v in batch['frame_ids'][0].cpu().tolist()]
                indices={'context':[0,1],'target_all':list(range(len(frame_ids))),
                         'novel':[i for i,v in enumerate(frame_ids) if v in set(map(int,window['novel']))]}
                pred=out['render']['images_pred'];gt=batch['images_all']
                scene_values=scenes.setdefault(window['scene'],
                    {scope:{key:[] for key in ('psnr','ssim','lpips')} for scope in ('context','target_all','novel')})
                for scope,idx in indices.items():
                    if not idx:continue
                    row=metrics.calculate_all_metrics(pred[:,idx],gt[:,idx],reduction='none')
                    for key,value in row.items():
                        items=value.detach().cpu().reshape(-1).tolist()
                        values[scope][key].extend(items);scene_values[scope][key].extend(items)
    finally:
        restore_rng(rng);model.train(was)
    aggregate={scope:{key:float(sum(items)/len(items)) if items else None for key,items in row.items()}
               for scope,row in values.items()}
    by_scene={scene:{scope:{key:float(sum(items)/len(items)) if items else None for key,items in row.items()}
                     for scope,row in metrics.items()} for scene,metrics in scenes.items()}
    return dict(aggregate=aggregate,by_scene=by_scene)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('control', 'mask_guided', 'both'), default='both')
    parser.add_argument('--compare', action='store_true', help='compare fixed endpoint outputs after both evals')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('fixed endpoint evaluation uses the original CUDA evaluator')
    device = torch.device('cuda:0')
    arms = ('control', 'mask_guided') if args.arm == 'both' else (args.arm,)
    outputs = {arm: evaluate_arm(arm, device) for arm in arms}
    if args.compare:
        if set(outputs) != {'control', 'mask_guided'}:
            raise ValueError('--compare requires --arm both')
        compare_endpoints(outputs['control'], outputs['mask_guided'])


def compare_endpoints(control, mask_guided):
    """Create fixed endpoint deltas and registered conclusion-gate summary."""
    def metric(blob, split, scope, path):
        x = blob['splits'][split]
        for key in path:
            x = x[key]
        return float(x)
    hold_c = metric(control, 'same_scene_holdout8', 'novel', ('official','novel','target_map','map_50'))
    hold_m = metric(mask_guided, 'same_scene_holdout8', 'novel', ('official','novel','target_map','map_50'))
    rows = []
    for split in ('train_all56','same_scene_holdout8','dev8','val32'):
        for scope in ('context','target_all','novel'):
            cc=control['splits'][split]['local'][scope]
            mm=mask_guided['splits'][split]['local'][scope]
            arm_key='novel' if scope=='novel' else 'all'
            view_key='context' if scope=='context' else 'target'
            oc=control['splits'][split]['official'][arm_key]
            om=mask_guided['splits'][split]['official'][arm_key]
            official_control={k:oc.get(view_key+'_map',{}).get(k) if k in ('map','map_50') else oc.get(view_key+'_'+k)
                              for k in ('miou','pq','map','map_50')}
            official_mask_guided={k:om.get(view_key+'_map',{}).get(k) if k in ('map','map_50') else om.get(view_key+'_'+k)
                                  for k in ('miou','pq','map','map_50')}
            rows.append(dict(split=split,scope=scope,
                candidate_map_delta=mm.get('candidate_ap',{}).get('map',0)-cc.get('candidate_ap',{}).get('map',0),
                candidate_ap50_delta=mm.get('candidate_ap',{}).get('map_50',0)-cc.get('candidate_ap',{}).get('map_50',0),
                psnr_delta=mask_guided['splits'][split]['image_quality']['aggregate'][scope]['psnr']-control['splits'][split]['image_quality']['aggregate'][scope]['psnr'],
                ssim_delta=mask_guided['splits'][split]['image_quality']['aggregate'][scope]['ssim']-control['splits'][split]['image_quality']['aggregate'][scope]['ssim'],
                lpips_delta=mask_guided['splits'][split]['image_quality']['aggregate'][scope]['lpips']-control['splits'][split]['image_quality']['aggregate'][scope]['lpips'],
                official_control=official_control,official_mask_guided=official_mask_guided,
                official_map_delta=official_mask_guided['map']-official_control['map'] if official_mask_guided['map'] is not None and official_control['map'] is not None else None,
                official_ap50_delta=official_mask_guided['map_50']-official_control['map_50'] if official_mask_guided['map_50'] is not None and official_control['map_50'] is not None else None))
    root=Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1')
    ci_path=root/'paired_scene_bootstrap.json'
    paired_scene_bootstrap(root/'control/official/step_0448/same_scene_holdout8/novel',
        root/'mask_guided/official/step_0448/same_scene_holdout8/novel',ci_path)
    ci=json.loads(ci_path.read_text())
    guardrails=[]
    for split in ('train_all56','same_scene_holdout8','dev8','val32'):
        co=control['splits'][split]['official']['all']['context_map']['map_50']
        mo=mask_guided['splits'][split]['official']['all']['context_map']['map_50']
        ciq=control['splits'][split]['image_quality']['aggregate'];miq=mask_guided['splits'][split]['image_quality']['aggregate']
        context_delta=None if mo is None or co is None else float(mo)-float(co)
        guardrails.append(dict(split=split,context_official_ap50_delta=context_delta,
            context_psnr_delta=miq['context']['psnr']-ciq['context']['psnr'],
            true_novel_psnr_delta=miq['novel']['psnr']-ciq['novel']['psnr']))
    context_train_ok=guardrails[0]['context_official_ap50_delta'] is not None and guardrails[0]['context_official_ap50_delta']>=-.02
    psnr_ok=all(x['context_psnr_delta']>=-.5 and x['true_novel_psnr_delta']>=-.5 for x in guardrails)
    lower=ci.get('ap50_difference_ci95',[None,None])[0]
    effective=lower is not None and lower>0 and context_train_ok and psnr_ok
    conclusion='B: small-scale feedback modification shows task value.' if effective else 'C: this experiment did not establish task value.'
    endpoint=Path('/space/mawb/ssst/workspace_group_plus/object_locus_mask_guided_v1/mask_guided/checkpoint_epoch_64.pt')
    saved=torch.load(endpoint,map_location='cpu',weights_only=False,mmap=True)['model']
    inject_norms={lid:float(saved[f'panoptic.layers.{lid}.W_inject.weight'].norm()) for lid in ('L6','L8','L10','L12')}
    intervention_active=any(v>0 for v in inject_norms.values())
    smoke_paths=[Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1')/arm/'eight_smoke.json'
                 for arm in ('control','mask_guided')]
    engineering_smoke=all(p.is_file() and json.loads(p.read_text()).get('status')=='PASS' for p in smoke_paths)
    if not intervention_active:
        conclusion='未形成有效干预：mask-guided 四层注入权重均未活动。'
    if not effective:
        conclusion += f" Holdout true-novel official packed AP50 M-C={hold_m-hold_c:.6f}; 95% CI={ci.get('ap50_difference_ci95')}"
    report = dict(primary_metric='same_scene_holdout8 true-novel official packed AP50',
        primary_delta=hold_m-hold_c, paired_scene_bootstrap=ci, rows=rows,
        guardrails=guardrails, context_train_guardrail_pass=context_train_ok,
        psnr_guardrails_pass=psnr_ok, injection_weight_norms=inject_norms,
        engineering_status='A: route/gradient/smoke path connected' if engineering_smoke and intervention_active else 'engineering path not established',
        effective_intervention=intervention_active, conclusion=conclusion,
        gates={'train_all56_context_ap50_delta_floor':-.02,
               'four_split_context_and_novel_psnr_delta_floor_db':-.5})
    write_json(root/'paired_endpoint_comparison.json',report)


def paired_scene_bootstrap(control_root,mask_guided_root,output):
    """Reaggregate official packed-mask predictions under paired scene resampling."""
    siu3r=Path('/space/mawb/SIU3R')
    pin='8ea80166be76854f938e90521f1a5b688b755c87'
    current=subprocess.check_output(['git','-C',str(siu3r),'rev-parse','HEAD'],text=True).strip()
    if current!=pin:raise RuntimeError(f'pinned evaluator revision mismatch: {current}')
    code=r'''import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/space/mawb/SIU3R')
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
from torchmetrics.detection import MeanAveragePrecision
roots=[Path(sys.argv[1]),Path(sys.argv[2])];output=Path(sys.argv[3])
if any(not p.is_dir() for p in roots):raise FileNotFoundError('both packed endpoint prediction roots are required')
cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,
 eval_target_miou=False,eval_target_pq=False,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
 id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(roots[0]))
ev=Evaluator(cfg);ev.setup();dirs=sorted(p.name for p in roots[0].iterdir() if p.is_dir())
if len(dirs)!=8:raise RuntimeError(f'expected eight holdout scenes, got {len(dirs)}')
cache=[{},{}]
for b,root in enumerate(roots):
 for name in dirs:
  if not (root/name/'target_seg_pred').is_dir():raise RuntimeError(f'missing matched scene prediction {root/name}')
  scene=name.split('_context')[0]
  data=ev.process_segmentation(root/name/'target_seg_pred',root/name/'target_seg_gt')
  cache[b].setdefault(scene,[]).append((data['map_pred'],data['map_gt']))
scenes=sorted(cache[0]);rng=np.random.default_rng(2026);delta=[]
for it in range(2000):
 sampled=rng.choice(scenes,size=len(scenes),replace=True);vals=[]
 for b in range(2):
  metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
  pairs=[pair for scene in sampled for pair in cache[b][scene]]
  metric.update([p for p,g in pairs],[g for p,g in pairs]);v=metric.compute()
  vals.append((float(v['map']),float(v['map_50'])))
 delta.append([vals[1][i]-vals[0][i] for i in range(2)])
 if (it+1)%100==0:print(f'paired scene bootstrap {it+1}/2000',flush=True)
ci=np.percentile(np.asarray(delta),[2.5,97.5],axis=0)
output.write_text(json.dumps(dict(status='AVAILABLE',unit='same_scene_holdout8 scene',seed=2026,resamples=2000,
 scenes=scenes,scope='true-novel official packed panoptic',map_difference_ci95=ci[:,0].tolist(),
 ap50_difference_ci95=ci[:,1].tolist(),differences=delta,
 method='Paired scene resampling; recompute global MeanAveragePrecision from each resample packed predictions/GT; M-C.'),indent=2)+'\n')
'''
    subprocess.run(['/space/mawb/SIU3R/.venv_gpu_v4/bin/python','-c',code,
        str(control_root),str(mask_guided_root),str(output)],check=True)


if __name__ == '__main__':
    main()
