"""Deferred fixed endpoint evaluator; training never invokes this module."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from scripts.object_locus_region_class_runtime import (
    REPORT_ROOT, arm_dirs, build_model, build_batch, manifest_and_plan, write_json,
)
from scripts.eval_object_locus_panoptic_v1 import evaluate_windows

SPLITS = ('train_all56', 'same_scene_holdout8', 'dev8', 'val32')


def evaluate_arm(arm, device):
    reports, run = arm_dirs(arm)
    endpoint = run / 'checkpoint_epoch_64.pt'
    if not endpoint.is_file():
        raise FileNotFoundError(f'fixed epoch64 checkpoint missing: {endpoint}')
    model, opt = build_model(arm, device, report=False)
    blob = torch.load(endpoint, map_location='cpu', mmap=True, weights_only=False)
    if (blob.get('epoch'), blob.get('completed_updates'), blob.get('exposures')) != (64, 448, 3584):
        raise RuntimeError('fixed epoch64 endpoint must be 64/448/3584')
    model.load_state_dict(blob['model'], strict=True)
    del blob
    model.understanding_step = 3584
    model.eval()
    manifest, _ = manifest_and_plan()
    results = {}
    for split in SPLITS:
        result, per_gt, queries = evaluate_windows(model, opt, manifest[split], 448,
            split, reports, device, build_batch, official=True, panels=True)
        result['image_quality'] = image_quality_windows(model, opt, manifest[split], device)
        write_json(reports / f'eval_{split}_epoch64.json', dict(split=split, epoch=64,
            updates=448, exposures=3584, scopes=result,
            scope_mapping={'context':'official.all.context','target_all':'official.all.target',
                'true_novel':'official.novel.target'}, camera_poses='ground-truth posed setting'))
        write_json(reports / f'per_gt_epoch64_{split}.json', per_gt)
        write_json(reports / f'queries_epoch64_{split}.json', queries)
        results[split] = result
    aggregate = dict(arm=arm, checkpoint=str(endpoint), epoch=64, updates=448,
        exposures=3584, splits=results, scope_mapping={'context':'official.all.context',
        'target_all':'official.all.target','true_novel':'official.novel.target'},
        camera_poses='ground-truth posed setting', evaluator='fixed SIU3R protocol')
    write_json(reports / 'eval_epoch64.json', aggregate)
    return aggregate


def image_quality_windows(model, opt, windows, device):
    """Saved PSNR/SSIM/LPIPS by fixed scope and scene, from existing eval policy."""
    from scripts.eval_object_locus_v3_set import _run
    from scripts.object_locus_v3_set_runtime import capture_rng, restore_rng
    from tokengs.utils.metrics import MetricsCalculator
    metric = MetricsCalculator(device=device, lpips_net='vgg')
    scopes = ('context', 'target_all', 'novel')
    values = {s:{k:[] for k in ('psnr','ssim','lpips')} for s in scopes}
    scenes = {}
    prior_training = model.training
    rng = capture_rng()
    model.eval()
    try:
        with torch.no_grad():
            for window in windows:
                batch, out = _run(model, opt, window, build_batch, device)
                frame_ids = [int(v) for v in batch['frame_ids'][0].cpu().tolist()]
                indices = {'context':[0,1], 'target_all':list(range(len(frame_ids))),
                    'novel':[i for i,v in enumerate(frame_ids) if v in set(map(int,window['novel']))]}
                per_scene = scenes.setdefault(window['scene'], {s:{k:[] for k in ('psnr','ssim','lpips')} for s in scopes})
                for scope, idx in indices.items():
                    if not idx: continue
                    vals = metric.calculate_all_metrics(out['render']['images_pred'][:,idx],
                        batch['images_all'][:,idx], reduction='none')
                    for key, tensor in vals.items():
                        samples = tensor.detach().cpu().reshape(-1).tolist()
                        values[scope][key].extend(samples)
                        per_scene[scope][key].extend(samples)
    finally:
        restore_rng(rng)
        model.train(prior_training)
    avg = lambda rows: sum(rows)/len(rows) if rows else None
    return dict(aggregate={s:{k:avg(v) for k,v in x.items()} for s,x in values.items()},
        by_scene={scene:{s:{k:avg(v) for k,v in vals.items()} for s,vals in data.items()} for scene,data in scenes.items()})


def compare_endpoints():
    c = json.loads((REPORT_ROOT/'control/eval_epoch64.json').read_text())
    r = json.loads((REPORT_ROOT/'region_class/eval_epoch64.json').read_text())
    bootstrap_path = REPORT_ROOT/'paired_scene_bootstrap_val32.json'
    paired_scene_bootstrap_val32(
        REPORT_ROOT/'control/official/step_0448/val32/novel',
        REPORT_ROOT/'region_class/official/step_0448/val32/novel', bootstrap_path)
    bootstrap = json.loads(bootstrap_path.read_text())
    rows=[]
    def f(v): return None if v is None else float(v)
    for split in SPLITS:
        for scope in ('context','target_all','novel'):
            csplit=c['splits'][split];rsplit=r['splits'][split]
            cl=csplit['local'][scope];rl=rsplit['local'][scope]
            key='novel' if scope=='novel' else 'all'; view='context' if scope=='context' else 'target'
            co=csplit['official'][key];ro=rsplit['official'][key]
            def getoff(x,name): return x.get(view+'_map',{}).get(name) if name in ('map','map_50') else x.get(view+'_'+name)
            row={'split':split,'scope':scope,
                'official_mIoU_C':f(getoff(co,'miou')),'official_mIoU_R':f(getoff(ro,'miou')),
                'official_PQ_C':f(getoff(co,'pq')),'official_PQ_R':f(getoff(ro,'pq')),
                'official_mAP_C':f(getoff(co,'map')),'official_mAP_R':f(getoff(ro,'map')),
                'official_AP50_C':f(getoff(co,'map_50')),'official_AP50_R':f(getoff(ro,'map_50')),
                'candidate_mAP_C':f(cl.get('candidate_ap',{}).get('map')),'candidate_mAP_R':f(rl.get('candidate_ap',{}).get('map')),
                'candidate_AP50_C':f(cl.get('candidate_ap',{}).get('map_50')),'candidate_AP50_R':f(rl.get('candidate_ap',{}).get('map_50')),
                'raw_best_mask_coverage_C':f(cl.get('raw_best_iou_ge_0_5_fraction')),
                'raw_best_mask_coverage_R':f(rl.get('raw_best_iou_ge_0_5_fraction')),
                'matched_class_accuracy_C':f(cl.get('matched_19_class_accuracy')),
                'matched_class_accuracy_R':f(rl.get('matched_19_class_accuracy')),
                'PSNR_C':f(csplit['image_quality']['aggregate'][scope]['psnr']),
                'PSNR_R':f(rsplit['image_quality']['aggregate'][scope]['psnr']),
                'SSIM_C':f(csplit['image_quality']['aggregate'][scope]['ssim']),
                'SSIM_R':f(rsplit['image_quality']['aggregate'][scope]['ssim']),
                'LPIPS_C':f(csplit['image_quality']['aggregate'][scope]['lpips']),
                'LPIPS_R':f(rsplit['image_quality']['aggregate'][scope]['lpips'])}
            for family in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw'):
                for k in ('tp','fp','fn','precision','recall'):
                    row[family+'_'+k+'_C']=f(cl.get(family,{}).get(k));row[family+'_'+k+'_R']=f(rl.get(family,{}).get(k))
            rows.append(row)
    primary=next(x for x in rows if x['split']=='val32' and x['scope']=='novel')
    dm=primary['official_mAP_R']-primary['official_mAP_C']
    def delta(split,scope,metric):
        row=next(x for x in rows if x['split']==split and x['scope']==scope)
        return row[metric+'_R']-row[metric+'_C']
    gates={
        'val32_true_novel_official_packed_mAP_ci_lower_gt_zero':bootstrap['map_difference_ci95'][0]>0,
        'val32_true_novel_AP50_delta_ge_minus_0_01':delta('val32','novel','official_AP50')>=-0.01,
        'val32_true_novel_PQ_delta_ge_minus_0_01':delta('val32','novel','official_PQ')>=-0.01,
        'train_all56_context_AP50_delta_ge_minus_0_02':delta('train_all56','context','official_AP50')>=-0.02,
        'four_split_context_and_novel_PSNR_delta_ge_minus_0_5':all(delta(s,sc,'PSNR')>=-0.5 for s in SPLITS for sc in ('context','novel'))}
    report=dict(primary_metric='val32 true-novel official packed mAP',primary_delta=dm,
        paired_scene_bootstrap=bootstrap,rows=rows,scope_mapping={'context':'official.all.context',
        'target_all':'official.all.target','true_novel':'official.novel.target'},gates=gates,
        conclusion='预注册收益判据全部通过。' if all(gates.values()) else '未通过预注册收益判据；分类与packed指标分别报告。',
        camera_poses='ground-truth posed setting')
    write_json(REPORT_ROOT/'paired_endpoint_comparison.json',report)
    return report


def paired_scene_bootstrap_val32(c_root,r_root,output):
    """Use the pinned official segmentation path and recompute global AP by scene."""
    evaluator_repo=Path('/space/mawb/SIU3R')
    pinned='8ea80166be76854f938e90521f1a5b688b755c87'
    actual=subprocess.check_output(['git','-C',str(evaluator_repo),'rev-parse','HEAD'],text=True).strip()
    if actual != pinned:
        raise RuntimeError('official evaluator revision differs from the pinned protocol')
    code=r'''import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/space/mawb/SIU3R')
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
from torchmetrics.detection import MeanAveragePrecision
roots=[Path(sys.argv[1]),Path(sys.argv[2])];out=Path(sys.argv[3])
cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,
 eval_target_miou=False,eval_target_pq=False,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
 id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(roots[0]))
ev=Evaluator(cfg);ev.setup();names=sorted(p.name for p in roots[0].iterdir() if p.is_dir())
cache=[{},{}]
for arm,root in enumerate(roots):
 for name in names:
  pred=root/name/'target_seg_pred';gt=root/name/'target_seg_gt'
  if not pred.is_dir() or not gt.is_dir():raise FileNotFoundError(str(root/name))
  scene=name.split('_context')[0];data=ev.process_segmentation(pred,gt)
  cache[arm].setdefault(scene,[]).append((data['map_pred'],data['map_gt']))
if set(cache[0])!=set(cache[1]):raise RuntimeError('paired scene identities differ')
scenes=sorted(cache[0]);rng=np.random.default_rng(2026);draws=[]
for _ in range(2000):
 sampled=rng.choice(scenes,size=len(scenes),replace=True);vals=[]
 for arm in range(2):
  metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
  pairs=[item for scene in sampled for item in cache[arm][scene]]
  metric.update([p for p,g in pairs],[g for p,g in pairs]);x=metric.compute()
  vals.append((float(x['map']),float(x['map_50'])))
 draws.append([vals[1][i]-vals[0][i] for i in range(2)])
ci=np.percentile(np.asarray(draws),[2.5,97.5],axis=0)
out.write_text(json.dumps(dict(status='AVAILABLE',unit='val32 scene',seed=2026,resamples=2000,scenes=scenes,
 scope='true-novel official packed panoptic',map_difference_ci95=ci[:,0].tolist(),ap50_difference_ci95=ci[:,1].tolist(),
 differences=draws,method='paired scene resampling; recompute global MeanAveragePrecision per draw; R-C'),indent=2)+'\n')
'''
    subprocess.run(['/space/mawb/SIU3R/.venv_gpu_v4/bin/python','-c',code,
        str(c_root),str(r_root),str(output)],check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true',help='only after user notification that both arms finished')
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--arm',choices=('control','region_class','both'),default='both')
    args=parser.parse_args()
    if not args.run:
        raise SystemExit('Deferred: pass --run only after both fixed training arms complete and the user notifies.')
    if not torch.cuda.is_available():raise RuntimeError('fixed endpoint evaluation requires CUDA')
    device=torch.device('cuda:0')
    arms=('control','region_class') if args.arm=='both' else (args.arm,)
    for arm in arms:
        if (REPORT_ROOT/arm/'eval_epoch64.json').exists():
            raise RuntimeError(f'{arm} evaluation output already exists; refusing overwrite')
        evaluate_arm(arm,device)
    if args.compare:
        if set(arms)!={'control','region_class'}:raise ValueError('--compare requires --arm both')
        compare_endpoints()


if __name__=='__main__':main()
