"""Fixed-window endpoint evaluator. Never launched by the training jobs."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
from scripts.object_locus_gc_sweep_runtime import *

def image_depth_metrics(model,opt,manifest,device):
    from scripts.eval_object_locus_v3_set import _run
    from tokengs.siu3r_protocol import TorchMetricBackend,depth_metrics
    backend=TorchMetricBackend(device=device,image_metrics=True,segmentation_metrics=False)
    scopes=('context','target-all','true-novel')
    accum={s:{scope:{k:[] for k in ('psnr','ssim','lpips','absrel','rmse')} for scope in scopes} for s in SPLITS}
    was=model.training;model.eval()
    try:
      with torch.no_grad():
       for split in SPLITS:
        for win in manifest[split]:
         batch,out=_run(model,opt,win,build_batch,device)
         frame_ids=batch['frame_ids'][0].detach().cpu().tolist()
         ids={'context':[0,1],'target-all':list(range(len(frame_ids))),'true-novel':[i for i,f in enumerate(frame_ids) if int(f) in set(map(int,win['novel']))]}
         pred=out['render']['images_pred'];truth=batch['images_all'];depth=out['render']['depths_pred'];depth=depth[:,:,0] if depth.ndim==5 else depth
         gt=batch['depth_gt_m_all'][:,:,0];valid=batch['depth_gt_valid_all'][:,:,0].bool()
         for scope,views in ids.items():
          for v in views:
           im=backend.reconstruction_one(pred[0,v:v+1],truth[0,v:v+1])
           for metric,value in im.items():accum[split][scope][metric].append(value)
           backend.psnr.reset();backend.ssim.reset();backend.lpips.reset()
           p=depth[0,v].detach().cpu().numpy();g=gt[0,v].detach().cpu().numpy();m=valid[0,v].detach().cpu().numpy()&(g>0)
           if m.any():
            d=depth_metrics(p,np.where(m,g,0));accum[split][scope]['absrel'].append(d['absrel']);accum[split][scope]['rmse'].append(d['rmse'])
         del batch,out
    finally:model.train(was)
    return {s:{scope:{k:float(np.mean(v)) if v else None for k,v in vals.items()} for scope,vals in scopeset.items()} for s,scopeset in accum.items()}

def paired_scene_bootstrap():
    """Official packed global mAP with one shared saved scene-index matrix."""
    sys.path.insert(0,'/space/mawb/SIU3R')
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    from torchmetrics.detection import MeanAveragePrecision
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,eval_target_miou=False,eval_target_pq=False,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(REPORT_ROOT/'gc001/official/step_1008/val32/novel'))
    evaluator=Evaluator(cfg);evaluator.setup();base=Path(cfg.eval_path);dirs=sorted(p.name for p in base.iterdir() if p.is_dir())
    if len(dirs)!=32:raise RuntimeError(f'expected 32 val32 exports, got {len(dirs)}')
    scenes=sorted({name.split('_context')[0] for name in dirs})
    if len(scenes)!=32:raise RuntimeError('val32 scene identities are not unique')
    caches={}
    for arm in ARMS:
      root=REPORT_ROOT/arm/'official/step_1008/val32/novel';arm_dirs=sorted(p.name for p in root.iterdir() if p.is_dir())
      if arm_dirs!=dirs:raise RuntimeError(f'{arm} exported window identities differ')
      rows={scene:[] for scene in scenes}
      for name in dirs:
       data=evaluator.process_segmentation(root/name/'target_seg_pred',root/name/'target_seg_gt');rows[name.split('_context')[0]].append((data['map_pred'],data['map_gt']))
      caches[arm]=rows
    indices=np.random.default_rng(2026).choice(len(scenes),size=(2000,len(scenes)),replace=True)
    index_path=REPORT_ROOT/'paired_scene_bootstrap_indices_seed2026.npy';np.save(index_path,indices);reports={}
    for arm in ('gc010','gc100'):
      differences=[]
      for sampled in indices:
       values=[]
       for candidate in (arm,'gc001'):
        metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
        pairs=[pair for scene_i in sampled for pair in caches[candidate][scenes[int(scene_i)]]]
        metric.update([pred for pred,_ in pairs],[gt for _,gt in pairs]);result=metric.compute();values.append((float(result['map']),float(result['map_50'])))
       differences.append([values[0][0]-values[1][0],values[0][1]-values[1][1]])
      ci=np.percentile(np.asarray(differences),[2.5,97.5],axis=0)
      reports[arm]={'against':'gc001','resamples':2000,'seed':2026,'mAP_difference_ci95':ci[:,0].tolist(),'AP50_difference_ci95':ci[:,1].tolist(),'differences':differences}
    write_json(REPORT_ROOT/'paired_scene_bootstrap.json',{'status':'AVAILABLE','unit':'val32 scene','official_metric':'global packed panoptic mAP from official process_segmentation','shared_scene_indices_file':str(index_path),'shared_scene_indices_sha256':sha256(index_path),'scene_names':scenes,'multiple_comparison_note':'Exploratory intervals; no multiplicity-adjusted significance claim.','comparisons':reports})

def comparison_report():
    evaluations={arm:json.loads((REPORT_ROOT/arm/'eval_epoch8.json').read_text()) for arm in ARMS}
    bootstrap=json.loads((REPORT_ROOT/'paired_scene_bootstrap.json').read_text())
    reference=evaluations['gc001'];comparisons={}
    for arm in ('gc010','gc100'):
        current=evaluations[arm];vnew=current['splits']['val32']['official']['novel'];vbase=reference['splits']['val32']['official']['novel']
        nmap=vnew.get('target_map',{});bmap=vbase.get('target_map',{});ci=bootstrap['comparisons'][arm]['mAP_difference_ci95']
        checks={'val32_true_novel_mAP_CI_lower_gt_0':ci[0]>0,
            'val32_true_novel_AP50_drop_le_0.01':float(nmap.get('map_50',float('nan')))-float(bmap.get('map_50',float('nan')))>=-.01,
            'val32_true_novel_PQ_drop_le_0.01':float(vnew.get('target_pq',float('nan')))-float(vbase.get('target_pq',float('nan')))>=-.01}
        psnr_rows={}
        for split in SPLITS:
            for scope in ('context','true-novel'):
                a=current['per_image_reconstruction_and_depth'][split][scope]['psnr'];b=reference['per_image_reconstruction_and_depth'][split][scope]['psnr']
                key=f'{split}_{scope}_PSNR_drop_le_0.5dB';checks[key]=(a-b)>=-.5;psnr_rows[key]={'gc001':b,arm:a,'difference':a-b}
        an=current['per_image_reconstruction_and_depth']['val32']['true-novel']['absrel'];bn=reference['per_image_reconstruction_and_depth']['val32']['true-novel']['absrel']
        checks['val32_true_novel_AbsRel_relative_increase_le_5pct']=an<=bn*1.05
        comparisons[arm]={'gc001':{'val32_true_novel_mAP':bmap.get('map'),'AP50':bmap.get('map_50'),'PQ':vbase.get('target_pq'),'AbsRel':bn},'candidate':{'val32_true_novel_mAP':nmap.get('map'),'AP50':nmap.get('map_50'),'PQ':vnew.get('target_pq'),'AbsRel':an},'mAP_difference_CI95':ci,'psnr_by_split_scope':psnr_rows,'criteria':checks,'all_criteria_met':all(checks.values())}
    write_json(REPORT_ROOT/'comparison_criteria.json',{'comparisons':comparisons,'interpretation':'Registered endpoint comparison only. Bootstrap is exploratory and is not multiplicity adjusted. No automatic model selection or follow-on training.'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epoch',type=int,choices=(4,8),required=True);args=ap.parse_args()
    device=init_distributed()
    if rank_world()[1]!=1:raise RuntimeError('fixed evaluator runs on one GPU')
    manifest,plan,plan_sha=ensure_shared_plan();completed=args.epoch*126
    for arm,alpha in ARMS.items():
        blob=torch.load(RUN_ROOT/arm/f'checkpoint_epoch{args.epoch}.pt',map_location='cpu',weights_only=False,mmap=True)
        if blob['alpha']!=alpha or blob['completed_updates']!=completed or blob['plan_sha256']!=plan_sha:raise RuntimeError('endpoint checkpoint provenance mismatch')
        model,opt,source=build_model(device);model.load_state_dict(blob['model'],strict=True);model.eval();del blob
        from scripts.eval_object_locus_panoptic_v1 import evaluate_windows
        results={}
        for split in SPLITS:
            result,_,_=evaluate_windows(model,opt,manifest[split],completed,split,REPORT_ROOT/arm,device,build_batch,official=True,panels=False)
            results[split]=result
        extras=image_depth_metrics(model,opt,manifest,device)
        write_json(REPORT_ROOT/arm/f'eval_epoch{args.epoch}.json',{'arm':arm,'alpha':alpha,'epoch':args.epoch,'updates':completed,'new_exposures':completed*8,'model_exposure':SOURCE_EXPOSURES+completed*8,'scope_mapping':{'context':'all/context','target-all':'all/target','true-novel':'novel/target'},'splits':results,'per_image_reconstruction_and_depth':extras,'source_checkpoint_sha256':SOURCE_SHA256,'plan_sha256':plan_sha})
        del model,opt
    if args.epoch==8:paired_scene_bootstrap();comparison_report()

if __name__=='__main__':main()
