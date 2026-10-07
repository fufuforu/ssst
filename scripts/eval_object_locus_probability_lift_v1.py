#!/usr/bin/env python3
"""Fixed Val32 paired C/P official evaluation; one GPU, one loaded checkpoint."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, socket, subprocess, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import object_locus_panoptic_v1_runtime as base
from scripts.object_locus_v3_set_runtime import build_batch, capture_rng, restore_rng
from scripts.eval_object_locus_v3_set import _run, _candidate_stats
from scripts.export_object_locus_v3_set_official import write_official_pair
from scripts.invoke_siu3r_official_evaluator import evaluate
from tokengs.models.object_locus_probability_lift_eval_v1 import LocusGSObjectLocusProbabilityLiftEvalV1

CKPT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
MANIFEST = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
REPORT = Path('/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1')
EXPECTED_CKPT = '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'
EXPECTED_SIU3R = '8ea80166be76854f938e90521f1a5b688b755c87'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def state_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        x=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()

def compare(a,b):
    return float((a.detach().float()-b.detach().float()).abs().max()) if a.numel() else 0.

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    global REPORT
    if args.smoke: REPORT=REPORT/'smoke'
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'logs').mkdir(exist_ok=True)
    if socket.gethostname()!='3dimage-17': raise RuntimeError('hardware contract requires 3dimage-17')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 4090': raise RuntimeError('logical cuda:0 must be RTX4090')
    if sha(CKPT)!=EXPECTED_CKPT: raise RuntimeError('checkpoint hash mismatch')
    if subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()!=EXPECTED_SIU3R: raise RuntimeError('SIU3R revision mismatch')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    model,opt=base.build_model(device,report=False)
    blob=torch.load(CKPT,map_location='cpu',weights_only=False,mmap=True)
    meta={k:blob.get(k) for k in ('epoch','completed_updates','completed_exposures')}
    if meta!={'epoch':6,'completed_updates':6258,'completed_exposures':50064}: raise RuntimeError(f'checkpoint metadata mismatch: {meta}')
    model.load_state_dict(blob['model'],strict=True); del blob
    model.__class__=LocusGSObjectLocusProbabilityLiftEvalV1
    model.understanding_step=50064;model.eval()
    pre=state_digest(model)
    manifest=json.loads(MANIFEST.read_text());wins=manifest['monitor_splits']['val32'];dev=manifest['monitor_splits']['dev8']
    if len(wins)!=32 or len({w['scene'] for w in wins})!=32 or len(dev)!=8 or not {w['scene'] for w in dev}<={w['scene'] for w in wins}: raise RuntimeError('registered cohort structure mismatch')
    train=set(manifest['actual_train_scenes']);
    if any(w['scene'] in train for w in wins): raise RuntimeError('Val32 scene overlaps actual train scenes')
    devsc={w['scene'] for w in dev}; mainwins=[w for w in wins if w['scene'] not in devsc]
    if len(mainwins)!=24: raise RuntimeError('main cohort is not 24 windows')
    spec={'checkpoint_sha256':EXPECTED_CKPT,'checkpoint_metadata':meta,'siu3r_commit':EXPECTED_SIU3R,
          'manifest_sha256':sha(MANIFEST),'val32_count':len(wins),'dev8_count':len(dev),'main_count':len(mainwins),
          'val32_windows':wins,'main_windows':mainwins,'effective_exposure':50064,'feedback_beta':.1,
          'new_exposures':0,'optimizer_updates':0,'training_backward':0,'precision':'FP32','TF32':False,
          'host':socket.gethostname(),'gpu':torch.cuda.get_device_name(0),'job_id':os.environ.get('SLURM_JOB_ID')}
    (REPORT/'experiment_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (REPORT/'manifests.json').write_text(json.dumps({'manifest_sha256':spec['manifest_sha256'],
       'val32_all':wins,'val32_excluding_dev8_scenes':mainwins,'dev8':dev},indent=2)+'\n')
    (REPORT/'provenance.json').write_text(json.dumps({k:v for k,v in spec.items() if k not in ('val32_windows','main_windows')}
       | {'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'science_base':'7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3',
          'training_entry_base':'b2624d57ea9ad5a73fd6a8375bafbe4e4c263c9c'},indent=2)+'\n')
    cohorts={'val32_all':wins,'val32_excluding_dev8_scenes':mainwins}
    arms=('C','P'); scopes=('all','novel')
    for arm in arms:
      for cohort,cohortwins in cohorts.items():
        for scope in scopes:
          (REPORT/arm/cohort/'official'/scope).mkdir(parents=True,exist_ok=True)
    per=[]; pergt=[]; perquery=[]; maxdiff=[]
    from torchmetrics.detection import MeanAveragePrecision
    local_ap={(arm,scope):MeanAveragePrecision(iou_type='segm',sync_on_compute=False).to(device)
              for arm in ('C','P') for scope in ('context','target-all','novel')}
    selected=wins[:2] if args.smoke else wins
    for ix,w in enumerate(selected):
      batch=build_batch(opt,w,device)
      rng=capture_rng()
      original_out=None
      if args.smoke and ix==0:
        model.__class__=base.LocusGSObjectLocusPanopticV1Recon
        restore_rng(rng)
        with torch.autocast(device_type='cuda',enabled=False), torch.no_grad(): _,original_out=_run(model,opt,w,lambda *_:batch,device)
        model.__class__=LocusGSObjectLocusProbabilityLiftEvalV1
      model.readout_mode='C'
      with torch.autocast(device_type='cuda',enabled=False), torch.no_grad(): bC,outC=_run(model,opt,w,lambda *_:batch,device)
      if original_out is not None:
        parity={k:compare(original_out[k],outC[k]) for k in ('gaussians','alpha','lifting_mass','lifting_gate','p_class')}
        parity['rgb']=compare(original_out['render']['images_pred'],outC['render']['images_pred'])
        for key in ('q','c','s'): parity[key]=compare(original_out['states'][-1][key],outC['states'][-1][key])
        if any(v>1e-6 for v in parity.values()): raise RuntimeError(f'original C/new C numerical mismatch {parity}')
        from scripts.export_object_locus_v3_set_official import assemble_panoptic
        sa,ia,_=assemble_panoptic(original_out);sb,ib,_=assemble_panoptic(outC)
        if not torch.equal(sa,sb) or not torch.equal(ia,ib): raise RuntimeError('original C/new C packed PNG parity mismatch')
        del original_out
      rngC=capture_rng(); restore_rng(rng); model.readout_mode='P'
      with torch.autocast(device_type='cuda',enabled=False), torch.no_grad(): bP,outP=_run(model,opt,w,lambda *_:batch,device)
      if model.understanding_step!=50064 or abs(float(outP['beta'])-.1)>1e-8: raise RuntimeError('exposure/beta mismatch')
      required=('assignment','gaussian_membership','gaussian_mask_logits','membership_mass','region_mass','pixel_membership',
                'p_class','class_logits19','semantic_scores','alpha','pixel_void_mass','anchor_membership','lifting_mass','lifting_gate')
      if any(k not in outP for k in required) or outP['gaussian_feature'] is not None or outP['readout_domain']!='probability':
        raise RuntimeError('P output contract mismatch')
      if not torch.isfinite(outP['gaussian_mask_logits']).all() or not torch.isfinite(outP['region_mass']).all(): raise FloatingPointError('nonfinite P output')
      diff={k:compare(outC[k],outP[k]) for k in ('gaussians','alpha','lifting_mass','lifting_gate','p_class')}
      for key in ('q','c','s'):
        diff[key]=compare(outC['states'][-1][key],outP['states'][-1][key])
      diff['rgb']=compare(outC['render']['images_pred'],outP['render']['images_pred'])
      maxdiff.append({'scene':w['scene'],'diff':diff})
      if any(v>1e-6 for v in diff.values()): raise RuntimeError(f'C/P invariant parity failed: {w["scene"]}: {diff}')
      stat={}
      for branch,out in (('C',outC),('P',outP)):
        for scope,ids in (('context',[0,1]),('target-all',list(range(len(batch['frame_ids'][0])))),('novel',[i for i,f in enumerate(batch['frame_ids'][0].tolist()) if f in set(w['novel'])])):
          row=_candidate_stats(out,batch,ids)
          payload=row.pop('_map_payload')
          for key in ('pred','target'):
            masks=payload[key]['masks']
            payload[key]['masks']=(masks.reshape(masks.shape[0],-1,masks.shape[-1]) if masks.shape[0]
                                   else torch.zeros((0,len(ids)*256,256),dtype=torch.bool,device=device))
          local_ap[(branch,scope)].update([payload['pred']],[payload['target']])
          stat[(branch,scope)]=row
        for cohort,cwins in cohorts.items():
          if w['scene'] not in {x['scene'] for x in cwins}: continue
          for exp_scope,frames in (('all','all'),('novel','novel')):
            root=REPORT/branch/cohort/'official'/exp_scope
            write_official_pair(out,batch,w,root,target_frames=frames)
        # free each branch output after writing CPU diagnostics below
      for scope in ('context','target-all','novel'):
        a=stat[('C',scope)];b=stat[('P',scope)]
        frameids=batch['frame_ids'][0].tolist()
        ixscope={'context':[0,1],'target-all':list(range(len(frameids))),
                 'novel':[i for i,f in enumerate(frameids) if f in set(w['novel'])]}[scope]
        psnr={}
        for branch,out in (('C',outC),('P',outP)):
          mse=(out['render']['images_pred'][0,ixscope]-batch['images_all'][0,ixscope]).square().mean().clamp_min(1e-12)
          psnr[branch]=float((-10*torch.log10(mse)).cpu())
        per.append({'scene':w['scene'],'scope':scope,'C_candidate_count':a['candidate_count'],'P_candidate_count':b['candidate_count'],
                    'C_eligible_queries':sum(x['joint_eligible'] for x in a['query_rows']),'P_eligible_queries':sum(x['joint_eligible'] for x in b['query_rows']),
                    'C_nonempty_candidates':a['candidate_count'],'P_nonempty_candidates':b['candidate_count'],
                    'C_packed_instances':a['panoptic_ca']['tp']+a['panoptic_ca']['fp'],'P_packed_instances':b['panoptic_ca']['tp']+b['panoptic_ca']['fp'],
                    'C_psnr':psnr['C'],'P_psnr':psnr['P'],'P_minus_C_psnr':psnr['P']-psnr['C'],
                    'C_candidate_ca':a['candidate_ca'],'P_candidate_ca':b['candidate_ca'],
                    'C_panoptic_ca':a['panoptic_ca'],'P_panoptic_ca':b['panoptic_ca'],
                    'C_panoptic_cw':a['panoptic_cw'],'P_panoptic_cw':b['panoptic_cw'],
                    'C_raw_best_iou_ge_0_5':sum(x>=.5 for x in a['raw_best_ious']),'P_raw_best_iou_ge_0_5':sum(x>=.5 for x in b['raw_best_ious']),
                    'C_raw_best_iou_ge_0_75':sum(x>=.75 for x in a['raw_best_ious']),'P_raw_best_iou_ge_0_75':sum(x>=.75 for x in b['raw_best_ious'])})
        for arm in arms:
          for scope in ('context','target-all','novel'):
            row=stat[(arm,scope)]
            for x in row['per_gt']:pergt.append({'arm':arm,'scene':w['scene'],'scope':scope,**{k:v for k,v in x.items() if isinstance(v,(str,int,float,bool)) or v is None}})
            for x in row['query_rows']:perquery.append({'arm':arm,'scene':w['scene'],'scope':scope,**x})
      restore_rng(rngC)
      del batch,outC,outP
      torch.cuda.empty_cache()
      print(f'window {ix+1}/{len(selected)} {w["scene"]} complete',flush=True)
    for cohort in cohorts:
      for arm in arms:
        for scope in scopes:
          root=REPORT/arm/cohort/'official'/scope
          result=evaluate(root,device='cuda',segmentation=True,image_quality=False,depth_quality=False)
          (REPORT/arm/cohort/f'official_{scope}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    def csvwrite(path,rows):
      with path.open('w',newline='') as f:
        if rows:
          wr=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));wr.writeheader();wr.writerows(rows)
    csvwrite(REPORT/'per_window.csv',per);csvwrite(REPORT/'per_gt.csv',pergt);csvwrite(REPORT/'per_query.csv',perquery)
    candidate_ap={}
    for (arm,scope),metric in local_ap.items():
      r=metric.compute();candidate_ap[f'{arm}:{scope}']={'map':float(r['map']),'map_50':float(r['map_50']),
       'definition':'local candidate masks before panoptic packing; not official AP'}
    (REPORT/'local_candidate_ap.json').write_text(json.dumps(candidate_ap,indent=2)+'\n')
    (REPORT/'parity.json').write_text(json.dumps({'windows':maxdiff,'state_dict_before':pre,'state_dict_after':state_digest(model),'unchanged':pre==state_digest(model)},indent=2)+'\n')
    if pre!=state_digest(model) or any(p.grad is not None for p in model.parameters()): raise RuntimeError('state/grad integrity failure')
    if args.smoke:
      (REPORT/'complete.json').write_text(json.dumps({'status':'SMOKE_ONLY','windows':len(selected),'job_id':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')
      (REPORT.parent/'smoke.json').write_text(json.dumps({'status':'PASS','windows':len(selected),'job_id':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')
    else:
      subprocess.run([sys.executable,str(REPO/'scripts/summarize_object_locus_probability_lift_v1.py')],check=True)
      (REPORT/'complete.json').write_text(json.dumps({'status':'COMPLETE','windows':len(selected),'job_id':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')

if __name__=='__main__': main()
