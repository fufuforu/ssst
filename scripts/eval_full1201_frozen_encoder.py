"""Read-only Full1201 frozen-encoder evaluation/export worker.

Independent window shards (no DDP) export official packed predictions and
RGB/depth from one model forward per window. Aggregation runs pinned SIU3R.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts import object_locus_panoptic_full1201_frozen_encoder_runtime as frozen
from scripts import object_locus_panoptic_v1_runtime as trainable
from scripts.object_locus_v3_set_runtime import build_batch
from scripts.export_object_locus_v3_set_official import write_official_pair
from scripts.eval_object_locus_v3_set import _candidate_stats
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
BASE_REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
BASE_RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu')
INIT_DIGEST='22f5a27a651ecff262840b238ad28c478410b44b648645b042eb96f33711b8d3'
SIU3R=Path('/space/mawb/SIU3R')
SIU3R_COMMIT='8ea80166be76854f938e90521f1a5b688b755c87'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def normalize_window(w):
    if 'scene' in w:
        context=list(map(int,w['context']))
        target=list(map(int,w.get('target',context+w.get('novel',[]))))
        novel=list(map(int,w.get('novel',[x for x in target if x not in context])))
        return dict(w,scene=str(w['scene']),context=context,target=target,novel=novel)
    return dict(scene=str(w['scan']),context=list(map(int,w['context_ids'])),
      target=list(map(int,w['target_ids'])),novel=[int(x) for x in w['target_ids'] if x not in w['context_ids']])

def read_full():
    p=BASE_REPORT/'delivery_epoch6/full_validation_manifest.json'
    if not p.exists(): p=BASE_REPORT/'full_validation_manifest.json'
    x=json.loads(p.read_text())
    return [normalize_window(w) for w in x],p

def checkpoint_path(frozen_model,epoch):
    return (RUN if frozen_model else BASE_RUN)/f'checkpoint_epoch_{epoch:02}.pt'

def load_model(frozen_model,epoch,device):
    runtime=frozen if frozen_model else trainable
    if frozen_model: runtime.configure()
    model,opt=runtime.build_model(device,report=False)
    p=checkpoint_path(frozen_model,epoch)
    blob=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    exp=int(blob.get('completed_exposures',-1)); updates=int(blob.get('completed_updates',-1))
    if exp!=updates*8: raise RuntimeError(f'checkpoint exposure/update mismatch: {exp}/{updates}')
    if updates!=epoch*1043: raise RuntimeError(f'epoch/update mismatch: {epoch}/{updates}')
    manifest_path=(REPORT/'run_manifest.json') if frozen_model else (BASE_REPORT/'run_manifest.json')
    expected_sha=json.loads(manifest_path.read_text())['git_sha']
    if blob.get('git_sha')!=expected_sha: raise RuntimeError('checkpoint training SHA mismatch')
    model.load_state_dict(blob['model'],strict=True)
    if frozen_model:
        if int(os.environ.get('LOCAL_RANK',os.environ.get('SLURM_PROCID','0')))==0:
            digest=frozen.frozen_state_digest(model)
            if digest!=INIT_DIGEST: raise RuntimeError('encoder digest differs from initialized state')
            model._eval_encoder_digest=digest
        if model.understanding.encoder.training: raise RuntimeError('frozen encoder is in train mode')
    model.eval()
    if model.training: raise RuntimeError('model.eval() did not take effect')
    return model,opt,exp,p,blob

def make_visualizer():
    sys.path.insert(0,str(SIU3R))
    # Import the pinned Visualizer implementation without pulling Hydra's
    # unrelated training configuration stack into the inference environment.
    import types
    from importlib.machinery import ModuleSpec
    from dataclasses import dataclass
    cfgmod=types.ModuleType('src.config')
    @dataclass
    class VisualizerCfg:
        log_colored_depth: bool=False
        log_rendered_video: bool=False
        log_gaussian_ply: bool=False
        save_sh_dc_only: bool=False
        dataset_name: str='scannet'
        overlay_mask_alpha: float=.5
        write_to: str=''
    cfgmod.VisualizerCfg=VisualizerCfg
    sys.modules['src.config']=cfgmod
    # The official visualizer imports kornia unconditionally, but its depth
    # color-map branch is disabled; the uint-mm export path does not use it.
    if 'kornia' not in sys.modules:
        mod=types.ModuleType('kornia');mod.__spec__=ModuleSpec('kornia',loader=None);sys.modules['kornia']=mod
    if 'cv2' not in sys.modules:
        mod=types.ModuleType('cv2');mod.__spec__=ModuleSpec('cv2',loader=None);mod.__version__='4.8.0';sys.modules['cv2']=mod
    from src.visualizer import Visualizer
    return Visualizer(VisualizerCfg())

def scope_local(out,batch,idx):
    row=_candidate_stats(out,batch,idx)
    payload=row.pop('_map_payload')
    for branch in ('pred','target'):
        m=payload[branch]['masks']
        payload[branch]['masks']=m.reshape(m.shape[0],-1,m.shape[-1]) if m.shape[0] else torch.zeros((0,len(idx)*256,256),dtype=torch.bool,device=m.device)
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric=MeanAveragePrecision(iou_type='segm',sync_on_compute=False)
        metric.update([{k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in payload['pred'].items()}],
                      [{k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in payload['target'].items()}])
        ap=metric.compute()
        row['candidate_ap']={k:(float(ap[src]) if np.isfinite(float(ap[src])) else None)
          for k,src in (('map','map'),('map_50','map_50'))}
    except Exception as e: row['candidate_ap']={'error':f'{type(e).__name__}: {e}'}
    for k in ('panoptic_semantic','panoptic_instance','raw_masks','gt_semantic','gt_instance','predictions','query_rows','per_gt'):
        row.pop(k,None)
    return row

def write_qualitative(batch,out,window,path,title):
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    sem,ins,_=assemble_panoptic(out)
    tile=192;headers=['input RGB','rendered RGB','GT depth (0-6m display)','pred depth (0-6m display)','GT instances','pred instances']
    canvas=Image.new('RGB',(tile*len(headers),36+tile*2),'white')
    from PIL import ImageDraw
    draw=ImageDraw.Draw(canvas);draw.text((4,4),title,fill='black')
    for c,label in enumerate(headers):draw.text((c*tile+4,20),label,fill='black')
    for row,view in enumerate((0,1)):
        gtsem=batch['semantic_label_all'][0,view].detach().cpu().numpy()
        gtins=batch['instance_label_all'][0,view].detach().cpu().numpy()
        psem=sem[view].detach().cpu().numpy();pins=ins[view].detach().cpu().numpy()
        gt_rgb=(batch['images_all'][0,view].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        pr_rgb=(out['render']['images_pred'][0,view].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        def depth_rgb(x):
            a=x.detach().cpu().numpy()
            z=np.clip(a/6.,0,1)
            return np.stack((z*255,(1-np.abs(z-.5)*2)*255,(1-z)*255),-1).astype(np.uint8)
        def inst_rgb(s,i):
            z=np.zeros((*s.shape,3),np.uint8)
            for cls in range(20):z[s==cls]=((37*cls+53)%255,(97*cls+31)%255,(173*cls+71)%255)
            for iid in np.unique(i):
                if iid>0:z[i==iid]=((31*int(iid)+71)%255,(67*int(iid)+137)%255,(131*int(iid)+29)%255)
            return z
        gt_depth=batch['depth_gt_m_all'][0,view,0]
        pred_depth=out['render']['depths_pred'][0,view,0]/.15
        imgs=[gt_rgb,pr_rgb,depth_rgb(gt_depth),depth_rgb(pred_depth),inst_rgb(gtsem,gtins),inst_rgb(psem,pins)]
        for col,img in enumerate(imgs):canvas.paste(Image.fromarray(img).resize((tile,tile)),(col*tile,36+row*tile))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def export_rgb_depth(visualizer,batch,out,window,all_root,epoch,split,rank,qual_dir=None,wi=0):
    scene=window['scene'];ctx=window['context'];frames=[int(x) for x in batch['frame_ids'][0].cpu().tolist()]
    pred=out['render']['depths_pred']
    if tuple(pred.shape)!=(1,len(frames),1,*batch['images_all'].shape[-2:]):
        raise RuntimeError(f'depth render shape mismatch {tuple(pred.shape)}')
    gt_m=batch['depth_gt_m_all'];gt_scene=batch['depth_gt_scene_all'];valid=batch['depth_gt_valid_all']
    if gt_m.shape!=pred.shape or gt_scene.shape!=pred.shape or valid.shape!=pred.shape:
        raise RuntimeError(f'GT/render depth shape mismatch: pred={pred.shape} gt_m={gt_m.shape} gt_scene={gt_scene.shape} valid={valid.shape}')
    if not torch.allclose(gt_scene,gt_m*.15,atol=1e-6,rtol=1e-6): raise RuntimeError('provider scene-depth scale contract mismatch')
    if not torch.equal(valid.bool(),torch.isfinite(gt_m)&(gt_m>0)): raise RuntimeError('provider GT depth validity mask mismatch')
    pred_m=pred/.15
    if batch['images_all'].shape[1]!=len(frames): raise RuntimeError('RGB/frame count mismatch')
    d=all_root/(f'{scene}_context'+'_'.join(map(str,ctx))); d.mkdir(parents=True,exist_ok=True)
    good=[];bad=[]
    for i,f in enumerate(frames):
        ok=bool(torch.isfinite(pred_m[0,i]).all()) and bool(torch.isfinite(gt_m[0,i]).all()) and bool((gt_m[0,i]>0).any())
        if ok: good.append(i)
        else: bad.append({'scene':scene,'context':ctx,'frame':f,'reason':'nonfinite_prediction_or_GT_or_no_positive_GT'})
    if good:
        ids=[frames[i] for i in good]
        visualizer.visualize_recon_image(out['render']['images_pred'][:,good],pred_m[:,good,0],
          batch['images_all'][:,good],gt_m[:,good,0],str(all_root),[scene],[ctx],[ids])
    (d/'rgb').mkdir(exist_ok=True);(d/'rgb_gt').mkdir(exist_ok=True)
    for i,f in enumerate(frames):
        name=f'{scene}_{f}.png'
        if not (d/'rgb'/name).exists():
            a=(out['render']['images_pred'][0,i].detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            Image.fromarray(a).save(d/'rgb'/name)
        if not (d/'rgb_gt'/name).exists():
            a=(batch['images_all'][0,i].detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            Image.fromarray(a).save(d/'rgb_gt'/name)
    if qual_dir is not None:
        write_qualitative(batch,out,window,qual_dir/f'{wi:02d}_{scene}_context_{"_".join(map(str,ctx))}.png',f'{split} epoch {epoch}')
    return bad

def forward_one(model,opt,window,exposure,device):
    batch=build_batch(opt,window,device);mi,_=split_data(batch,opt)
    dec=ModelInputDecoder(cam_view=batch['cam_view_all'],intrinsics=batch['intrinsics_all'])
    with torch.no_grad():
        out=model.forward_object_locus(ModelInput(mi.encoder,dec),render_decoder_input=dec,
            context_decoder=dec,step=exposure)
    if model.training or (hasattr(model,'understanding') and model.understanding.encoder.training):
        raise RuntimeError('eval mode was lost')
    return batch,out

@torch.no_grad()
def run_window(model,opt,window,exposure,device,all_root,novel_root,visualizer,epoch,split,rank,qual_dir=None,wi=0):
    batch,out=forward_one(model,opt,window,exposure,device)
    ra=write_official_pair(out,batch,window,all_root,target_frames='all')
    rn=write_official_pair(out,batch,window,novel_root,target_frames='novel')
    if ra['frame_ids']!=rn['frame_ids']: raise RuntimeError('official exports disagree on frame identities')
    failed=export_rgb_depth(visualizer,batch,out,window,all_root,epoch,split,rank,qual_dir,wi)
    ids=[int(x) for x in batch['frame_ids'][0].cpu().tolist()]
    idxs={'context':[0,1],'target_all':list(range(len(ids))),
      'novel':[i for i,x in enumerate(ids) if x in set(window['novel'])]}
    diagnostics={s:scope_local(out,batch,ix) for s,ix in idxs.items()}
    return {'scene':window['scene'],'context':window['context'],'target':window['target'],'novel':window['novel'],
      'frame_ids':ids,'exposure':int(exposure),'missing_depth':failed,'diagnostics':diagnostics,
      'all_export':ra,'novel_export':rn}

def load_manifest(split):
    m=json.loads((REPORT/'manifest.json').read_text())
    if split in ('dev8','val32'): return [normalize_window(w) for w in m['monitor_splits'][split]]
    if split=='full': return read_full()[0]
    raise ValueError(split)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['dev8','best','baseline_depth','smoke'],required=True)
    ap.add_argument('--split',choices=['dev8','val32','full'],default=None)
    ap.add_argument('--epoch',type=int,default=None);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--reuse-dev8',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    import subprocess
    if subprocess.check_output(['git','-C',str(SIU3R),'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT:
        raise RuntimeError('pinned SIU3R commit mismatch')
    eval_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if a.profile=='dev8': epochs=[1,2,4,6,8];frozen_model=True;splits=['dev8']
    elif a.profile=='smoke': epochs=[a.epoch or 1];frozen_model=True;splits=[a.split or 'dev8']
    elif a.profile=='best':
        selected=a.epoch
        if selected in (None,0): selected=int(json.loads((REPORT/'evaluation/dev8_selection.json').read_text())['selected_epoch'])
        epochs=[selected];frozen_model=True;splits=['val32','full']
    else: epochs=[6];frozen_model=False;splits=['full']
    rank=int(os.environ.get('LOCAL_RANK',os.environ.get('SLURM_PROCID','0')))
    world=int(os.environ.get('WORLD_SIZE','1'))
    if torch.cuda.is_available(): torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    else: device=torch.device('cpu')
    visualizer=make_visualizer();work=[]
    qualitative_keys={(w['scene'],tuple(w['context'])) for w in load_manifest('val32')[:2]}
    if a.profile in ('dev8','smoke') and rank==0:
        model0,opt0,exp0,ckpt0,blob0=load_model(True,0,device)
        record0={'status':'PASS','epoch':0,'checkpoint':str(ckpt0),'checkpoint_sha256':sha(ckpt0),
          'completed_updates':int(blob0['completed_updates']),'completed_exposures':exp0,
          'frozen_encoder_digest':getattr(model0,'_eval_encoder_digest',None),'strict_load':True,
          'forward_count':0,'optimizer_updates':0}
        out0=a.output.resolve()/'epoch_00_verification.json';out0.parent.mkdir(parents=True,exist_ok=True)
        out0.write_text(json.dumps(record0,indent=2)+'\n')
        del model0,opt0,blob0
    for epoch in epochs:
        model,opt,exposure,ckpt,blob=load_model(frozen_model,epoch,device)
        for split in splits:
            windows=load_manifest(split)
            if a.limit: windows=windows[:a.limit]
            reused=set()
            if a.reuse_dev8 and split!='dev8':
                reused={(w['scene'],tuple(w['context'])) for w in load_manifest('dev8')}
                if split=='full': reused|={(w['scene'],tuple(w['context'])) for w in load_manifest('val32')}
            ids=[str(w['scene'])+'_context_'+'_'.join(map(str,w['context'])) for w in windows]
            if len(ids)!=len(set(ids)): raise RuntimeError(f'duplicate window identity in {split}')
            indexed=list(enumerate(windows))
            shard=[(j,w) for j,w in indexed if
                   (world==1 or j%world==rank) and
                   (w['scene'],tuple(w['context'])) not in reused]
            root=a.output.resolve()/f'epoch_{epoch:02}'/split/f'rank{rank:02}'
            all_root=root/'all';novel_root=root/'novel';all_root.mkdir(parents=True,exist_ok=True);novel_root.mkdir(parents=True,exist_ok=True)
            rec=[];failed=[];t0=time.perf_counter()
            for wi,win in shard:
                if a.profile=='baseline_depth':
                    batch,out=forward_one(model,opt,win,exposure,device)
                    failed_depth=export_rgb_depth(visualizer,batch,out,win,all_root,epoch,split,rank)
                    row={'scene':win['scene'],'context':win['context'],'target':win['target'],'novel':win['novel'],
                      'frame_ids':[int(x) for x in batch['frame_ids'][0].cpu().tolist()],
                      'exposure':int(exposure),'missing_depth':failed_depth}
                else:
                    qualitative_dir=(a.output.resolve()/'qualitative'/f'epoch_{epoch:02}'/'val32'
                        if (win['scene'],tuple(win['context'])) in qualitative_keys else None)
                    row=run_window(model,opt,win,exposure,device,all_root,novel_root,visualizer,epoch,split,rank,
                      qualitative_dir,wi)
                rec.append(row);failed.extend(row['missing_depth'])
            (root/'window_records.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
            (root/'failures.json').write_text(json.dumps(failed,indent=2)+'\n')
            done={'status':'PASS','rank':rank,'epoch':epoch,'split':split,'windows':len(rec),'manifest_windows':len(windows),
              'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt) if rank==0 else None,'completed_updates':int(blob['completed_updates']),
              'completed_exposures':exposure,'frozen_encoder_digest':getattr(model,'_eval_encoder_digest',None),
              'missing_depth_count':len(failed),'seconds':time.perf_counter()-t0,'optimizer_updates':0,'no_grad':True,
              'job_id':os.environ.get('SLURM_JOB_ID'),'eval_git_sha':eval_sha}
            (root/'worker_done.json').write_text(json.dumps(done,indent=2)+'\n');work.append(done)
        del model,opt,blob
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps({'status':'PASS','rank':rank,'work':work},indent=2),flush=True)

if __name__=='__main__': main()
