"""Independent no-grad evaluation using the unchanged registered evaluator."""
import argparse, csv, dataclasses, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scripts import object_locus_panoptic_full1201_runtime as rt
from scripts import eval_object_locus_panoptic_v1 as evaluator

SCOPES = ('context','target_all','novel')
EPOCHS = (0,1,2,4,6,8)
EVAL = rt.REPORT/'evaluation'

def write(path, value): rt.base.write_json(Path(path),value)
def read(path): return json.loads(Path(path).read_text())
def window_hash(windows): return hashlib.sha256(json.dumps(windows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def code_sha(): return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def csvout(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)

def full_windows():
    registration=read(rt.REPORT/'deferred_evaluation_plan.json')
    source=Path('/space/mawb/SIU3R/data/scannet/val_pair.json')
    assert rt.sha(source)==registration['full_manifest_sha256']==rt.sha(rt.REPORT/'full_validation_manifest.json')
    result=[]
    for i,r in enumerate(read(source)):
        context=list(map(int,r['context_ids']));target=list(map(int,r['target_ids']))
        assert len(context)==2 and set(context)<=set(target)
        result.append(dict(scene=r['scan'],context=context,novel=[f for f in target if f not in context],
                           target=target,pair_iou=r.get('iou'),official_index=i,camera_source='GT camera poses'))
    assert len(result)==1860 and len({w['scene'] for w in result})==312
    assert len({(w['scene'],tuple(w['context'])) for w in result})==len(result)
    return result

def panel(batch,out,window,path,title):
    from scripts.export_object_locus_panoptic_v1_official import assemble_panoptic
    sem,ins,_=assemble_panoptic(out); p=out['p_class'][0];score,classes=p[:,:18].max(-1)
    eligible=(p.argmax(-1)!=18)&(score>=.05);queries=torch.argsort(score,descending=True,stable=True)[:5].tolist()
    size=144;cols=12;views=batch['images_all'].shape[1]
    canvas=Image.new('RGB',(cols*size,55+views*(size+22)),'white');draw=ImageDraw.Draw(canvas);draw.text((3,3),title+' | first two rows=context; remaining=true novel',fill='black')
    def palette(s,ids=None):
        s=s.detach().cpu().numpy();rgb=np.zeros((*s.shape,3),np.uint8)
        for c in range(21):rgb[s==c]=((37*c+53)%255,(97*c+31)%255,(173*c+71)%255)
        if ids is not None:
            ids=ids.detach().cpu().numpy()
            for iid in np.unique(ids):
                if iid>0:rgb[ids==iid]=((31*int(iid)+71)%255,(67*int(iid)+137)%255,(131*int(iid)+29)%255)
        return rgb
    for v in range(views):
        rgb=lambda x:(x.detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        gtsem=batch['semantic_label_all'][0,v];gtins=batch['instance_label_all'][0,v]
        predsem=out['semantic_scores'][0,v].argmax(0);predsem=torch.where(out['alpha'][0,v,0]>.05,predsem,20)
        overlay=np.zeros((256,256,3),np.uint8)
        for q in range(100):
            if bool(eligible[q]):
                mask=((out['region_mass'][0,v,q]>=.5)&(out['alpha'][0,v,0]>.05)).cpu().numpy();overlay[mask]=((31*(q+1)+71)%255,(67*(q+1)+137)%255,(131*(q+1)+29)%255)
        images=[rgb(batch['images_all'][0,v]),rgb(out['render']['images_pred'][0,v]),palette(gtsem),palette(predsem),palette(gtsem,gtins),palette(sem[v],ins[v]),overlay]
        labels=['GT RGB','pred RGB','GT semantic','pred semantic','GT panoptic','pred panoptic','eligible candidates']
        for q in queries:
            mask=((out['region_mass'][0,v,q]>=.5)&(out['alpha'][0,v,0]>.05)).cpu().numpy();images.append(np.repeat((mask*255).astype(np.uint8)[...,None],3,axis=-1));labels.append(f'q{q} c{int(classes[q])+2} s{float(score[q]):.3f} '+('eligible' if bool(eligible[q]) else 'ineligible'))
        for col,(image,label) in enumerate(zip(images,labels)):
            y=55+v*(size+22);canvas.paste(Image.fromarray(image).resize((size,size)),(col*size,y));draw.text((col*size+2,y-18),label,fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def load(epoch,device):
    assert read(rt.REPORT/'training_verification.json')['status']=='PASS'
    model,opt=rt.base.build_model(device,report=False)
    record=next(r for r in read(rt.REPORT/'training_verification.json')['checkpoint_records'] if r['epoch']==epoch)
    assert rt.sha(record['path'])==record['sha256']
    blob=torch.load(record['path'],map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(blob['model'],strict=True);model.understanding_step=blob.get('completed_exposures',blob['completed_updates']*8)
    exposure_basis='completed_exposures' if 'completed_exposures' in blob else 'completed_updates*8'
    del blob; model.eval()
    return model,opt,record,exposure_basis

def run_split(model,opt,record,windows,split,root,device,basis):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    source=dict(checkpoint_sha256=record['sha256'],window_sha256=window_hash(windows),evaluator_sha256=rt.sha(Path(evaluator.__file__)),
                runner_sha256=rt.sha(Path(__file__)),evaluation_sha=code_sha(),training_sha=read(rt.REPORT/'training_verification.json')['training_sha'],
                epoch=record['epoch'],completed_updates=record['updates'],understanding_step=record['exposures'],exposure_basis=basis,
                optimizer_updates=0,backward=0,eval_mode=True,no_grad=True,distributed=False,config=dataclasses.asdict(opt),
                official_commit='8ea80166be76854f938e90521f1a5b688b755c87',job_id=os.environ.get('SLURM_JOB_ID'))
    source=rt.base.jsonable(source)
    done=root/'complete.json'
    if done.exists():
        old=read(done);oldsource=old['source']; comparison={k:v for k,v in source.items() if k not in ('job_id','evaluation_sha')}
        assert {k:v for k,v in oldsource.items() if k not in ('job_id','evaluation_sha')}==comparison
        for name,digest in old['files'].items():assert rt.sha(root/name)==digest
        print('REUSED '+str(root),flush=True);return
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    ssim=StructuralSimilarityIndexMeasure(sync_on_compute=False).to(device)
    lpips=LearnedPerceptualImagePatchSimilarity('vgg',normalize=True,sync_on_compute=False).to(device).eval()
    original_run=evaluator._run;original_stats=evaluator._candidate_stats;original_factory=evaluator.local_ap_metric;original_panel=evaluator.write_panel
    extras=[];stats=[];metrics=[];start=time.time()
    def metric_factory():
        m=original_factory();metrics.append(m);return m
    def capture(model,opt,window,builder,device):
        assert not torch.is_grad_enabled() and not model.training
        batch,out=original_run(model,opt,window,builder,device)
        frames=batch['frame_ids'][0].cpu().tolist();indices={'context':[0,1],'target_all':list(range(len(frames))),'novel':[i for i,f in enumerate(frames) if f in window['novel']]}
        quality={}
        for scope,idx in indices.items():
            a=out['render']['images_pred'][0,idx].clamp(0,1);b=batch['images_all'][0,idx].clamp(0,1)
            # Same SSIM/LPIPS implementation/settings as pinned official image metrics;
            # average individual frames, leaving the existing local PSNR unchanged.
            values=[]
            for av,bv in zip(a,b):
                ssim.reset();lpips.reset();values.append((float(ssim(av[None],bv[None])),float(lpips(av[None],bv[None]))))
            quality[scope]=dict(ssim=float(np.mean([v[0] for v in values])),lpips=float(np.mean([v[1] for v in values])))
        extras.append(dict(scene=window['scene'],context=window['context'],novel=window['novel'],quality=quality))
        print(f"LOCAL {split} {len(extras)}/{len(windows)} {window['scene']}",flush=True)
        return batch,out
    def capture_stats(out,batch,idx):
        row=original_stats(out,batch,idx)
        from scripts.eval_object_locus_v3_set import _targets
        sem=batch['semantic_label_all'][0,list(idx)].long();ins=batch['instance_label_all'][0,list(idx)].long();valid,targets=_targets(sem,ins)
        raw=(out['region_mass'][0,list(idx),:100]>=.5)&(out['alpha'][0,list(idx),0][:,None]>.05)
        for item,(_,_,gtmask) in zip(row['per_gt'],targets):
            pred=raw[:,item['best_query']]&valid;intersection=int((pred&gtmask).sum());pa=int(pred.sum());ga=int(gtmask.sum())
            qinter=((raw&valid[:,None])&gtmask[:,None]).sum((0,2,3));qarea=(raw&valid[:,None]).sum((0,2,3));qiou=qinter/(qarea+ga-qinter).clamp_min(1)
            eligible=torch.tensor([q['independent_candidate'] for q in row['query_rows']],device=raw.device)
            item.update(raw_prediction_area=pa,gt_area=ga,raw_intersection=intersection,raw_precision=intersection/pa if pa else 'UNDEFINED',raw_recall=intersection/ga if ga else 'UNDEFINED',
                        eligible_queries_iou_ge_0_5=int(((qiou>=.5)&eligible).sum()))
        stats.append({k:row[k] for k in ('semantic_confusion','panoptic_semantic_confusion','classification_confusion','gt_count','matched_gt_count','raw_best_ious','panoptic_pq','candidate_ca','candidate_cw','panoptic_ca','panoptic_cw')})
        # The registered evaluator excludes these intermediates from its saved
        # result. Retaining `predictions` until the final reduction holds CUDA
        # masks for every preceding window, exhausting memory on long splits.
        # All metric updates use _map_payload; aggregation uses the scalars,
        # confusion matrices and per-GT rows retained below.
        for key in ('predictions','raw_masks','panoptic_semantic','panoptic_instance','gt_semantic','gt_instance'):
            row.pop(key, None)
        return row
    evaluator._run=capture;evaluator._candidate_stats=capture_stats;evaluator.local_ap_metric=metric_factory;evaluator.write_panel=panel
    try:
        result,gt,queries=evaluator.evaluate_windows(model,opt,windows,record['updates'],split,root,device,rt.base.build_batch,official=True,panels=not split.startswith('full_'))
    finally:
        evaluator._run=original_run;evaluator._candidate_stats=original_stats;evaluator.local_ap_metric=original_factory;evaluator.write_panel=original_panel
    assert len(extras)==len(windows) and len(stats)==len(windows)*3 and len(metrics)==3
    for i,w in enumerate(result['windows']):
        for j,scope in enumerate(SCOPES):w['scopes'][scope].update(extras[i]['quality'][scope]);w['scopes'][scope].update(stats[3*i+j])
    for scope in SCOPES:
        for key in ('ssim','lpips'):result['local'][scope][key]=float(np.mean([x['quality'][scope][key] for x in extras]))
        local=result['local'][scope];conf=np.asarray(local['classification_confusion']);den=int(conf.sum());no=int(conf[:,18].sum())
        local.update(matched_no_object_count=no,matched_denominator=den,matched_no_object_fraction=no/den if den else 'UNDEFINED',
                     matching_source='context final_hungarian; repeated for each render scope, NOT novel-only classification')
    for row in gt:row['matching_source']='context final_hungarian';row['epoch']=record['epoch']
    write(root/'result.json',result);write(root/'source.json',source);csvout(root/'per_gt.csv',gt);csvout(root/'queries.csv',queries)
    states={scope:{k:getattr(metric.cpu(),k) for k in metric._defaults} for scope,metric in zip(SCOPES,metrics)}
    torch.save(states,root/'candidate_ap_states.pt')
    files={name:rt.sha(root/name) for name in ('result.json','source.json','per_gt.csv','queries.csv','candidate_ap_states.pt')}
    write(done,dict(status='COMPLETE',source=source,windows=len(windows),files=files,seconds=time.time()-start))
    print('COMPLETE '+str(root),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','monitor','full']);p.add_argument('--epoch',type=int,default=8);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8);args=p.parse_args()
    if args.mode=='monitor' and 'SLURM_ARRAY_TASK_ID' in os.environ:args.epoch=EPOCHS[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    torch.set_num_threads(4);assert not torch.distributed.is_initialized();assert os.uname().nodename.startswith(('3dimage-13','3dimage-11'))
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090';torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',0);model,opt,record,basis=load(args.epoch,device);manifest=read(rt.REPORT/'manifest.json')
    if args.mode=='smoke':run_split(model,opt,record,manifest['monitor_splits']['dev8'][:1],'evaluator_smoke',EVAL/'smoke',device,basis)
    elif args.mode=='monitor':
        assert (EVAL/'smoke/complete.json').exists()
        for split,windows in manifest['monitor_splits'].items():run_split(model,opt,record,windows,split,EVAL/f'epoch{args.epoch:02}'/split,device,basis)
    else:
        assert read(rt.REPORT/'checkpoint_selection.json')['best_epoch'] in EPOCHS
        windows=full_windows();scenes=sorted({w['scene'] for w in windows});selected=set(scenes[args.shard::args.shards]);windows=[w for w in windows if w['scene'] in selected]
        run_split(model,opt,record,windows,f'full_{args.shard:02}',EVAL/f'full_epoch{args.epoch:02}'/f'shard{args.shard:02}',device,basis)

if __name__=='__main__':main()
