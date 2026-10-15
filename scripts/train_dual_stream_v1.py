"""Short learning gate -> full four-epoch training, one resumable frozen-backbone run."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import time
import traceback
from datetime import timedelta
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.dual_stream_training_policy import CONFIG,epoch_order,rank_microbatches,lr_factor,acceptance
from scripts.dual_stream_frozen_cache import install_frozen_caches,cpu_tree,atomic_torch_save

MANIFEST=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
HELDOUT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')
FROZEN=('frozen_omega.','sam_proposals.')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family(name):
    if name.startswith('understanding.'):return 'understanding'
    if name.startswith('query_initializer.'):return 'queries'
    if name.startswith('panoptic.'):return 'panoptic'
    if name.startswith(('anchor_decoder.refine_mu.','anchor_decoder.refine_rho.')):return 'geometry'
    return 'reconstruction'


def optimizer_groups(model):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        f=family(name);lr=CONFIG['geometry_refinement_lr'] if f=='geometry' else CONFIG['understanding_lr'] if f in ('understanding','queries','panoptic') else CONFIG['reconstruction_lr']
        wd=0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else CONFIG['weight_decay']
        key=(f,lr,wd)
        groups.setdefault(key,[]).append(p)
    return [dict(params=ps,lr=lr,base_lr=lr,weight_decay=wd,family=f) for (f,lr,wd),ps in groups.items()]


def cohorts():
    raw=json.loads(MANIFEST.read_text())
    windows=raw.get('expanded_train_windows') or raw.get('train_windows') or raw['windows']
    assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    # Fixed, prediction-independent choice: 16 distinct training scenes, >=2 instances.
    rng=random.Random(CONFIG['seed']); candidates=list(range(len(windows)));rng.shuffle(candidates)
    candidates=[i for i in candidates if windows[i].get('thing_instance_count',0)>=2]
    chosen=[]; seen=set(); covered=set()
    while len(chosen)<16:
        choices=[i for i in candidates if windows[i]['scene'] not in seen]
        best=max(choices,key=lambda i:len(set(windows[i].get('semantic_classes_context',[]))-covered))
        chosen.append(best);seen.add(windows[best]['scene']);covered.update(windows[best].get('semantic_classes_context',[]))
    records=json.loads(HELDOUT.read_text())['records']; validation=[]
    for r in records:
        pair=r['official_pair']
        validation.append(dict(scene=pair['scene'],context=pair['context'],novel=[f for f in r['frame_ids'] if f not in pair['context']],
            source_index=r['source_index'],expected_rgb_sha256=r['input_rgb_sha256']))
    assert len(validation)==128 and not {w['scene'] for w in validation}&{w['scene'] for w in windows}
    return windows,chosen,validation


def build_batch(opt,window,device):
    from scripts.object_locus_v3_set_runtime import build_batch as original
    # Pin data RNG without resetting model RNG / dropout clocks.
    py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state()
    seed=42+int(window.get('source_index',window.get('index',0)))
    random.seed(seed);np.random.seed(seed);torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    try: batch=original(opt,window,device)
    finally:random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)
    if 'expected_rgb_sha256' in window:
        actual=hashlib.sha256(batch['images_all'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        if actual!=window['expected_rgb_sha256']:raise ValueError('Held-out RGB differs from old fixed cohort')
    return batch


@torch.no_grad()
def quality(prediction):
    pairs=prediction['final_pairs'];targets=prediction['final_targets'];regions=prediction['region_mass']
    soft=[];binary=[];correct_nonempty=[]
    for bi,(qi,ki) in enumerate(pairs):
        if not qi.numel():continue
        valid=targets['valid_pixels'][bi].flatten()
        p=regions[bi,:,:100].permute(1,0,2,3).reshape(100,-1)[qi][:,valid]
        y=targets['gt_pixel_masks'][bi][ki].flatten(1)[:,valid].float()
        inter=(p*y).sum(-1);soft.extend((inter/(p.sum(-1)+y.sum(-1)-inter).clamp_min(1e-6)).tolist())
        binary_p=p>.5;inter=(binary_p*y).sum(-1)
        binary.extend((inter/(binary_p.sum(-1)+y.sum(-1)-inter).clamp_min(1)).tolist())
        correct=prediction['thing_logits19'][bi,qi].argmax(-1)==targets['gt_classes'][bi][ki]-2
        correct_nonempty.extend((correct&(binary_p.sum(-1)>=16)).float().tolist())
    return dict(matched_soft_iou=float(np.mean(soft)) if soft else 0.,matched_binary_iou=float(np.mean(binary)) if binary else 0.,
        correct_class_nonempty_fraction=float(np.mean(correct_nonempty)) if correct_nonempty else 0.)


def scalar_metrics(metrics,prediction):
    keep=('loss','loss_recon','loss_understanding','psnr','loss_classification','loss_thing_bce','loss_thing_dice','loss_stuff_bce','loss_stuff_dice','matched_classification_accuracy')
    result={k:float(metrics[k].detach()) if torch.is_tensor(metrics[k]) else float(metrics[k]) for k in keep}
    result.update(quality(prediction))
    if not all(math.isfinite(v) for v in result.values()):raise FloatingPointError('Nonfinite training/monitor metric')
    return result


class Task(nn.Module):
    """Only return the live loss to DDP's unused-parameter traversal."""
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,batch,step):
        output,metrics=self.model.step_loss(batch,step=step,understanding_weight=1.)
        return metrics['loss'],scalar_metrics(metrics,output['prediction'])


def save_preview(model,batch,prediction,path):
    from PIL import Image,ImageDraw
    rows=[]
    for vi in range(2):
        panels=[]
        for image in (batch['images_input'][0,vi],prediction['render']['images_pred'][0,vi]):
            panels.append(Image.fromarray((image.detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255+.5).astype(np.uint8)))
        pairs=prediction['final_pairs'][0];target=prediction['final_targets']
        if pairs[0].numel():
            gt=target['gt_pixel_masks'][0];index=int(gt.flatten(1).sum(1).argmax());where=(pairs[1]==index).nonzero()
            if where.numel():
                qi=int(pairs[0][int(where[0])]);truth=gt[index,vi].float();mask=prediction['region_mass'][0,vi,qi]
                for m in (truth,mask):panels.append(Image.fromarray((m.detach().cpu().clamp(0,1).numpy()*255+.5).astype(np.uint8)).convert('RGB'))
        row=Image.new('RGB',(256*len(panels),280),'white')
        for i,image in enumerate(panels):row.paste(image,(256*i,24))
        ImageDraw.Draw(row).text((4,4),'Context RGB | rendered RGB | largest GT mask | matched raw mask',fill='black')
        rows.append(row)
    canvas=Image.new('RGB',(max(x.width for x in rows),sum(x.height for x in rows)),'white')
    y=0
    for row in rows:canvas.paste(row,(0,y));y+=row.height
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def monitor(model,opt,train_windows,validation_windows,root,label,rank,world,device,step):
    from tokengs.models.input_types import ModelInputDecoder
    model.eval(); rows=[]
    with torch.no_grad():
        for scope,windows in [('train',train_windows),('validation',validation_windows)]:
            for index in range(rank,len(windows),world):
                batch=build_batch(opt,windows[index],device)
                output,metrics=model.step_loss(batch,step=step,understanding_weight=1.)
                prediction=output['prediction'];row=scalar_metrics(metrics,prediction)
                calibration=prediction['target_camera_calibration']
                render=model.render_reconstruction(prediction['reconstruction'],ModelInputDecoder(cam_view=torch.linalg.inv(calibration['c2w']).transpose(-1,-2),intrinsics=calibration['intrinsics']))
                for name,sl in [('context',slice(0,2)),('true_novel',slice(2,None))]:
                    mse=(render['images_pred'][:,sl]-batch['images_all'][:,sl]).square().flatten(2).mean(-1)
                    row[name+'_psnr']=float((-10*torch.log10(mse.clamp_min(1e-12))).mean())
                rows.append(dict(scope=scope,index=index,scene=windows[index]['scene'],metrics=row))
                if index<2:save_preview(model,batch,prediction,root/'previews'/label/f'{scope}_{index:02d}.png')
                del output,prediction,metrics,render,batch
    gathered=[None]*world;dist.all_gather_object(gathered,rows)
    allrows=[r for group in gathered for r in group]
    report={}
    for scope in ('train','validation'):
        selected=[r['metrics'] for r in allrows if r['scope']==scope]
        assert len(selected)==8
        report[scope]={key:sum(r[key] for r in selected)/len(selected) for key in selected[0]}
    if rank==0:write(root/'monitor'/f'{label}.json',dict(summary=report,windows=allrows,diagnostic_only=True,step=step))
    model.train();torch.cuda.empty_cache()
    return report


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def trainable_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith(FROZEN)}


def restore_trainable_state(model,state):
    target=model.state_dict();expected={k for k in target if not k.startswith(FROZEN)}
    if set(state)!=expected:raise ValueError('Checkpoint key set differs from explicit non-frozen whitelist')
    with torch.no_grad():
        for k in expected:
            if state[k].shape!=target[k].shape:raise ValueError('Checkpoint shape mismatch: '+k)
            target[k].copy_(state[k])


def checkpoint(model,optimizer,progress,root,meta,rank,world,retain=None):
    flags=torch.stack([torch.isfinite(p).all() for p in model.parameters() if p.requires_grad])
    if not bool(flags.all()):raise FloatingPointError('Nonfinite trainable parameter before checkpoint')
    states=[None]*world;dist.all_gather_object(states,rng_state())
    if rank==0:
        payload=dict(model=trainable_state(model),optimizer=cpu_tree(optimizer.state_dict()),progress=progress,rank_rng=states,meta=meta)
        target=root/'checkpoint_latest.pt';atomic_torch_save(payload,target)
        if retain:
            retained=root/retain
            if retained.exists():raise ValueError('Refusing overwrite of retained checkpoint')
            os.link(target,retained)
        write(root/'checkpoint_latest.json',dict(progress=progress,recipe=CONFIG['recipe'],checkpoint=str(target),retained=retain))
    dist.barrier()


def run(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world not in (1,2,4,8) or 8%world:raise ValueError('Unsupported world size')
    torch.cuda.set_device(local);torch.set_num_threads(4);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    if rank==0:
        if (root/'STARTED.json').exists() and not args.resume:raise ValueError('Existing attempt; use explicit --resume')
        write(root/'STARTED.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),world_size=world,pid=os.getpid(),config=dict(CONFIG,recipe='omega_gfm_two_adapter_sam_pool_v2_frozen_joint4') if args.architecture=='gfm_v2' else CONFIG,architecture=args.architecture))
    from tokengs.options import config_defaults
    from tokengs.models.object_locus_dual_stream_v1 import LocusGSDualStreamV1
    source_config=dict(CONFIG)
    if args.architecture=='gfm_v2':
        from tokengs.models.object_locus_gfm_dual_stream_v2 import LocusGSGFMDualStreamV2
        CONFIG['recipe']='omega_gfm_two_adapter_sam_pool_v2_frozen_joint4'
        CONFIG['required_gradient_families']=['reconstruction','understanding','queries','panoptic','gfm_reconstruction_adapter','gfm_query_projection','anchor_gfm_content']
        opt=config_defaults['train_siu3r_object_locus_gfm_dual_stream_v2']
        constructor=LocusGSGFMDualStreamV2
    else:
        opt=config_defaults['train_siu3r_object_locus_dual_stream_v1'];constructor=LocusGSDualStreamV1
    windows,short_indices,validation=cohorts();short=[windows[i] for i in short_indices]
    model=constructor(opt,write_strength=0.).to(device)
    cache=install_frozen_caches(model,args.cache_root or root/'frozen_cache')
    optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8)
    task=DDP(Task(model),device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    meta=dict(recipe=CONFIG['recipe'],config=CONFIG,world_size=world,global_batch=8,accumulation=8//world,
        execution_git_sha=sha,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),
        train_windows=len(windows),train_scenes=len({w['scene'] for w in windows}),short_indices=short_indices,
        frozen_cache_identity=cache['identity'])
    progress=dict(phase='short',short_updates=0,full_updates=0,epoch=0,update_in_epoch=0,gate_passed=False,
        epoch_counts=[0]*len(windows),total_counts=[0]*len(windows),proof=dict(finite=True,updated_families={}))
    if args.resume:
        blob=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
        if 'architecture_transition' in blob['meta']:meta['architecture_transition']=blob['meta']['architecture_transition']
        if blob['meta']!=meta:raise ValueError('Resume metadata/code/data/config mismatch')
        restore_trainable_state(model,blob['model']);optimizer.load_state_dict(blob['optimizer']);progress=blob['progress'];restore_rng(blob['rank_rng'][rank]);del blob
    if args.transition_from:
        if args.resume or args.architecture!='gfm_v2':raise ValueError('Architecture transfer requires a fresh GFM V2 attempt')
        from scripts.gfm_dual_stream_transition import transfer,verify_source,SOURCE50_SHA256
        valid=[verify_source(args.transition_from) if rank==0 else None]
        dist.broadcast_object_list(valid,src=0)
        if not valid[0]:raise ValueError('Preserved source50 checkpoint checksum differs')
        blob=torch.load(args.transition_from,map_location='cpu',weights_only=False,mmap=True)
        # Full source SHA is checked by prefix and all remaining scientific metadata exactly.
        oldmeta=blob['meta']
        ignored={'world_size','accumulation','execution_git_sha','recipe','config'}
        if any(oldmeta.get(k)!=v for k,v in meta.items() if k not in ignored):raise ValueError('Transition data/asset identity differs')
        if oldmeta['config']!=source_config or oldmeta['recipe']!=source_config['recipe'] or oldmeta['world_size']!=2 or oldmeta['accumulation']!=4 or not oldmeta['execution_git_sha'].startswith('7d38d04'):
            raise ValueError('Wrong explicit V1 source')
        if blob['progress']['short_updates']!=50 or blob['progress']['full_updates']!=0:raise ValueError('Expected preserved short update50 boundary')
        plan=transfer(model,optimizer,blob,optimizer_groups)
        progress=blob['progress'];progress['proof']=dict(finite=True,updated_families={})
        if rank<2:restore_rng(blob['rank_rng'][rank])
        else:
            seed=CONFIG['seed']+50000+rank
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        meta['architecture_transition']=dict(source_checkpoint=str(args.transition_from.resolve()),source_git_sha=oldmeta['execution_git_sha'],source_checkpoint_sha256=SOURCE50_SHA256,source_world_size=2,target_world_size=world,
            source_completed_short_updates=50,retained_optimizer_states=plan['retained_optimizer_states'],new_short_checks=[178,306],new_baseline_at_source_update=50)
        if rank==0:write(root/'ARCHITECTURE_TRANSITION.json',dict(**meta['architecture_transition'],**plan,optimizer_moments_retained_by_parameter_name=True,new_parameters_have_fresh_moments=True))
        del blob
    short_origin=meta.get('architecture_transition',{}).get('source_completed_short_updates',0)
    short_checks=[short_origin+step for step in CONFIG['short_checks']]
    frozen_versions={n:p._version for n,p in model.named_parameters() if n.startswith(FROZEN)}
    probes={}
    for name,p in model.named_parameters():
        f=family(name)
        if p.requires_grad and f not in probes:probes[f]=(name,p.detach().clone())
    if args.architecture=='gfm_v2':
        named=dict(model.named_parameters())
        for label,name in [('gfm_reconstruction_adapter','reconstruction_adapter.projection.0.weight'),('gfm_query_projection','query_initializer.content_projection.1.weight'),('anchor_gfm_content','anchor_visual_projection.1.weight')]:
            probes[label]=(name,named[name].detach().clone())
    if rank==0:
        write(root/'configuration.json',dict(**meta,optimizer_groups=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups],
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gpu=torch.cuda.get_device_name(),short_windows=short,monitor_validation=validation[:8]))
        print('MODEL_AND_OPTIMIZER_READY',json.dumps(dict(world_size=world,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))),flush=True)
    baseline_path=root/'monitor'/'initial.json'
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text())['summary']
    else:baseline=monitor(model,opt,short[:8],validation[:8],root,'initial',rank,world,device,(progress['short_updates']+progress['full_updates'])*8)
    if not args.resume:checkpoint(model,optimizer,progress,root,meta,rank,world,retain='checkpoint_initial.pt')

    def update(indices,phase,update_index,total):
        model.train();optimizer.zero_grad(set_to_none=True)
        factor=lr_factor(phase,update_index-short_origin if phase=='short' else update_index,total-short_origin if phase=='short' else total)
        for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
        rows=[];started=time.monotonic()
        for micro,index in enumerate(indices):
            window=short[index] if phase=='short' else windows[index]
            batch=build_batch(opt,window,device)
            sync=contextlib.nullcontext() if micro==len(indices)-1 else task.no_sync()
            exposure=(progress['short_updates']+progress['full_updates'])*8
            with sync:
                loss,metrics=task(batch,exposure)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                (loss/len(indices)).backward()
            rows.append(metrics);del loss,batch
        grad_norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],CONFIG['clip_norm'],error_if_nonfinite=True)
        parameters=dict(model.named_parameters())
        probe_gradients={f:float(parameters[name].grad.detach().norm()) if parameters[name].grad is not None else 0. for f,(name,_) in probes.items()}
        optimizer.step()
        # Meaningful value changes in representative tensors, rather than optimizer counters alone.
        for f,(name,before) in probes.items():
            p=parameters[name]
            if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite representative parameter '+name)
            changed=(not torch.equal(p.detach(),before)) and probe_gradients[f]>0
            progress['proof']['updated_families'][f]=progress['proof']['updated_families'].get(f,False) or changed
        row={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
        row.update(phase=phase,phase_update=update_index+1,grad_norm=float(grad_norm),seconds=time.monotonic()-started,
            cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30,learning_rate_factor=factor,local_indices=indices,representative_grad_norms=probe_gradients)
        append(root/f'train_rank{rank:02d}.jsonl',row)
        if rank==0 and (update_index<3 or (update_index+1)%10==0):print('TRAIN',json.dumps(row),flush=True)
        return row

    if progress['phase']=='short':
        for step in range(progress['short_updates'],max(short_checks)):
            order=list(range(16));offset=(step*8)%16
            indices=[order[(offset+micro*world+rank)%16] for micro in range(8//world)]
            row=update(indices,'short',step,max(short_checks));progress['short_updates']=step+1
            if step==short_origin or (step+1)%CONFIG['checkpoint_every_updates']==0:
                checkpoint(model,optimizer,progress,root,meta,rank,world)
                if rank==0 and step==short_origin:write(root/'FIRST_UPDATE.json',dict(parameter_updates=step+1,new_structure_updates=1,source_completed_updates=short_origin,proof=progress['proof'],metrics=row))
            if step+1 in short_checks:
                report=monitor(model,opt,short[:8],validation[:8],root,f'short_{step+1:04d}',rank,world,device,(step+1)*8)
                proof_rows=[None]*world;dist.all_gather_object(proof_rows,progress['proof'])
                global_proof=dict(finite=all(p['finite'] for p in proof_rows),updated_families={f:all(p['updated_families'].get(f,False) for p in proof_rows) for f in CONFIG.get('required_gradient_families',('reconstruction','understanding','queries','panoptic'))})
                gate=acceptance(baseline['train'],report['train'],baseline['validation'],report['validation'],global_proof)
                if rank==0:write(root/f'gate_update{step+1:04d}.json',gate)
                if gate['passed']:
                    progress.update(phase='full',gate_passed=True)
                    checkpoint(model,optimizer,progress,root,meta,rank,world,retain='checkpoint_short_passed.pt')
                    if rank==0:write(root/'SHORT_PASSED.json',dict(short_updates=step+1,gate=gate,automatic_full_training_authorized=True))
                    break
        if not progress['gate_passed']:
            checkpoint(model,optimizer,progress,root,meta,rank,world,retain='checkpoint_short_gate_failed.pt')
            if rank==0:write(root/'GATE_FAILED.json',dict(short_updates=progress['short_updates'],action='stopped_without_full_training',gate=gate))
            dist.barrier();dist.destroy_process_group();raise SystemExit(3)

    full_total=math.ceil(len(windows)/8)*CONFIG['epochs']
    for epoch in range(progress['epoch'],CONFIG['epochs']):
        order=epoch_order(len(windows),epoch);updates=len(order)//8
        for step in range(progress['update_in_epoch'],updates):
            indices=rank_microbatches(order,step,rank,world)
            row=update(indices,'full',progress['full_updates'],full_total)
            for index in order[step*8:(step+1)*8]:
                progress['epoch_counts'][index]+=1;progress['total_counts'][index]+=1
            progress['full_updates']+=1;progress['update_in_epoch']=step+1
            count=progress['full_updates']
            if count%CONFIG['checkpoint_every_updates']==0:checkpoint(model,optimizer,progress,root,meta,rank,world)
            if count==1 and rank==0:write(root/'FULL_STARTED.json',dict(full_updates=1,short_updates=progress['short_updates'],metrics=row))
            if count%CONFIG['full_monitor_every_updates']==0:
                monitor(model,opt,short[:8],validation[:8],root,f'full_{count:05d}',rank,world,device,(progress['short_updates']+count)*8)
        if not all(x>=1 for x in progress['epoch_counts']):raise ValueError('Full epoch skipped a window')
        if sum(progress['epoch_counts'])!=updates*8:raise ValueError('Full epoch exposure clock differs')
        epoch_counts=progress['epoch_counts']
        progress.update(epoch=epoch+1,update_in_epoch=0,epoch_counts=[0]*len(windows))
        checkpoint(model,optimizer,progress,root,meta,rank,world,retain=f'checkpoint_epoch_{epoch+1:02d}.pt')
        report=monitor(model,opt,short[:8],validation[:8],root,f'epoch_{epoch+1:02d}',rank,world,device,(progress['short_updates']+progress['full_updates'])*8)
        if rank==0:write(root/f'epoch_{epoch+1:02d}.json',dict(epoch=epoch+1,summary=report,counts=epoch_counts,full_updates=progress['full_updates']))
    if any(p._version!=frozen_versions[n] for n,p in model.named_parameters() if n in frozen_versions):raise RuntimeError('Frozen Omega/SAM changed')
    if rank==0:write(root/'TRAINING_COMPLETE.json',dict(status='FULL_FOUR_EPOCHS_COMPLETE',short_updates=progress['short_updates'],full_updates=progress['full_updates'],
        full_exposures=progress['full_updates']*8,full_window_unique_count=sum(x>0 for x in progress['total_counts']),frozen_unchanged=True,cache=cache))
    dist.barrier()
    from scripts.dual_stream_official_export import export
    export(model,opt,validation,root/'evaluation',rank,world,device)
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--architecture',choices=('v1','gfm_v2'),default='v1')
    parser.add_argument('--transition-from',type=Path)
    parser.add_argument('--cache-root',type=Path)
    args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        if not isinstance(exc,SystemExit) or exc.code not in (0,3):
            write(args.output_dir/f"FAILED_rank{os.environ.get('RANK','unknown')}.json",dict(error=repr(exc),traceback=traceback.format_exc(),job_id=os.environ.get('SLURM_JOB_ID')))
        raise

if __name__=='__main__':main()
