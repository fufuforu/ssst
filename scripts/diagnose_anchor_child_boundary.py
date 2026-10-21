"""Fixed A4/A4+4 context diagnostics. No optimizer, loss or reverse AD."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,hashlib,json,os,time,traceback,subprocess
from unittest.mock import patch
import numpy as np
import torch
from diagnostics.anchor_child_boundary.forward_lift import ForwardLift,no_reverse_ad
from diagnostics.anchor_child_boundary.metrics import feature_metrics,boundary_response,boundary_metrics


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def learned_digest(model,keys):
    h=hashlib.sha256();state=model.state_dict()
    for k in sorted(keys):
        x=state[k].detach().cpu().contiguous();h.update(k.encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def colorize(ids):
    x=ids.astype(np.int64);rgb=np.stack([(x*67+31)%251,(x*131+71)%251,(x*197+111)%251],-1).astype(np.uint8)
    rgb[x==0]=0
    return rgb


def montage(path,rgb,pred,raw,soft,gt):
    from PIL import Image,ImageDraw
    # Fixed scale raw response: clipping for display only; NPZ preserves raw float32.
    heat=np.stack([np.clip(raw/.2,0,1),np.zeros_like(raw),np.zeros_like(raw)],-1)
    soft_rgb=np.stack([soft,soft*.35,np.zeros_like(soft)],-1)
    overlay=rgb*(1-.65*soft[...,None])+np.array([1.,0.,0.])*.65*soft[...,None]
    panels=[rgb,colorize(pred)/255.,heat,soft_rgb,overlay,np.repeat(gt[...,None],3,-1).astype(float)]
    names=['Context RGB','Pred instance (official)','Raw response [0,.2]','Soft boundary [0,1]','Boundary overlay','GT instance boundary']
    h,w=raw.shape;canvas=Image.new('RGB',(w*3,(h+24)*2),'white');draw=ImageDraw.Draw(canvas)
    for k,(panel,name) in enumerate(zip(panels,names)):
        x=k%3*w;y=k//3*(h+24)
        draw.text((x+3,y+3),name,fill='black');canvas.paste(Image.fromarray((np.clip(panel,0,1)*255+.5).astype(np.uint8)),(x,y+24))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def run(args):
    from tokengs.options import config_defaults
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.models.superpoint_structural_variants import LocusGSSuperpointDual
    from scripts.train_unified_query_v1 import restore_state
    from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    from tokengs.models.object_locus_panoptic_v1_lift import RendererTranspose
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);np.random.seed(42)
    config=json.loads(args.config.read_text());stage=args.output
    if args.mode=='full':
        smoke=json.loads((args.config.parent/'smoke/SMOKE_PASSED.json').read_text())
        assert smoke['status']=='SMOKE_PASSED' and smoke['config_sha256']==digest_file(args.config)
        assert smoke['code_commit']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stage.mkdir(parents=True,exist_ok=True)
    assert digest_file(HELDOUT)==config['cohort_sha256'] and digest_file(MANIFEST)==config['manifest_sha256']
    _,_,windows=cohorts();indices=list(range(2 if args.mode=='smoke' else 128))
    for label,source in config['sources'].items():
        assert digest_file(source['checkpoint'])==source['sha256'],'Checkpoint source changed'
        saved=torch.load(source['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        meta=saved['meta'];assert meta['architecture']=='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1'
        assert saved['progress']==dict(epoch=source['epoch'],update_in_epoch=0,full_updates=source['updates'])
        assert meta['heldout_sha256']==config['cohort_sha256'] and meta['manifest_sha256']==config['manifest_sha256']
        opt=config_defaults['train_siu3r_unified_query_v1']
        model=LocusGSSuperpointDual(opt,config=UnifiedQueryConfig(**meta['config'])).to(device)
        assert model.frozen_omega.source_identity==meta['omega_identity']
        assert (model.sam_proposals.source_identity if model.sam_proposals is not None else None)==meta['sam_identity']
        restore_state(model,saved['model']);keys=list(saved['model']);del saved
        model.requires_grad_(False).eval()
        versions={n:t._version for n,t in list(model.named_parameters())+list(model.named_buffers())}
        before=learned_digest(model,keys);records=[]
        for index in indices[rank::world]:
            started=time.monotonic();torch.cuda.reset_peak_memory_stats();window=windows[index]
            capture={}
            def hook(name):
                def fn(module,inputs,output):capture[name]=(inputs[0],output)
                return fn
            handles=[model.core.panoptic.child_mlp.register_forward_hook(hook('residual')),
                     model.core.panoptic.W_res.register_forward_hook(hook('wres'))]
            try:
                with no_reverse_ad():
                    batch=build_batch(opt,window,device)
                    sem=batch['semantic_label_all'][0,:2];ins=batch['instance_label_all'][0,:2]
                    lift=ForwardLift(sem,ins,chunk=config['contribution_chunk'])
                    with patch.object(RendererTranspose,'apply',side_effect=lift.apply):
                        output=model.generate(batch['images_input'])
                    assert not any(p.grad is not None for p in model.parameters())
                    assert output['gaussians'].shape[1]%64==0
                    n=output['gaussians'].shape[1]//64
                    assert torch.equal(output['gaussian_parent_query'],torch.arange(n,device=device).repeat_interleave(64))
                    residual=capture['residual'][1];child,wres=capture['wres']
                    shared=output['states'][-1]['f_anchor']
                    assert shared.shape[1]==n and residual.shape==(1,n,64,256)
                    torch.testing.assert_close(child,(shared[:,:,None]+residual).reshape(1,n*64,256))
                    fused=lift.gate*lift.evidence+(1-lift.gate)*child[0]+.1*wres[0].tanh()
                    q=output['states'][-1]['q'];embedding=model.core.semantic_memory.mask_embedder(q)
                    logits=fused[None]@embedding.transpose(1,2)
                    torch.testing.assert_close(logits,output['gaussian_mask_logits'],rtol=2e-5,atol=2e-5)
                    lift.checks['logit_recomposition_max_abs']=float((logits-output['gaussian_mask_logits']).abs().max())
                    def cpu(x):return x.detach().float().cpu().numpy()
                    stats,anchors=feature_metrics(cpu(shared[0]),cpu(residual[0]),cpu(fused),cpu(output['gaussian_membership'][0]),cpu(lift.contribution),cpu(lift.mass))
                    components={'fm':lift.gate*lift.evidence,'shared':(1-lift.gate)*shared[:,:,None].expand(-1,-1,64,-1).reshape(n*64,256),
                                'residual':(1-lift.gate)*residual.reshape(n*64,256),'wres':.1*wres[0].tanh()}
                    for name,x in components.items():stats['fused_component_'+name+'_mean_norm']=float(x.norm(dim=-1).mean())
                    _,pred,_=assemble_panoptic(output)
                    region=cpu(output['region_mass'][0]);pclass=output['p_class'][0]
                    eligible=((pclass.argmax(-1)!=18)&(pclass[:,:18].max(-1).values>=.05)&output['instance_eligible']).cpu().numpy()
                    if region.shape[-2:]!=(256,256) or region.shape[0]!=2:raise ValueError('Unexpected native output resolution')
                    views=[];folder=stage/label/f'window_{index:03d}_{window["scene"]}'
                    folder.mkdir(parents=True,exist_ok=True)
                    for vi in range(2):
                        raw=boundary_response(region[vi,:n][eligible])
                        bstats,soft,gt=boundary_metrics(raw,cpu(sem[vi]).astype(int),cpu(ins[vi]).astype(int))
                        # Diagnostic reference before NMS; never changes official prediction.
                        raw_all=boundary_response(region[vi,:n])
                        bstats['all_query_interior_raw_response_fraction']=boundary_metrics(raw_all,cpu(sem[vi]).astype(int),cpu(ins[vi]).astype(int))[0]['interior_raw_response_fraction']
                        views.append(bstats)
                        np.savez_compressed(folder/f'context{vi}_boundary.npz',raw=raw.astype(np.float32),soft=soft.astype(np.float32))
                        montage(folder/f'context{vi}_sixpanel.png',cpu(batch['images_input'][0,vi]).transpose(1,2,0),pred[vi].cpu().numpy(),raw,soft,gt)
                    torch.cuda.synchronize()
                    row=dict(label=label,index=index,scene=window['scene'],context=window['context'],feature=stats,boundary=views,
                             equivalence=lift.checks,anchors=anchors,gt_instance_pairs=lift.gt_pairs.cpu().tolist(),
                             input_rgb_sha256=hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest(),
                             source_superpoint_sha256=hashlib.sha256(output['superpoints']['source_superpoint_ids'].cpu().numpy().tobytes()).hexdigest(),
                             gt_sha256=hashlib.sha256(sem.cpu().numpy().tobytes()+ins.cpu().numpy().tobytes()).hexdigest(),
                             seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),backward_calls=0,parameter_updates=0)
                    write(folder/'metrics.json',row);records.append(dict(index=index,path=str(folder/'metrics.json')))
                    print(json.dumps(dict(label=label,index=index,seconds=row['seconds'],peak_gpu_bytes=row['peak_gpu_bytes'],equivalence=lift.checks)),flush=True)
            finally:
                for handle in handles:handle.remove()
            capture.clear();del batch,output,lift,child,residual,shared,wres,fused,components,logits,q,embedding
            torch.cuda.empty_cache()
        after=learned_digest(model,keys)
        assert before==after and all(t._version==versions[n] for n,t in list(model.named_parameters())+list(model.named_buffers()))
        assert all(p.grad is None for p in model.parameters())
        write(stage/f'{label}_rank{rank:02d}_complete.json',dict(records=records,weight_sha256_before=before,weight_sha256_after=after,
              source_sha256=source['sha256'],backward_calls=0,parameter_updates=0,rank=rank,world=world))
        del model;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['smoke','full'],required=True)
    args=p.parse_args()
    try:run(args)
    except Exception:
        write(args.output/f'JOB_FAILED_rank{os.environ.get("RANK","0")}.json',dict(traceback=traceback.format_exc(),parameter_updates=0))
        raise

if __name__=='__main__':main()
